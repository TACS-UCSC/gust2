#!/bin/bash
# Cold-temperature collapse-diagnostics sweep with the per-position mask ON.
#
# Studies *diffusive* collapse under cold sampling (the mean-reversion mode
# that survives the position mask), as opposed to the positional-OOD
# explosion the April no-mask sweeps showed. Per model:
#
#   Stage 1 (parallel, 1 GPU each):  multi-trajectory rollout per temperature
#       with --train_tokens_path (per-position mask) and --log_topk
#       (top-K logit traces for offline diagnostics).
#   Stage 2 (1 GPU, depends on stage 1):  multitraj_survival (EMD-threshold
#       explosion times; needs the GPU for VQ-VAE decode), then the three
#       CPU analyzers in sequence: analyze_logits (per cfg),
#       analyze_logits_aligned, analyze_position_ood (sanity: OOD ≡ 0 with
#       the mask on).
#
# All stages log to wandb project gust2-diagnostics-bridges,
# group <run_name>-<group_tag>, job_type per stage.
#
# Usage:
#   ./scripts/bridges/sweep_diagnostics_cold.sh              # both models
#   ./scripts/bridges/sweep_diagnostics_cold.sh --model sc341
#   ./scripts/bridges/sweep_diagnostics_cold.sh --dry-run
#   ./scripts/bridges/sweep_diagnostics_cold.sh --list

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
TOKENS_BASE="${OCEAN}/experiments/tokens"
VQVAE_BASE="${OCEAN}/experiments/vqvae"
AR_BASE="${OCEAN}/experiments/ar-robust-scaling"
DIAG_BASE="${OCEAN}/experiments/diagnostics"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Sweep config ----------
GROUP_TAG="posmask-cold"
WANDB_PROJECT="gust2-diagnostics-bridges"
N_STEPS=2000
START_FRAME=0
SEED=0
BATCH_SIZE=64

# NOTE: temperatures < 1.0 are DELIBERATE here, overriding the usual
# "rollout sweeps always use --temperature 1.0, greedy is broken" project
# convention. This sweep *studies* diffusive collapse under cold sampling
# with the per-position mask ON. Do not "fix" these back to 1.0.
TEMPERATURES=(0.7 0.8 0.9 1.0 1.2)

# "<vqvae_name>:<run_name>:<n_traj>:<log_topk>"
# sc341 flagship at N=25/K=64 matches the April Derecho multitraj sweep for
# direct comparison; sc917 runs reduced N/K to cap rollout_logits.npz size
# (bytes ≈ N × n_steps × tokens × K × 4).
MODELS=(
    "small-sc341:small-sc341-nsp-s18:25:64"
    "small-sc917:small-sc917-nsp-s34:12:32"
)

# ---------- Parse args ----------
DRY_RUN=false
FILTER_MODEL=""
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --model) FILTER_MODEL="$2"; shift 2 ;;
        --list) LIST_ONLY=true; shift ;;
        --help|-h)
            cat <<EOF
Usage: $0 [--model <substr>] [--dry-run] [--list]
  --model <substr>   Filter models by substring (e.g. sc341).
  --dry-run          Print actions without submitting.
  --list             Print the job matrix and exit.
EOF
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ "${LIST_ONLY}" = true ]; then
    echo "Cold-temperature diagnostics sweep (${GROUP_TAG}):"
    echo ""
    printf "  %-14s %-24s %7s %6s\n" "VQ" "NSP run" "n_traj" "log_K"
    printf "  %-14s %-24s %7s %6s\n" "--" "-------" "------" "-----"
    for spec in "${MODELS[@]}"; do
        IFS=':' read -r vq run ntraj topk <<< "${spec}"
        printf "  %-14s %-24s %7s %6s\n" "${vq}" "${run}" "${ntraj}" "${topk}"
    done
    echo ""
    echo "Temperatures: ${TEMPERATURES[*]}  (cold sweep — deliberate, see header)"
    echo "Rollout:      ${N_STEPS} steps, start_frame=${START_FRAME}, posmask ON"
    echo "Jobs/model:   ${#TEMPERATURES[@]} rollout (6h) + 1 diagnostics (8h)"
    echo "Wandb:        ${WANDB_PROJECT}, group=<run>-${GROUP_TAG}"
    exit 0
fi

echo "=========================================="
echo "Cold-temp posmask diagnostics sweep"
echo "  Models:        ${#MODELS[@]} (filter: '${FILTER_MODEL:-none}')"
echo "  Temperatures:  ${TEMPERATURES[*]}"
echo "  Output base:   ${DIAG_BASE}/${GROUP_TAG}"
echo "  Wandb project: ${WANDB_PROJECT}"
echo "  Dry run:       ${DRY_RUN}"
echo "=========================================="

N_SUBMITTED=0

for spec in "${MODELS[@]}"; do
    IFS=':' read -r VQVAE_NAME RUN_NAME N_TRAJ LOG_TOPK <<< "${spec}"

    if [ -n "${FILTER_MODEL}" ] && [[ "${RUN_NAME}" != *"${FILTER_MODEL}"* ]]; then
        continue
    fi

    CHECKPOINT_DIR="${AR_BASE}/${RUN_NAME}"
    VAL_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}-val.npz"
    TRAIN_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}.npz"   # position-mask source
    VQVAE_DIR="${VQVAE_BASE}/${VQVAE_NAME}"
    SWEEP_ROOT="${DIAG_BASE}/${GROUP_TAG}/${RUN_NAME}"
    LOG_DIR="${DIAG_BASE}/${GROUP_TAG}/logs"
    WANDB_GROUP="${RUN_NAME}-${GROUP_TAG}"

    if [ "${DRY_RUN}" = false ]; then
        if [ ! -f "${CHECKPOINT_DIR}/training_state.json" ]; then
            echo "[skip] ${RUN_NAME}: no NSP checkpoint at ${CHECKPOINT_DIR}"
            continue
        fi
        if [ ! -f "${VAL_TOKENS}" ]; then
            echo "[skip] ${RUN_NAME}: no val tokens at ${VAL_TOKENS}"
            continue
        fi
        if [ ! -f "${TRAIN_TOKENS}" ]; then
            echo "[skip] ${RUN_NAME}: no TRAIN tokens at ${TRAIN_TOKENS} (needed for position mask)"
            continue
        fi
        if [ ! -f "${VQVAE_DIR}/training_state.json" ]; then
            echo "[skip] ${RUN_NAME}: no VQ-VAE checkpoint at ${VQVAE_DIR}"
            continue
        fi
        mkdir -p "${SWEEP_ROOT}" "${LOG_DIR}" "${WANDB_BASE}"
    fi

    # ---- Stage 1: one rollout job per temperature ----
    ROLLOUT_JOBIDS=()
    for TEMP in "${TEMPERATURES[@]}"; do
        CFG="T${TEMP/./p}-pm"
        ROLLOUT_DIR="${SWEEP_ROOT}/${CFG}/rollout"

        if [ -f "${ROLLOUT_DIR}/rollout_logits.npz" ]; then
            echo "[skip] ${RUN_NAME}/${CFG}: rollout_logits.npz already exists"
            continue
        fi
        if [ "${DRY_RUN}" = false ]; then
            mkdir -p "${ROLLOUT_DIR}"
        fi

        TMPFILE="$(mktemp /tmp/diagroll_${RUN_NAME}_${CFG}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J dr-${RUN_NAME}-${CFG}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t 6:00:00
#SBATCH -o ${LOG_DIR}/${RUN_NAME}-${CFG}-%j.out
#SBATCH -e ${LOG_DIR}/${RUN_NAME}-${CFG}-%j.err

set -euo pipefail

cd "${REPODIR}"
source "${OCEAN}/.venvs/gust/bin/activate"
module load cuda/12.6.1

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:          \${SLURM_JOB_ID}"
echo "Node:         \$(hostname)"
echo "Started:      \$(date)"
echo "Run:          ${RUN_NAME} / ${CFG}"
echo "Rollout:      ${N_STEPS} steps × ${N_TRAJ} traj, T=${TEMP}, posmask ON, log_topk=${LOG_TOPK}"
echo "Output:       ${ROLLOUT_DIR}"
echo "=========================================="

python rollout_nsp.py \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --tokens_path "${VAL_TOKENS}" \\
    --train_tokens_path "${TRAIN_TOKENS}" \\
    --start_frame ${START_FRAME} \\
    --n_steps ${N_STEPS} \\
    --n_trajectories ${N_TRAJ} \\
    --seed ${SEED} \\
    --temperature ${TEMP} \\
    --log_topk ${LOG_TOPK} \\
    --output_dir "${ROLLOUT_DIR}"

echo "Finished:     \$(date)"
SBATCH_EOF

        if [ "${DRY_RUN}" = true ]; then
            echo "[dry-run] rollout ${RUN_NAME}/${CFG}  (T=${TEMP}, N=${N_TRAJ}, K=${LOG_TOPK})"
        else
            echo "Submitting rollout ${RUN_NAME}/${CFG}  (T=${TEMP})..."
            JOBID=$(sbatch --parsable "${TMPFILE}")
            echo "  -> ${JOBID}"
            ROLLOUT_JOBIDS+=("${JOBID}")
            N_SUBMITTED=$((N_SUBMITTED + 1))
        fi
        rm -f "${TMPFILE}"
    done

    # ---- Stage 2: diagnostics chain (depends on this model's rollouts) ----
    if [ -f "${SWEEP_ROOT}/position_ood/position_ood.npz" ]; then
        echo "[skip] ${RUN_NAME}: diagnostics already complete (position_ood.npz exists)"
        continue
    fi

    DEPENDENCY=""
    if [ ${#ROLLOUT_JOBIDS[@]} -gt 0 ]; then
        DEPENDENCY="--dependency=afterok:$(IFS=:; echo "${ROLLOUT_JOBIDS[*]}")"
    fi

    TMPFILE="$(mktemp /tmp/diaganalysis_${RUN_NAME}_XXXXXX.sbatch)"
    cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J dx-${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t 8:00:00
#SBATCH -o ${LOG_DIR}/${RUN_NAME}-diagnostics-%j.out
#SBATCH -e ${LOG_DIR}/${RUN_NAME}-diagnostics-%j.err

set -euo pipefail

cd "${REPODIR}"
source "${OCEAN}/.venvs/gust/bin/activate"
module load cuda/12.6.1

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:          \${SLURM_JOB_ID}"
echo "Node:         \$(hostname)"
echo "Started:      \$(date)"
echo "Run:          ${RUN_NAME} diagnostics chain"
echo "Sweep root:   ${SWEEP_ROOT}"
echo "Wandb:        ${WANDB_PROJECT} / group=${WANDB_GROUP}"
echo "=========================================="

# (a) survival — EMD-threshold explosion times (GPU: VQ-VAE decode)
echo "[stage a] multitraj_survival..."
python multitraj_survival.py \\
    --sweep_root "${SWEEP_ROOT}" \\
    --vqvae_dir "${VQVAE_DIR}" \\
    --data_path "${DATA_PATH}" \\
    --output_dir "${SWEEP_ROOT}/survival" \\
    --batch_size ${BATCH_SIZE} \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_group "${WANDB_GROUP}" \\
    --wandb_name "${WANDB_GROUP}-survival" \\
    --wandb_dir "${WANDB_BASE}"

# (b) per-cfg logit diagnostics (CPU)
for CFG_DIR in "${SWEEP_ROOT}"/T*; do
    CFG=\$(basename "\${CFG_DIR}")
    echo "[stage b] analyze_logits \${CFG}..."
    python analyze_logits.py \\
        --rollout_dir "\${CFG_DIR}/rollout" \\
        --output_dir "\${CFG_DIR}/logits" \\
        --cfg_name "\${CFG}" \\
        --survival_json "${SWEEP_ROOT}/survival/survival.json" \\
        --wandb_project ${WANDB_PROJECT} \\
        --wandb_group "${WANDB_GROUP}" \\
        --wandb_name "${WANDB_GROUP}-\${CFG}-logits" \\
        --wandb_dir "${WANDB_BASE}"
done

# (c) explosion-aligned logit traces (CPU)
echo "[stage c] analyze_logits_aligned..."
python analyze_logits_aligned.py \\
    --logits_root "${SWEEP_ROOT}" \\
    --survival_json "${SWEEP_ROOT}/survival/survival.json" \\
    --output_dir "${SWEEP_ROOT}/logits_aligned" \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_group "${WANDB_GROUP}" \\
    --wandb_name "${WANDB_GROUP}-logits-aligned" \\
    --wandb_dir "${WANDB_BASE}"

# (d) position-OOD (CPU; sanity with posmask ON: rate must be ≡ 0)
echo "[stage d] analyze_position_ood..."
python analyze_position_ood.py \\
    --train_tokens "${TRAIN_TOKENS}" \\
    --logits_root "${SWEEP_ROOT}" \\
    --survival_json "${SWEEP_ROOT}/survival/survival.json" \\
    --output_dir "${SWEEP_ROOT}/position_ood" \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_group "${WANDB_GROUP}" \\
    --wandb_name "${WANDB_GROUP}-position-ood" \\
    --wandb_dir "${WANDB_BASE}"

echo "=========================================="
echo "Finished:     \$(date)"
echo "=========================================="
SBATCH_EOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] diagnostics ${RUN_NAME}  (deps: ${#ROLLOUT_JOBIDS[@]} rollout jobs)"
    else
        echo "Submitting diagnostics ${RUN_NAME} ${DEPENDENCY:+(${DEPENDENCY})}..."
        JOBID=$(sbatch --parsable ${DEPENDENCY} "${TMPFILE}")
        echo "  -> ${JOBID}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
    rm -f "${TMPFILE}"
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
