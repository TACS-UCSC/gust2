#!/bin/bash
# E2 — clean 3-way support-mask ablation on ONE sc917 cell (paper F5.1, M3).
#
# Cell: small-sc917-nsp-s34 (the sc917 D/P anchor arch; decision 2026-07-01).
# Arms (mask granularity), all trained WITH substitution noise 0.1 so the
# ONLY variable is the mask:
#   nomask    — --loss_mask none       + --emission_mask none      (fresh training)
#   perscale  — --loss_mask per_scale  + --emission_mask per_scale (fresh training)
#   pertoken  — the existing ar-robust-scaling checkpoint (per-token mask +
#               noise; no training needed)
# Sampling arms per model: cold T=0.7 and warm T=1.6 (sc917 swept optimum).
# The cold/per-token arm folds in old E6: mask alone under cold still
# diffusive-collapses -> mask ⊥ temperature.
#
# Training recipe mirrors sweep_robust_scaling.sh verbatim for sc917:
# batch 128 (32/GPU on 4× H100), lr 2e-4, 400 epochs, refine 2, seed 42.
#
# Fan-out: 2 training jobs; 6 rollout+analysis jobs (3 arms × 2 temps),
# fresh-arm rollouts Slurm-depend (afterok) on their training job, per-token
# rollouts submit immediately. All steps skip-resume (training_state.json /
# rollout_tokens.npz / metrics.json), so resubmitting is always safe.
#
# OOD-rate traces for the F5.1 overlay are computed OFFLINE from the saved
# rollout_tokens.npz + train tokens (both token-space; no extra GPU work).
#
# Usage:
#   ./scripts/bridges/sweep_mask_ablation.sh                # everything
#   ./scripts/bridges/sweep_mask_ablation.sh --arms nomask,perscale
#   ./scripts/bridges/sweep_mask_ablation.sh --rollout-only # skip training jobs
#   ./scripts/bridges/sweep_mask_ablation.sh --temps "0.7 1.0 1.6"
#   ./scripts/bridges/sweep_mask_ablation.sh --dry-run

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
VENV="${OCEAN}/.venvs/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
TOKENS_BASE="${OCEAN}/experiments/tokens"
VQVAE_BASE="${OCEAN}/experiments/vqvae"
ROBUST_BASE="${OCEAN}/experiments/ar-robust-scaling"
ABLATION_BASE="${OCEAN}/experiments/mask-ablation"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"
WANDB_PROJECT="gust2-mask-ablation"

# ---------- The one cell ----------
VQVAE_NAME="small-sc917"
LABEL="s34"
N_LAYER=5
N_EMBD=576
N_HEAD=9
CELL="${VQVAE_NAME}-nsp-${LABEL}"

TRAIN_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}.npz"
VAL_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}-val.npz"
VQVAE_DIR="${VQVAE_BASE}/${VQVAE_NAME}"

# ---------- Training recipe (== sweep_robust_scaling.sh, sc917) ----------
N_REFINE_LAYERS=2
EPOCHS=400
BATCH_SIZE=128
LR=2e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
SAVE_EVERY=5
SEED=42
SUBSTITUTION_RATE=0.1
TRAIN_WALLTIME="12:00:00"

# ---------- Rollout / analysis config (protocol == N=128 sweep, smaller N) ----------
N_STEPS=2000
N_TRAJ=16
START_FRAME=0
ROLLOUT_SEED=42
ANALYSIS_BATCH=64
TEMPS="0.7 1.6"
ROLLOUT_WALLTIME="1:30:00"

# ---------- Parse args ----------
DRY_RUN=false
ARMS="nomask,perscale,pertoken"
ROLLOUT_ONLY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --arms) ARMS="$2"; shift 2 ;;
        --temps) TEMPS="$2"; shift 2 ;;
        --rollout-only) ROLLOUT_ONLY=true; shift ;;
        --help|-h)
            cat <<EOF
Usage: $0 [--arms nomask,perscale,pertoken] [--temps "<t1> <t2> ..."] \\
          [--rollout-only] [--dry-run]
  --arms <csv>       Subset of arms to submit (default: all three).
  --temps "<list>"   Sampling temperatures per arm (default: "${TEMPS}").
  --rollout-only     Skip training jobs (rollouts submit without afterok;
                     use once the fresh arms have finished training).
  --dry-run          Print actions without submitting.
EOF
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "=========================================="
echo "E2 3-way mask ablation — ${CELL}"
echo "  Arms:      ${ARMS}"
echo "  Temps:     ${TEMPS}   N_traj=${N_TRAJ}   steps=${N_STEPS}"
echo "  Output:    ${ABLATION_BASE}"
echo "  Wandb:     ${WANDB_PROJECT}"
echo "  Dry run:   ${DRY_RUN}"
echo "=========================================="

if [ "${DRY_RUN}" = false ]; then
    for f in "${TRAIN_TOKENS}" "${VAL_TOKENS}" "${VQVAE_DIR}/training_state.json"; do
        if [ ! -e "${f}" ]; then
            echo "FATAL: missing prerequisite ${f}" >&2
            exit 1
        fi
    done
    mkdir -p "${ABLATION_BASE}/logs" "${WANDB_BASE}"
fi

# ---------- Wandb id helper (persistent resume, as in robust sweep) ----------
get_or_create_wandb_id() {
    local ckpt_dir=$1
    local id_file="${ckpt_dir}/wandb_id.txt"
    if [ -f "${id_file}" ]; then
        cat "${id_file}"
    else
        local id
        id=$(head /dev/urandom | tr -dc 'a-z0-9' | head -c 8)
        if [ "${DRY_RUN}" = false ]; then
            mkdir -p "${ckpt_dir}"
            echo "${id}" > "${id_file}"
        fi
        echo "${id}"
    fi
}

# Maps arm -> (loss_mask, emission_mask, checkpoint_dir, needs_training)
arm_loss_mask()   { case "$1" in nomask) echo "none";; perscale) echo "per_scale";; pertoken) echo "per_token";; esac; }
arm_ckpt_dir()    {
    case "$1" in
        pertoken) echo "${ROBUST_BASE}/${CELL}" ;;
        *)        echo "${ABLATION_BASE}/${CELL}-$1" ;;
    esac
}
arm_needs_train() { case "$1" in pertoken) echo false;; *) echo true;; esac; }

N_TRAIN_SUBMITTED=0
N_ROLLOUT_SUBMITTED=0

IFS=',' read -ra ARM_LIST <<< "${ARMS}"
for ARM in "${ARM_LIST[@]}"; do
    LOSS_MASK="$(arm_loss_mask "${ARM}")"
    if [ -z "${LOSS_MASK}" ]; then
        echo "Unknown arm '${ARM}' (use nomask|perscale|pertoken)" >&2
        exit 1
    fi
    CKPT_DIR="$(arm_ckpt_dir "${ARM}")"
    NEEDS_TRAIN="$(arm_needs_train "${ARM}")"
    TRAIN_JOBID=""

    # ---- Training job (fresh arms only) ----
    if [ "${NEEDS_TRAIN}" = true ] && [ "${ROLLOUT_ONLY}" = false ]; then
        RUN_NAME="${CELL}-${ARM}"
        RESUME_FLAG=""
        if [ -f "${CKPT_DIR}/training_state.json" ]; then
            RESUME_FLAG="--resume"
        fi
        WANDB_ID=$(get_or_create_wandb_id "${CKPT_DIR}")

        TMPFILE="$(mktemp /tmp/e2_${RUN_NAME}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J e2-${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:4
#SBATCH -t ${TRAIN_WALLTIME}
#SBATCH -o ${ABLATION_BASE}/logs/${RUN_NAME}-train-%j.out
#SBATCH -e ${ABLATION_BASE}/logs/${RUN_NAME}-train-%j.err

set -euo pipefail

cd "${REPODIR}"
source "${VENV}/bin/activate"
module load cuda/12.6.1
export PYTHONUNBUFFERED=1

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:        \${SLURM_JOB_ID}   (\$(hostname), \$(date))"
echo "E2 arm:     ${ARM}  (loss_mask=${LOSS_MASK}, substitution=${SUBSTITUTION_RATE})"
echo "Arch:       ${N_LAYER}L d=${N_EMBD} h=${N_HEAD} refine=${N_REFINE_LAYERS}"
echo "Ckpt:       ${CKPT_DIR}   resume=${RESUME_FLAG:-no}"
echo "=========================================="

python train_nsp.py \\
    --tokens_path "${TRAIN_TOKENS}" \\
    --train_tokens_path "${TRAIN_TOKENS}" \\
    --substitution_rate ${SUBSTITUTION_RATE} \\
    --loss_mask ${LOSS_MASK} \\
    --n_layer ${N_LAYER} \\
    --n_head ${N_HEAD} \\
    --n_embd ${N_EMBD} \\
    --n_refine_layers ${N_REFINE_LAYERS} \\
    --batch_size ${BATCH_SIZE} \\
    --epochs ${EPOCHS} \\
    --lr ${LR} \\
    --weight_decay ${WEIGHT_DECAY} \\
    --grad_clip ${GRAD_CLIP} \\
    --save_every ${SAVE_EVERY} \\
    --seed ${SEED} \\
    --checkpoint_dir "${CKPT_DIR}" \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_name ${RUN_NAME}-train \\
    --wandb_group training \\
    --wandb_dir "${WANDB_BASE}" \\
    --wandb_id ${WANDB_ID} \\
    ${RESUME_FLAG}

echo "Finished: \$(date)"
SBATCH_EOF

        if [ "${DRY_RUN}" = true ]; then
            echo "[dry-run] TRAIN ${RUN_NAME}  loss_mask=${LOSS_MASK}  resume=${RESUME_FLAG:-no}"
            TRAIN_JOBID="<jobid-${RUN_NAME}>"
        else
            echo "Submitting TRAIN ${RUN_NAME} (loss_mask=${LOSS_MASK})..."
            TRAIN_JOBID=$(sbatch --parsable "${TMPFILE}")
            echo "  -> ${TRAIN_JOBID}"
            N_TRAIN_SUBMITTED=$((N_TRAIN_SUBMITTED + 1))
        fi
        rm -f "${TMPFILE}"
    fi

    if [ "${ARM}" = "pertoken" ] && [ "${DRY_RUN}" = false ] \
        && [ ! -f "${CKPT_DIR}/training_state.json" ]; then
        echo "[skip] ${ARM}: no robust-scaling checkpoint at ${CKPT_DIR}"
        continue
    fi

    # ---- Rollout + analysis jobs: one per temperature ----
    # Emission mask matches the arm's training mask; per-token additionally
    # needs the train tokens to rebuild the position mask at inference.
    case "${ARM}" in
        nomask)   EMIT_ARGS="--emission_mask none" ;;
        perscale) EMIT_ARGS="--emission_mask per_scale" ;;
        pertoken) EMIT_ARGS="--emission_mask per_token --train_tokens_path ${TRAIN_TOKENS}" ;;
    esac

    for TEMP in ${TEMPS}; do
        TP="${TEMP/./p}"
        ROUT="${ABLATION_BASE}/rollouts/${ARM}/T${TP}/rollout"
        AOUT="${ABLATION_BASE}/rollouts/${ARM}/T${TP}/analysis"

        DEP_FLAG=""
        if [ -n "${TRAIN_JOBID}" ] && [ "${DRY_RUN}" = false ]; then
            DEP_FLAG="--dependency=afterok:${TRAIN_JOBID}"
        fi

        TMPFILE="$(mktemp /tmp/e2_${ARM}_T${TP}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J e2-${ARM}-T${TP}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t ${ROLLOUT_WALLTIME}
#SBATCH -o ${ABLATION_BASE}/logs/${ARM}-T${TP}-%j.out
#SBATCH -e ${ABLATION_BASE}/logs/${ARM}-T${TP}-%j.err

set -euo pipefail

cd "${REPODIR}"
source "${VENV}/bin/activate"
module load cuda/12.6.1
export PYTHONUNBUFFERED=1

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:        \${SLURM_JOB_ID}   (\$(hostname), \$(date))"
echo "E2 rollout: arm=${ARM}  T=${TEMP}  N=${N_TRAJ}  steps=${N_STEPS}"
echo "Ckpt:       ${CKPT_DIR}"
echo "=========================================="

if [ -f "${AOUT}/metrics.json" ]; then
    echo "[skip] ${ARM} T=${TEMP}: analysis already complete"
else
    if [ -f "${ROUT}/rollout_tokens.npz" ]; then
        echo "[reuse] ${ARM} T=${TEMP}: rollout exists, running analysis only"
    else
        python rollout_nsp.py \\
            --checkpoint_dir "${CKPT_DIR}" \\
            --tokens_path "${VAL_TOKENS}" \\
            ${EMIT_ARGS} \\
            --start_frame ${START_FRAME} \\
            --n_steps ${N_STEPS} \\
            --n_trajectories ${N_TRAJ} \\
            --seed ${ROLLOUT_SEED} \\
            --temperature ${TEMP} \\
            --output_dir "${ROUT}"
    fi

    python analyze_rollout.py \\
        --rollout_dir "${ROUT}" \\
        --vqvae_dir "${VQVAE_DIR}" \\
        --data_path "${DATA_PATH}" \\
        --output_dir "${AOUT}" \\
        --batch_size ${ANALYSIS_BATCH} \\
        --seed ${ROLLOUT_SEED} \\
        --wandb_project ${WANDB_PROJECT} \\
        --wandb_name "${ARM}-T${TP}" \\
        --wandb_group "${ARM}" \\
        --wandb_dir "${WANDB_BASE}"
fi

echo "Finished: \$(date)"
SBATCH_EOF

        if [ "${DRY_RUN}" = true ]; then
            echo "[dry-run] ROLLOUT ${ARM} T=${TEMP}  ${EMIT_ARGS}  dep=${TRAIN_JOBID:-none}"
        else
            echo "Submitting ROLLOUT ${ARM} T=${TEMP}..."
            JOBID=$(sbatch --parsable ${DEP_FLAG} "${TMPFILE}")
            echo "  -> ${JOBID}"
            N_ROLLOUT_SUBMITTED=$((N_ROLLOUT_SUBMITTED + 1))
        fi
        rm -f "${TMPFILE}"
    done
done

echo ""
echo "Done. ${N_TRAIN_SUBMITTED} training + ${N_ROLLOUT_SUBMITTED} rollout job(s) submitted."
