#!/bin/bash
# CANONICAL long-term temperature-optimal scaling sweep at N=128 trajectories.
#
# This supersedes sweep_scaling_tempopt.sh (N=4) as the proper long-rollout
# temperature sweep + scaling-law data. N=4 has too much luck: the per-cell
# best-T and collapse_rate were chosen from only 4 sampling-noise trajectories
# at one IC. N=128 gives 32x the sampling ensemble -> a stable per-cell EMD,
# a best-T that isn't a coin flip, and collapse_rate resolved to 1/128.
#
# Per (cell, temp): rollout_nsp.py (N=128, posmask ON, --ic_chunk to bound GPU
# memory) -> analyze_rollout.py (spectra, pooled EMD, collapse_rate). Then per
# CELL (after all temps): multitraj_survival.py over the cell's T<temp>/rollout
# subdirs -> survival-vs-temperature curves + per-trajectory EMD-vs-time traces.
#
# Logits are NOT logged here (top-K logits at N=128 are ~6 TB across the grid).
# For logit traces use the dedicated diagnostics sweep (sweep_diagnostics_temp.sh)
# on the flagship models at N=12-25.
#
# Grid: 3 VQ sizes x 3 sc-configs x 5 NSP archs = 45 cells. One Slurm job per
# cell, looping its config's 5-temp bracket internally, then survival. Honest
# metrics land in wandb gust2-scaling-tempopt-n128-{small,medium,large},
# group=<sc>, run=<cell>-T<temp> (+ <cell>-survival).
#
# Usage:
#   ./scripts/bridges/sweep_scaling_tempopt_n128.sh                 # all 45 cells
#   ./scripts/bridges/sweep_scaling_tempopt_n128.sh --size small
#   ./scripts/bridges/sweep_scaling_tempopt_n128.sh --vqvae sc1941 --label s73
#   ./scripts/bridges/sweep_scaling_tempopt_n128.sh --dry-run
#   ./scripts/bridges/sweep_scaling_tempopt_n128.sh --list

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
TOKENS_BASE="${OCEAN}/experiments/tokens"
VQVAE_BASE="${OCEAN}/experiments/vqvae"
AR_BASE="${OCEAN}/experiments/ar-robust-scaling"
TEMPOPT_BASE="${OCEAN}/experiments/scaling-tempopt-n128"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"
# One wandb project per VQ-VAE tier: gust2-scaling-tempopt-n128-{size}.
# Within each, group = sc-config, run = <cell>-T<temp> (+ <cell>-survival).
WANDB_PROJECT_PREFIX="gust2-scaling-tempopt-n128"

# ---------- Rollout / analysis config ----------
N_STEPS=2000
START_FRAME=0
N_TRAJ=128                        # the whole point: 32x the N=4 ensemble
SEED=42
BATCH_SIZE=64
SURV_PROBE_STEP=50                # survival window cadence (N=128 -> keep it coarse)

# Per-config IC/trajectory chunk to bound peak GPU memory for the N=128 vmap on
# the 80GB H100. sc341 (341 tok) fits the full 128-wide batch; sc917/sc1941
# (917/1941 tok, up to 1024-wide) need chunking. generate_step_chunked reuses
# one jitted forward per chunk, so this trades a little parallelism for safety.
ic_chunk_for() {
    case "$1" in
        sc341)  echo "0"  ;;      # 0 = no chunk (full 128-wide batch fits)
        sc917)  echo "64" ;;
        sc1941) echo "32" ;;
        *)      echo "32" ;;
    esac
}

# Coarse temperature bracket per config, centered on the posmask-temp optimum
# (sc341 1.0, sc917 1.6, sc1941 1.8). Same brackets as the N=4 sweep so the
# N=128 scaling law is directly comparable.
temps_for() {
    case "$1" in
        sc341)  echo "0.8 0.9 1.0 1.1 1.2" ;;
        sc917)  echo "1.2 1.4 1.6 1.8 2.0" ;;
        sc1941) echo "1.4 1.6 1.8 2.0 2.2" ;;
        *) echo "" ;;
    esac
}
# N=128 is 32x the trajectory work of the N=4 sweep (which used 1-2h). The H100
# absorbs much of that in parallel, but 5 temps x 2000 steps + analyze + survival
# at N=128 needs real headroom. The per-temp metrics.json skip + survival_data
# skip make any timeout resumable, so a resubmit always makes progress.
walltime_for() {
    case "$1" in
        sc341)  echo "6:00:00"  ;;
        sc917)  echo "12:00:00" ;;
        sc1941) echo "20:00:00" ;;
        *)      echo "12:00:00" ;;
    esac
}

# ---------- Sweep grid (matches sweep_robust_scaling.sh) ----------
SIZES_ALL=(small medium large)
TASKS=(
    "sc341:s06:2:256:4"
    "sc341:s09:1:384:6"
    "sc341:s13:3:384:6"
    "sc341:s18:6:384:6"
    "sc341:s24:4:512:8"

    "sc917:s13:3:384:6"
    "sc917:s22:8:384:6"
    "sc917:s34:5:576:9"
    "sc917:s50:9:576:9"
    "sc917:s74:3:1024:16"

    "sc1941:s31:4:576:9"
    "sc1941:s48:6:640:10"
    "sc1941:s73:7:768:12"
    "sc1941:s113:6:1024:16"
    "sc1941:s139:8:1024:8"
)

# ---------- Parse args ----------
DRY_RUN=false
FILTER_SIZE=""
FILTER_VQVAE=""
FILTER_LABEL=""
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --size) FILTER_SIZE="$2"; shift 2 ;;
        --vqvae) FILTER_VQVAE="$2"; shift 2 ;;
        --label) FILTER_LABEL="$2"; shift 2 ;;
        --list) LIST_ONLY=true; shift ;;
        --help|-h)
            cat <<EOF
Usage: $0 [--size <s>] [--vqvae <substr>] [--label <substr>] [--dry-run] [--list]
  --size <s>         Only this VQ size (small|medium|large). Default: all three.
  --vqvae <substr>   Filter by sc-config substring (e.g. sc1941).
  --label <substr>   Filter by NSP arch label (e.g. s73).
  --dry-run          Print actions without submitting.
  --list             Print the grid + temperature brackets and exit.
EOF
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SIZES=("${SIZES_ALL[@]}")
if [ -n "${FILTER_SIZE}" ]; then SIZES=("${FILTER_SIZE}"); fi

if [ "${LIST_ONLY}" = true ]; then
    echo "Canonical long-term scaling sweep — N=${N_TRAJ}, posmask ON, + survival:"
    echo ""
    echo "  Temperature brackets:"
    for sc in sc341 sc917 sc1941; do
        printf "    %-7s temps=[%s]  ic_chunk=%s  walltime=%s\n" \
            "${sc}" "$(temps_for ${sc})" "$(ic_chunk_for ${sc})" "$(walltime_for ${sc})"
    done
    echo ""
    n_cells=0
    for SIZE in "${SIZES[@]}"; do
        for spec in "${TASKS[@]}"; do
            IFS=':' read -r sc l _ _ _ <<< "${spec}"
            n_cells=$((n_cells + 1))
        done
    done
    echo "  Cells (jobs): ${n_cells}  (sizes: ${SIZES[*]})"
    echo "  Rollouts:     ${n_cells} × 5 temps = $((n_cells * 5))  (+ ${n_cells} survival passes)"
    echo "  Wandb:        ${WANDB_PROJECT_PREFIX}-{small,medium,large}, group=<sc>, run=<cell>-T<temp>"
    echo "  Output base:  ${TEMPOPT_BASE}"
    exit 0
fi

echo "=========================================="
echo "Canonical long-term scaling sweep (N=${N_TRAJ})"
echo "  Sizes:         ${SIZES[*]}"
echo "  posmask:       ON   N_traj: ${N_TRAJ}   steps: ${N_STEPS}   + survival"
echo "  Output base:   ${TEMPOPT_BASE}"
echo "  Wandb:         ${WANDB_PROJECT_PREFIX}-{small,medium,large}"
echo "  Dry run:       ${DRY_RUN}"
echo "=========================================="

N_SUBMITTED=0

for SIZE in "${SIZES[@]}"; do
    case "${SIZE}" in
        small|medium|large) ;;
        *) echo "Invalid size '${SIZE}'" >&2; exit 1 ;;
    esac

    for spec in "${TASKS[@]}"; do
        IFS=':' read -r SC LABEL N_LAYER N_EMBD N_HEAD <<< "${spec}"

        if [ -n "${FILTER_VQVAE}" ] && [[ "${SC}" != *"${FILTER_VQVAE}"* ]]; then continue; fi
        if [ -n "${FILTER_LABEL}" ] && [[ "${LABEL}" != *"${FILTER_LABEL}"* ]]; then continue; fi

        VQVAE_NAME="${SIZE}-${SC}"
        RUN_NAME="${VQVAE_NAME}-nsp-${LABEL}"
        CHECKPOINT_DIR="${AR_BASE}/${RUN_NAME}"
        VAL_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}-val.npz"
        TRAIN_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}.npz"
        VQVAE_DIR="${VQVAE_BASE}/${VQVAE_NAME}"
        CELL_DIR="${TEMPOPT_BASE}/${RUN_NAME}"
        LOG_DIR="${TEMPOPT_BASE}/logs"
        WANDB_PROJECT="${WANDB_PROJECT_PREFIX}-${SIZE}"
        WANDB_GROUP="${SC}"
        TEMPS="$(temps_for ${SC})"
        WALLTIME="$(walltime_for ${SC})"
        IC_CHUNK="$(ic_chunk_for ${SC})"

        if [ "${DRY_RUN}" = false ]; then
            if [ ! -f "${CHECKPOINT_DIR}/training_state.json" ]; then
                echo "[skip] ${RUN_NAME}: no NSP checkpoint"
                continue
            fi
            if [ ! -f "${VAL_TOKENS}" ] || [ ! -f "${TRAIN_TOKENS}" ]; then
                echo "[skip] ${RUN_NAME}: missing val or train tokens"
                continue
            fi
            if [ ! -f "${VQVAE_DIR}/training_state.json" ]; then
                echo "[skip] ${RUN_NAME}: no VQ-VAE checkpoint"
                continue
            fi
            mkdir -p "${CELL_DIR}" "${LOG_DIR}" "${WANDB_BASE}"
        fi

        TMPFILE="$(mktemp /tmp/n128_${RUN_NAME}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J n128-${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t ${WALLTIME}
#SBATCH -o ${LOG_DIR}/${RUN_NAME}-%j.out
#SBATCH -e ${LOG_DIR}/${RUN_NAME}-%j.err

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
echo "Cell:         ${RUN_NAME}  (N=${N_TRAJ}, ic_chunk=${IC_CHUNK}, posmask ON)"
echo "Temps:        ${TEMPS}"
echo "Wandb:        ${WANDB_PROJECT} / group=${WANDB_GROUP}"
echo "=========================================="

for TEMP in ${TEMPS}; do
    TP=\${TEMP/./p}
    ROUT="${CELL_DIR}/T\${TP}/rollout"
    AOUT="${CELL_DIR}/T\${TP}/analysis"
    if [ -f "\${AOUT}/metrics.json" ]; then
        echo "[skip] T=\${TEMP}: analysis already complete"
        continue
    fi
    echo "---- T=\${TEMP} ----"
    python rollout_nsp.py \\
        --checkpoint_dir "${CHECKPOINT_DIR}" \\
        --tokens_path "${VAL_TOKENS}" \\
        --train_tokens_path "${TRAIN_TOKENS}" \\
        --start_frame ${START_FRAME} \\
        --n_steps ${N_STEPS} \\
        --n_trajectories ${N_TRAJ} \\
        --ic_chunk ${IC_CHUNK} \\
        --seed ${SEED} \\
        --temperature \${TEMP} \\
        --output_dir "\${ROUT}"

    python analyze_rollout.py \\
        --rollout_dir "\${ROUT}" \\
        --vqvae_dir "${VQVAE_DIR}" \\
        --data_path "${DATA_PATH}" \\
        --output_dir "\${AOUT}" \\
        --batch_size ${BATCH_SIZE} \\
        --seed ${SEED} \\
        --wandb_project ${WANDB_PROJECT} \\
        --wandb_name "${RUN_NAME}-T\${TP}" \\
        --wandb_group "${WANDB_GROUP}" \\
        --wandb_dir "${WANDB_BASE}"
done

# ---- Survival curves across this cell's temperatures (once all temps done) ----
SURV_OUT="${CELL_DIR}/survival"
if [ -f "\${SURV_OUT}/survival_data.npz" ]; then
    echo "[skip] survival: already complete"
else
    echo "---- survival (all temps) ----"
    python multitraj_survival.py \\
        --sweep_root "${CELL_DIR}" \\
        --vqvae_dir "${VQVAE_DIR}" \\
        --data_path "${DATA_PATH}" \\
        --output_dir "\${SURV_OUT}" \\
        --probe_step ${SURV_PROBE_STEP} \\
        --batch_size ${BATCH_SIZE} \\
        --seed ${SEED} \\
        --wandb_project ${WANDB_PROJECT} \\
        --wandb_name "${RUN_NAME}-survival" \\
        --wandb_group "${WANDB_GROUP}" \\
        --wandb_dir "${WANDB_BASE}"
fi

echo "=========================================="
echo "Finished:     \$(date)"
echo "=========================================="
SBATCH_EOF

        if [ "${DRY_RUN}" = true ]; then
            echo "[dry-run] ${RUN_NAME}  temps=[${TEMPS}]  N=${N_TRAJ} ic_chunk=${IC_CHUNK} walltime=${WALLTIME}"
        else
            echo "Submitting ${RUN_NAME} (N=${N_TRAJ}, temps: ${TEMPS})..."
            JOBID=$(sbatch --parsable "${TMPFILE}")
            echo "  -> ${JOBID}"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        fi
        rm -f "${TMPFILE}"
    done
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
