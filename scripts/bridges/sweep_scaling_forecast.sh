#!/bin/bash
# Short-horizon FORECAST-SKILL sweep for the FULL scaling grid, the
# short-term companion to sweep_scaling_tempopt.sh.
#
# sweep_scaling_tempopt.sh measures the LONG-RUN climate: a 2000-step free
# rollout from one IC, time-averaged TKE/EMD. This sweep measures SHORT-TERM
# forecast skill: from N_ICS distinct ground-truth frames spread across the
# val set, free-run a short horizon (n_steps=10) and compute pixel-EMD +
# TKE/enstrophy RSE at lead times k in {1,2,5,10} (analyze_forecast.py).
#
# Two questions vs the long-term figures:
#   1. Does the scaling law change shape at short horizons?
#   2. Does the short-term optimal temperature diverge (colder) from the
#      long-term one? The temperature brackets below EXTEND COLDER than the
#      long-term sweep (cold floor 0.7) while overlapping it, so the local
#      plotter can pick each (cell, horizon)'s own best T by pixel-EMD.
#
# Grid: 3 VQ sizes × 3 sc-configs × 5 NSP archs = 45 cells. One Slurm job per
# cell, looping its config's temp bracket internally (forecast rollout +
# analyze_forecast per temp). Per-horizon metrics land in wandb
# gust2-scaling-forecast-{size}, group=<sc>, one run per (cell, temp) named
# <run>-T<temp>.
#
# Forecast rollouts are far cheaper than the long ones (n_steps 10 vs 2000),
# so walltimes are short even with the wider temp bracket and N_ICS=128.
#
# Usage:
#   ./scripts/bridges/sweep_scaling_forecast.sh                 # all 45 cells
#   ./scripts/bridges/sweep_scaling_forecast.sh --size small
#   ./scripts/bridges/sweep_scaling_forecast.sh --vqvae sc1941 --label s73
#   ./scripts/bridges/sweep_scaling_forecast.sh --dry-run
#   ./scripts/bridges/sweep_scaling_forecast.sh --list

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
TOKENS_BASE="${OCEAN}/experiments/tokens"
VQVAE_BASE="${OCEAN}/experiments/vqvae"
AR_BASE="${OCEAN}/experiments/ar-robust-scaling"
FORECAST_BASE="${OCEAN}/experiments/scaling-forecast"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"
# One wandb project per VQ-VAE tier (browsable): gust2-scaling-forecast-{size}.
# Within each, group = sc-config, run = <cell>-T<temp>.
WANDB_PROJECT_PREFIX="gust2-scaling-forecast"

# ---------- Rollout / analysis config ----------
N_STEPS=10                        # short horizon (covers leads 1..10)
N_ICS=128                         # distinct ICs spread across the val set
IC_CHUNK=32                       # IC sub-batch (bounds memory on sc1941)
HORIZONS="1,2,5,10"               # lead times to evaluate
SEED=42
BATCH_SIZE=64

# Temperature brackets — EXTENDED COLDER than the long-term sweep (cold floor
# 0.7; this project treats greedy/sub-0.7 sampling as historically broken),
# overlapping the long-term grid so the short-vs-long optimal-T comparison is
# honest. Long-term was sc341 0.8-1.2, sc917 1.2-2.0, sc1941 1.4-2.2.
temps_for() {
    case "$1" in
        sc341)  echo "0.7 0.8 0.9 1.0 1.1 1.2" ;;
        sc917)  echo "0.8 1.0 1.2 1.4 1.6 1.8 2.0" ;;
        sc1941) echo "1.0 1.2 1.4 1.6 1.8 2.0 2.2" ;;
        *) echo "" ;;
    esac
}
# Forecast is ~200x fewer AR steps than the long rollout; even the wider temp
# bracket fits comfortably. sc1941 gets more headroom (5.7x tokens/frame +
# N_ICS=128 decode). Timeouts are harmless — the per-temp metrics.json skip
# lets a resubmit resume.
walltime_for() {
    case "$1" in
        sc341|sc917) echo "0:30:00" ;;
        sc1941)      echo "1:00:00" ;;
        *)           echo "0:30:00" ;;
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
    echo "Short-horizon forecast scaling sweep — posmask ON, N_ICS=${N_ICS}, horizons=${HORIZONS}:"
    echo ""
    echo "  Temperature brackets (extended colder than long-term):"
    for sc in sc341 sc917 sc1941; do
        printf "    %-7s %s\n" "${sc}" "$(temps_for ${sc})"
    done
    echo ""
    n_cells=0
    n_rollouts=0
    for SIZE in "${SIZES[@]}"; do
        for spec in "${TASKS[@]}"; do
            IFS=':' read -r sc l _ _ _ <<< "${spec}"
            n_cells=$((n_cells + 1))
            n_rollouts=$((n_rollouts + $(temps_for ${sc} | wc -w)))
        done
    done
    echo "  Cells (jobs): ${n_cells}  (sizes: ${SIZES[*]})"
    echo "  Rollouts:     ${n_rollouts} (cells × per-config temps)"
    echo "  Wandb:        ${WANDB_PROJECT_PREFIX}-{small,medium,large}, group=<sc>, run=<cell>-T<temp>"
    exit 0
fi

echo "=========================================="
echo "Short-horizon forecast scaling sweep"
echo "  Sizes:         ${SIZES[*]}"
echo "  posmask:       ON   N_ICS: ${N_ICS}   horizon steps: ${N_STEPS}"
echo "  Horizons:      ${HORIZONS}"
echo "  Output base:   ${FORECAST_BASE}"
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
        CELL_DIR="${FORECAST_BASE}/${RUN_NAME}"
        LOG_DIR="${FORECAST_BASE}/logs"
        WANDB_PROJECT="${WANDB_PROJECT_PREFIX}-${SIZE}"
        WANDB_GROUP="${SC}"
        TEMPS="$(temps_for ${SC})"
        WALLTIME="$(walltime_for ${SC})"

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

        TMPFILE="$(mktemp /tmp/forecast_${RUN_NAME}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J fc-${RUN_NAME}
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
echo "Cell:         ${RUN_NAME}  (forecast, posmask ON, N_ICS=${N_ICS})"
echo "Horizons:     ${HORIZONS}"
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
        --n_ics ${N_ICS} \\
        --ic_chunk ${IC_CHUNK} \\
        --n_steps ${N_STEPS} \\
        --seed ${SEED} \\
        --temperature \${TEMP} \\
        --output_dir "\${ROUT}"

    python analyze_forecast.py \\
        --rollout_dir "\${ROUT}" \\
        --vqvae_dir "${VQVAE_DIR}" \\
        --data_path "${DATA_PATH}" \\
        --horizons "${HORIZONS}" \\
        --output_dir "\${AOUT}" \\
        --batch_size ${BATCH_SIZE} \\
        --seed ${SEED} \\
        --wandb_project ${WANDB_PROJECT} \\
        --wandb_name "${RUN_NAME}-T\${TP}" \\
        --wandb_group "${WANDB_GROUP}" \\
        --wandb_dir "${WANDB_BASE}"
done

echo "=========================================="
echo "Finished:     \$(date)"
echo "=========================================="
SBATCH_EOF

        if [ "${DRY_RUN}" = true ]; then
            echo "[dry-run] ${RUN_NAME}  temps=[${TEMPS}]  walltime=${WALLTIME}"
        else
            echo "Submitting ${RUN_NAME} (temps: ${TEMPS})..."
            JOBID=$(sbatch --parsable "${TMPFILE}")
            echo "  -> ${JOBID}"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        fi
        rm -f "${TMPFILE}"
    done
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
