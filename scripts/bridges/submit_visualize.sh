#!/bin/bash
# Batch submitter for visualize_diagnostics.py — one 1-GPU job per model,
# figures land in <sweep_root>/visual/ AND in wandb
# (gust2-diagnostics-bridges, group <run>-posmask-temp, job_type=visual).
#
# Companion to sweep_diagnostics_temp.sh: run it after that sweep's
# diagnostics jobs finish (needs <sweep_root>/survival/survival_data.npz
# and per-cfg logits/diagnostics.npz for the full figure set; missing
# pieces are skipped gracefully by the script itself).
#
# Usage:
#   ./scripts/bridges/submit_visualize.sh              # all 3 models
#   ./scripts/bridges/submit_visualize.sh --model sc1941
#   ./scripts/bridges/submit_visualize.sh --dry-run

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
VQVAE_BASE="${OCEAN}/experiments/vqvae"
DIAG_BASE="${OCEAN}/experiments/diagnostics"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Config (matches sweep_diagnostics_temp.sh) ----------
GROUP_TAG="posmask-temp"
WANDB_PROJECT="gust2-diagnostics-bridges"
BATCH_SIZE=64
SNAP_TIMES="0,250,500,1000,1500,2000"
SPECTRA_STRIDE=50
N_TRAJ_SPECTRA=4

# "<vqvae_name>:<run_name>"
MODELS=(
    "small-sc341:small-sc341-nsp-s18"
    "small-sc917:small-sc917-nsp-s34"
    "small-sc1941:small-sc1941-nsp-s73"
)

# ---------- Parse args ----------
DRY_RUN=false
FILTER_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --model) FILTER_MODEL="$2"; shift 2 ;;
        --help|-h)
            cat <<EOF
Usage: $0 [--model <substr>] [--dry-run]
  --model <substr>   Filter models by substring (e.g. sc1941).
  --dry-run          Print actions without submitting.
EOF
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

N_SUBMITTED=0

for spec in "${MODELS[@]}"; do
    IFS=':' read -r VQVAE_NAME RUN_NAME <<< "${spec}"

    if [ -n "${FILTER_MODEL}" ] && [[ "${RUN_NAME}" != *"${FILTER_MODEL}"* ]]; then
        continue
    fi

    SWEEP_ROOT="${DIAG_BASE}/${GROUP_TAG}/${RUN_NAME}"
    VQVAE_DIR="${VQVAE_BASE}/${VQVAE_NAME}"
    LOG_DIR="${DIAG_BASE}/${GROUP_TAG}/logs"
    WANDB_GROUP="${RUN_NAME}-${GROUP_TAG}"

    if [ "${DRY_RUN}" = false ]; then
        if [ ! -d "${SWEEP_ROOT}" ]; then
            echo "[skip] ${RUN_NAME}: no sweep root at ${SWEEP_ROOT}"
            continue
        fi
        if [ ! -f "${SWEEP_ROOT}/survival/survival_data.npz" ]; then
            echo "[warn] ${RUN_NAME}: no survival_data.npz yet — EMD/snapshot figures will be skipped"
        fi
        mkdir -p "${LOG_DIR}"
    fi

    TMPFILE="$(mktemp /tmp/diagviz_${RUN_NAME}_XXXXXX.sbatch)"
    cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J viz-${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t 2:00:00
#SBATCH -o ${LOG_DIR}/${RUN_NAME}-visual-%j.out
#SBATCH -e ${LOG_DIR}/${RUN_NAME}-visual-%j.err

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
echo "Run:          ${RUN_NAME} visualization"
echo "Sweep root:   ${SWEEP_ROOT}"
echo "Wandb:        ${WANDB_PROJECT} / group=${WANDB_GROUP} / ${WANDB_GROUP}-visual"
echo "=========================================="

python visualize_diagnostics.py \\
    --sweep_root "${SWEEP_ROOT}" \\
    --vqvae_dir "${VQVAE_DIR}" \\
    --data_path "${DATA_PATH}" \\
    --batch_size ${BATCH_SIZE} \\
    --snap_times "${SNAP_TIMES}" \\
    --spectra_stride ${SPECTRA_STRIDE} \\
    --n_traj_spectra ${N_TRAJ_SPECTRA} \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_group "${WANDB_GROUP}" \\
    --wandb_name "${WANDB_GROUP}-visual" \\
    --wandb_dir "${WANDB_BASE}"

echo "Finished:     \$(date)"
SBATCH_EOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] visualize ${RUN_NAME}  -> ${SWEEP_ROOT}/visual"
    else
        echo "Submitting visualize ${RUN_NAME}..."
        JOBID=$(sbatch --parsable "${TMPFILE}")
        echo "  -> ${JOBID}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
    rm -f "${TMPFILE}"
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
