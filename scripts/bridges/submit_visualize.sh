#!/bin/bash
# Batch submitter for visualize_diagnostics.py — one job per model,
# figures land in <sweep_root>/visual/ AND in wandb
# (gust2-diagnostics-bridges, group <run>-posmask-temp, job_type=visual).
#
# Companion to sweep_diagnostics_temp.sh: run it after that sweep's
# diagnostics jobs finish (needs <sweep_root>/survival/survival_data.npz
# and per-cfg logits/diagnostics.npz for the full figure set; missing
# pieces are skipped gracefully by the script itself).
#
# --cpu runs on RM-shared (no GPU) with JAX_PLATFORMS=cpu and a lighter
# decode budget (stride 100, 2 traj). The decode is slower on CPU but the
# whole figure set still renders; the temperature-selection primary metric
# (pixel-EMD) and figures 1-3 need no decode at all.
#
# Usage:
#   ./scripts/bridges/submit_visualize.sh              # all 3 models, GPU
#   ./scripts/bridges/submit_visualize.sh --cpu        # all 3 models, CPU node
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
CPU_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --cpu) CPU_MODE=true; shift ;;
        --model) FILTER_MODEL="$2"; shift 2 ;;
        --help|-h)
            cat <<EOF
Usage: $0 [--model <substr>] [--cpu] [--dry-run]
  --model <substr>   Filter models by substring (e.g. sc1941).
  --cpu              Submit to RM-shared (CPU only) instead of GPU-shared.
  --dry-run          Print actions without submitting.
EOF
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------- Partition / resource selection ----------
if [ "${CPU_MODE}" = true ]; then
    PARTITION="RM-shared"
    JOB_PREFIX="vizc"
    WALLTIME="6:00:00"
    RES_LINES="#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G"
    # JAX_PLATFORMS=cpu forces CPU and skips CUDA init on a node with no GPU.
    ENV_SETUP="export JAX_PLATFORMS=cpu"
    VIZ_STRIDE=100
    VIZ_NTRAJ=2
else
    PARTITION="GPU-shared"
    JOB_PREFIX="viz"
    WALLTIME="2:00:00"
    RES_LINES="#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009"
    ENV_SETUP='module load cuda/12.6.1
NVIDIA_LIBS=$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=$(find $NVIDIA_LIBS -name "lib" -type d | tr "\n" ":"):${LD_LIBRARY_PATH:-}'
    VIZ_STRIDE=${SPECTRA_STRIDE}
    VIZ_NTRAJ=${N_TRAJ_SPECTRA}
fi

echo "Mode: ${PARTITION}  (stride=${VIZ_STRIDE}, n_traj=${VIZ_NTRAJ})"

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
#SBATCH -J ${JOB_PREFIX}-${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH -N 1
${RES_LINES}
#SBATCH -t ${WALLTIME}
#SBATCH -o ${LOG_DIR}/${RUN_NAME}-visual-%j.out
#SBATCH -e ${LOG_DIR}/${RUN_NAME}-visual-%j.err

set -euo pipefail

cd "${REPODIR}"
source "${OCEAN}/.venvs/gust/bin/activate"
${ENV_SETUP}

echo "=========================================="
echo "Job:          \${SLURM_JOB_ID}"
echo "Node:         \$(hostname)"
echo "Started:      \$(date)"
echo "Run:          ${RUN_NAME} visualization (${PARTITION})"
echo "Sweep root:   ${SWEEP_ROOT}"
echo "Wandb:        ${WANDB_PROJECT} / group=${WANDB_GROUP} / ${WANDB_GROUP}-visual"
echo "=========================================="

python visualize_diagnostics.py \\
    --sweep_root "${SWEEP_ROOT}" \\
    --vqvae_dir "${VQVAE_DIR}" \\
    --data_path "${DATA_PATH}" \\
    --batch_size ${BATCH_SIZE} \\
    --snap_times "${SNAP_TIMES}" \\
    --spectra_stride ${VIZ_STRIDE} \\
    --n_traj_spectra ${VIZ_NTRAJ} \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_group "${WANDB_GROUP}" \\
    --wandb_name "${WANDB_GROUP}-visual" \\
    --wandb_dir "${WANDB_BASE}"

echo "Finished:     \$(date)"
SBATCH_EOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] visualize ${RUN_NAME} (${PARTITION})  -> ${SWEEP_ROOT}/visual"
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
