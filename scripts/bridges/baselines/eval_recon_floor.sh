#!/bin/bash
# B3 gate, test-time version — val-window quantized-recon floors.
#
# Runs eval_recon_floor.py over the flat trio AND the multi-scale small
# family in one job, so the F3.3 Pareto (EMD floor + spectra vs token count
# at fixed model size) is computed by a single code path. Train recon MSE
# already favors flat-1024 over sc917 (2.43 vs 2.75) — but flat recons look
# low-pass filtered (275/4096 codes), so the gate verdict rides on EMD +
# spectrum tail, which this job measures.
#
# One GPU-shared job, ~minutes per config; per-config skip via metrics.json.
#
# Usage:
#   ./scripts/bridges/baselines/eval_recon_floor.sh            # all 6
#   ./scripts/bridges/baselines/eval_recon_floor.sh --only flat
#   ./scripts/bridges/baselines/eval_recon_floor.sh --dry-run

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
VQ_BASE="${OCEAN}/experiments/vqvae"
OUT_BASE="${OCEAN}/experiments/analysis/recon-floor"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

VAL_START=20000
VAL_STOP=22000

# Tokenizers (all under experiments/vqvae/)
CONFIGS=(
    small-flat-sc256
    small-flat-sc576
    small-flat-sc1024
    small-sc341
    small-sc917
    small-sc1941
)

# ---------- Parse args ----------
DRY_RUN=false
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--only <substr>] [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

RUN_LIST=()
for cfg in "${CONFIGS[@]}"; do
    if [ -n "${ONLY}" ] && [[ "${cfg}" != *"${ONLY}"* ]]; then continue; fi
    if [ -f "${OUT_BASE}/${cfg}/metrics.json" ]; then
        echo "Skip ${cfg} (metrics.json exists)"
        continue
    fi
    RUN_LIST+=("${cfg}")
done

if [ ${#RUN_LIST[@]} -eq 0 ]; then
    echo "Nothing to do."
    exit 0
fi

echo "=========================================="
echo "Recon-floor eval: ${RUN_LIST[*]}"
echo "  val window: [${VAL_START},${VAL_STOP})"
echo "  Dry run: ${DRY_RUN}"
echo "=========================================="

LOG_DIR="${OUT_BASE}/logs"
[ "${DRY_RUN}" = false ] && mkdir -p "${LOG_DIR}" "${WANDB_BASE}"

TMPFILE="$(mktemp /tmp/recon_floor_XXXXXX.sbatch)"
cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J recon-floor
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 2:00:00
#SBATCH -o ${LOG_DIR}/recon-floor-%j.out
#SBATCH -e ${LOG_DIR}/recon-floor-%j.err

set -euo pipefail

cd "${REPODIR}"
source "${OCEAN}/.venvs/gust/bin/activate"
module load cuda/12.6.1
export PYTHONUNBUFFERED=1

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:      \${SLURM_JOB_ID} (\$(hostname), \$(date))"
echo "Configs:  ${RUN_LIST[*]}"
echo "=========================================="

for CFG in ${RUN_LIST[*]}; do
    if [ -f "${OUT_BASE}/\${CFG}/metrics.json" ]; then
        echo "--- \${CFG}: metrics.json exists, skipping"
        continue
    fi
    echo "--- \${CFG}"
    python eval_recon_floor.py \\
        --vqvae_dir "${VQ_BASE}/\${CFG}" \\
        --data_path "${DATA_PATH}" \\
        --sample_start ${VAL_START} \\
        --sample_stop ${VAL_STOP} \\
        --output_dir "${OUT_BASE}/\${CFG}" \\
        --wandb_name "recon-floor-\${CFG}" \\
        --wandb_dir "${WANDB_BASE}"
done

echo "Finished: \$(date)"
SBATCH_EOF

if [ "${DRY_RUN}" = true ]; then
    echo "[dry-run] 1 job, configs: ${RUN_LIST[*]}"
else
    JOB_ID=$(sbatch --parsable "${TMPFILE}")
    echo "Submitted job ${JOB_ID}"
fi
rm -f "${TMPFILE}"
