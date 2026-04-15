#!/bin/bash
# Tokenize validation data (frames 20000-22000) for all VQ-VAE checkpoints.
#
# Usage:
#   ./scripts/tokenize_val.sh              Submit 9 jobs
#   ./scripts/tokenize_val.sh --dry-run
#   ./scripts/tokenize_val.sh --force      Overwrite existing val tokens

set -euo pipefail

OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
VQVAE_BASE="${OCEAN}/experiments/vqvae"
TOKENS_BASE="${OCEAN}/experiments/tokens"
ACCOUNT="mth260004p"

SAMPLE_START=20000
SAMPLE_STOP=22000

VQVAE_SIZES=(small medium large)
SCALE_CONFIGS=(sc341 sc917 sc1941)

DRY_RUN=false
FORCE=false
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=true ;;
        --force) FORCE=true ;;
        --help|-h) echo "Usage: $0 [--dry-run] [--force]"; exit 0 ;;
    esac
done

mkdir -p "${TOKENS_BASE}" "${VQVAE_BASE}/logs"

echo "=========================================="
echo "Tokenize Validation Data (frames ${SAMPLE_START}-${SAMPLE_STOP})"
echo "=========================================="
echo ""

N_SUBMITTED=0

for SIZE in "${VQVAE_SIZES[@]}"; do
    for SC in "${SCALE_CONFIGS[@]}"; do
        RUN_NAME="${SIZE}-${SC}"
        CHECKPOINT_DIR="${VQVAE_BASE}/${RUN_NAME}"
        OUTPUT_PATH="${TOKENS_BASE}/${RUN_NAME}-val.npz"

        if [ ! -f "${CHECKPOINT_DIR}/training_state.json" ]; then
            echo "[skip] ${RUN_NAME}: no checkpoint"
            continue
        fi

        TRAIN_TOKENS="${TOKENS_BASE}/${RUN_NAME}.npz"
        if [ ! -f "${TRAIN_TOKENS}" ] && [ "${DRY_RUN}" = false ]; then
            echo "[skip] ${RUN_NAME}: training tokens not found"
            continue
        fi

        if [ -f "${OUTPUT_PATH}" ] && [ "${FORCE}" = false ]; then
            echo "[skip] ${RUN_NAME}: already tokenized (use --force to overwrite)"
            continue
        fi

        TMPFILE="$(mktemp /tmp/tokval_${RUN_NAME}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J tokval-${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:v100-32:1
#SBATCH --exclude=w009
#SBATCH -t 1:00:00
#SBATCH -o ${VQVAE_BASE}/logs/tokval-${RUN_NAME}-%j.out
#SBATCH -e ${VQVAE_BASE}/logs/tokval-${RUN_NAME}-%j.err

cd "${REPODIR}"
source /ocean/projects/mth260004p/sambamur/.venvs/gust/bin/activate
module load cuda/12.6.1
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\$LD_LIBRARY_PATH

echo "Tokenizing validation: ${RUN_NAME} (frames ${SAMPLE_START}-${SAMPLE_STOP})..."
python tokenizer.py save \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --data_path "${DATA_PATH}" \\
    --output "${OUTPUT_PATH}" \\
    --sample_start ${SAMPLE_START} \\
    --sample_stop ${SAMPLE_STOP} \\
    --batch_size 128 \\
    --fit_from "${TRAIN_TOKENS}"

echo "Done: ${OUTPUT_PATH}"
SBATCH_EOF

        if [ "${DRY_RUN}" = true ]; then
            echo "[dry-run] ${RUN_NAME} -> ${OUTPUT_PATH}"
        else
            echo "Submitting ${RUN_NAME}..."
            sbatch "${TMPFILE}"
        fi
        rm -f "${TMPFILE}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    done
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
