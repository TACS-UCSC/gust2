#!/bin/bash
# Tokenize all VQ-VAE checkpoints.
#
# Usage:
#   ./scripts/tokenize_all.sh              Submit 9 tokenization jobs
#   ./scripts/tokenize_all.sh --dry-run    Show what would be submitted

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
VQVAE_BASE="${OCEAN}/experiments/vqvae"
TOKENS_BASE="${OCEAN}/experiments/tokens"
ACCOUNT="mth260004p"

# ---------- Configurations ----------
VQVAE_SIZES=(small medium large)
SCALE_CONFIGS=(
    "sc341:1,2,4,8,16"
    "sc917:1,2,4,8,16,24"
    "sc1941:1,2,4,8,16,24,32"
)

# ---------- Parse args ----------
DRY_RUN=false
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=true ;;
        --help|-h) echo "Usage: $0 [--dry-run]"; exit 0 ;;
    esac
done

mkdir -p "${TOKENS_BASE}" "${VQVAE_BASE}/logs"

echo "=========================================="
echo "Tokenize All VQ-VAE Checkpoints"
echo "  Data:   ${DATA_PATH}"
echo "  VQVAEs: ${VQVAE_BASE}"
echo "  Output: ${TOKENS_BASE}"
echo "  Dry run: ${DRY_RUN}"
echo "=========================================="
echo ""

N_SUBMITTED=0

for SIZE in "${VQVAE_SIZES[@]}"; do
    for cfg in "${SCALE_CONFIGS[@]}"; do
        IFS=: read -r SC SCALES <<< "${cfg}"
        RUN_NAME="${SIZE}-${SC}"
        CHECKPOINT_DIR="${VQVAE_BASE}/${RUN_NAME}"
        OUTPUT_PATH="${TOKENS_BASE}/${RUN_NAME}.npz"

        # Skip if checkpoint doesn't exist
        if [ ! -f "${CHECKPOINT_DIR}/training_state.json" ]; then
            echo "[skip] ${RUN_NAME}: no checkpoint found"
            continue
        fi

        # Skip if already tokenized
        if [ -f "${OUTPUT_PATH}" ]; then
            echo "[skip] ${RUN_NAME}: already tokenized"
            continue
        fi

        TMPFILE="$(mktemp /tmp/tokenize_${RUN_NAME}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J tok-${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 1:00:00
#SBATCH -o ${VQVAE_BASE}/logs/tok-${RUN_NAME}-%j.out
#SBATCH -e ${VQVAE_BASE}/logs/tok-${RUN_NAME}-%j.err

cd "${REPODIR}"
source /ocean/projects/mth260004p/sambamur/.venvs/gust/bin/activate
module load cuda/12.6.1
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\$LD_LIBRARY_PATH

echo "Tokenizing ${RUN_NAME}..."
python tokenizer.py save \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --data_path "${DATA_PATH}" \\
    --output "${OUTPUT_PATH}" \\
    --batch_size 128

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
