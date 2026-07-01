#!/bin/bash
# B3 — flat (single-scale) VQ tokenizer trio (BASELINES_SPEC.md §B3).
#
# Trains the three raster-bound tokenizers as degenerate MultiScaleVQ configs
# via the existing train.py: scales=(16,) / (24,) / (32,)  →  256 / 576 / 1024
# tokens per frame. Names follow the token-count convention:
#   small-flat-sc256, small-flat-sc576, small-flat-sc1024
# under experiments/vqvae/, next to the multi-scale family.
#
# THE GATE (run FIRST, before writing §3.2/§7): if flat-1024's recon floor is
# comparable to sc917's at similar budget, M6 claim-reduces to "token count
# wins" — see paper/OUTLINE.md R1.
#
# Hyperparams MATCH THE TRAINED small-sc* FAMILY (sweep_scales.sh): beta=0.1,
# ema_decay=0.85, batch=64, epochs=100, lr=1e-4, K=4096, cdim=512. This
# deviates from BASELINES_SPEC's "best codebook config (beta 0.25 / EMA 0.90)"
# wording DELIBERATELY (decision 2026-07-01): the family the Pareto compares
# against was actually trained at 0.1/0.85, so matching it leaves the scale
# structure as the only changed variable.
#
# Usage:
#   ./scripts/bridges/baselines/train_flat_vq.sh              # submit all 3
#   ./scripts/bridges/baselines/train_flat_vq.sh --only sc576 # one config
#   ./scripts/bridges/baselines/train_flat_vq.sh --dry-run

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
EXPERIMENT_BASE="${OCEAN}/experiments/vqvae"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Model config (Small arch, matches sweep_scales.sh) ----------
D_MODEL=512
N_HEADS=8
MLP_DIM=1024
ENCODER_DEPTH=5
DECODER_DEPTH=5
CODEBOOK_DIM=512
CODEBOOK_SIZE=4096
ROPE_THETA=32.0

# ---------- Training config (matches the trained small-sc* family) ----------
BATCH_SIZE=64
EPOCHS=100
LR=1e-4
BETA=0.1
EMA_DECAY=0.85
SAMPLE_STOP=20000
SEED=42
WANDB_PROJECT="gust2-baselines"
WANDB_GROUP="flat-vq"

# ---------- Flat scale configs ----------
# name:scales:tokens — single-scale quantization on the 32x32 latent grid.
# (16,) and (24,) quantize a downsampled residual and upsample back; (32,)
# is classic full-resolution VQ.
FLAT_CONFIGS=(
    "flat-sc256:16:256"
    "flat-sc576:24:576"
    "flat-sc1024:32:1024"
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

echo "=========================================="
echo "B3 flat-VQ tokenizer trio (Small arch, 1 GPU each)"
echo "  beta=${BETA} ema_decay=${EMA_DECAY} K=${CODEBOOK_SIZE} cdim=${CODEBOOK_DIM}"
echo "  (family-matched — see header)"
echo "  Wandb: ${WANDB_PROJECT} / group=${WANDB_GROUP}"
echo "  Dry run: ${DRY_RUN}"
echo "=========================================="

N_SUBMITTED=0

for cfg in "${FLAT_CONFIGS[@]}"; do
    IFS=: read -r FLAT_NAME SCALES TOKENS <<< "${cfg}"
    if [ -n "${ONLY}" ] && [[ "${FLAT_NAME}" != *"${ONLY}"* ]]; then continue; fi

    RUN_NAME="small-${FLAT_NAME}"
    CHECKPOINT_DIR="${EXPERIMENT_BASE}/${RUN_NAME}"

    if [ "${DRY_RUN}" = false ]; then
        mkdir -p "${CHECKPOINT_DIR}" "${EXPERIMENT_BASE}/logs" "${WANDB_BASE}"
    fi

    RESUME_FLAG=""
    if [ -f "${CHECKPOINT_DIR}/training_state.json" ]; then
        RESUME_FLAG="--resume"
    fi

    TMPFILE="$(mktemp /tmp/flatvq_${RUN_NAME}_XXXXXX.sbatch)"
    cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 2-00:00:00
#SBATCH -o ${EXPERIMENT_BASE}/logs/${RUN_NAME}-%j.out
#SBATCH -e ${EXPERIMENT_BASE}/logs/${RUN_NAME}-%j.err

set -euo pipefail

cd "${REPODIR}"
source "${OCEAN}/.venvs/gust/bin/activate"
module load cuda/12.6.1
export PYTHONUNBUFFERED=1

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:       \${SLURM_JOB_ID}"
echo "Node:      \$(hostname)"
echo "Started:   \$(date)"
echo "Run:       ${RUN_NAME}  (flat scales=(${SCALES},), ${TOKENS} tokens/frame)"
echo "Ckpt dir:  ${CHECKPOINT_DIR}"
echo "Resume:    ${RESUME_FLAG:-no}"
echo "=========================================="

python train.py \\
    --data_path "${DATA_PATH}" \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --d_model ${D_MODEL} \\
    --n_heads ${N_HEADS} \\
    --mlp_dim ${MLP_DIM} \\
    --encoder_depth ${ENCODER_DEPTH} \\
    --decoder_depth ${DECODER_DEPTH} \\
    --codebook_dim ${CODEBOOK_DIM} \\
    --codebook_size ${CODEBOOK_SIZE} \\
    --scales ${SCALES} \\
    --rope_theta ${ROPE_THETA} \\
    --batch_size ${BATCH_SIZE} \\
    --epochs ${EPOCHS} \\
    --lr ${LR} \\
    --beta ${BETA} \\
    --ema_decay ${EMA_DECAY} \\
    --sample_stop ${SAMPLE_STOP} \\
    --seed ${SEED} \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_name ${RUN_NAME} \\
    --wandb_group ${WANDB_GROUP} \\
    --wandb_dir "${WANDB_BASE}" \\
    ${RESUME_FLAG}

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
SBATCH_EOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] ${RUN_NAME}: scales=(${SCALES},), ${TOKENS} tokens, 1 GPU"
    else
        echo "Submitting ${RUN_NAME}: scales=(${SCALES},), ${TOKENS} tokens..."
        sbatch "${TMPFILE}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
    rm -f "${TMPFILE}"
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
