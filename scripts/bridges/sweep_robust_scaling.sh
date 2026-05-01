#!/bin/bash
# 15-job robust-recipe NSP scaling sweep on Bridges (PSC).
#
# Five parameter counts per VQ config, geometrically spaced around the
# domain-specific Chinchilla optimum (D_uniq / P ≈ 0.54 anchored at
# small-sc341 × micro). All jobs use the robust training recipe:
#   * per-position loss mask (--train_tokens_path)
#   * same-position substitution noise (--substitution_rate 0.1)
#
# Submission pattern mirrors bridges/sweep_nsp.sh: one Slurm job per
# combo, 12 h walltime, auto-resume via training_state.json + persistent
# wandb id. Resubmit the script (or a single combo) to continue past
# wall clock. 4× H100-80 per job, global batch 64 (16/GPU) — matches
# the robust recipe validated on Derecho's 4× A100-40.
#
# Usage:
#   ./scripts/bridges/sweep_robust_scaling.sh                   # all 15 (small)
#   ./scripts/bridges/sweep_robust_scaling.sh --size medium     # 15 jobs against medium-* tokens
#   ./scripts/bridges/sweep_robust_scaling.sh --vqvae sc341     # 5 jobs
#   ./scripts/bridges/sweep_robust_scaling.sh --label s13       # all VQs at s13
#   ./scripts/bridges/sweep_robust_scaling.sh --size large --vqvae sc917 --label s34
#   ./scripts/bridges/sweep_robust_scaling.sh --dry-run
#   ./scripts/bridges/sweep_robust_scaling.sh --list

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"           # folder is still 'gust' but contains gust2 code
VENV="${OCEAN}/.venvs/gust"
TOKENS_BASE="${OCEAN}/experiments/tokens"
AR_BASE="${OCEAN}/experiments/ar-robust-scaling"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Shared training config ----------
N_REFINE_LAYERS=2
BATCH_SIZE=128                    # 32/GPU × 4 H100-80 — fills the headroom
EPOCHS=400
LR=2e-4                           # linear-scaled with 2× batch vs validated 1e-4 @ 64
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
SAVE_EVERY=5
SEED=42
SUBSTITUTION_RATE=0.1
WANDB_PROJECT="gust2-nsp-robust-scaling-bridges"

# ---------- Sweep grid ----------
# Five NSP sizes per VQ, bracketing the projected D/P ≈ 0.54 optimum:
#   sc341  optimum ~12.65 M  (anchor = s13)
#   sc917  optimum ~34.08 M  (anchor = s34)
#   sc1941 optimum ~72.18 M  (anchor = s73)
# Format per row: sc_cfg:label:L:d:h  (the VQ size prefix — small/medium/large
# — is prepended at submission time from --size, defaulting to small).
TASKS=(
    # sc341 — anchor s13 (existing micro arch)
    "sc341:s06:2:256:4"
    "sc341:s09:1:384:6"
    "sc341:s13:3:384:6"
    "sc341:s18:6:384:6"
    "sc341:s24:4:512:8"

    # sc917 — anchor s34
    "sc917:s13:3:384:6"
    "sc917:s22:8:384:6"
    "sc917:s34:5:576:9"
    "sc917:s50:9:576:9"
    "sc917:s74:3:1024:16"

    # sc1941 — anchor s73
    "sc1941:s31:4:576:9"
    "sc1941:s48:6:640:10"
    "sc1941:s73:7:768:12"
    "sc1941:s113:6:1024:16"
    "sc1941:s139:8:1024:8"
)

# ---------- Wandb id helper ----------
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

# ---------- Parse args ----------
DRY_RUN=false
FILTER_VQVAE=""
FILTER_LABEL=""
SIZE="small"
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --vqvae) FILTER_VQVAE="$2"; shift 2 ;;
        --label) FILTER_LABEL="$2"; shift 2 ;;
        --size) SIZE="$2"; shift 2 ;;
        --list) LIST_ONLY=true; shift ;;
        --help|-h)
            cat <<EOF
Usage: $0 [--size {small,medium,large}] [--vqvae <substr>] [--label <label>] [--dry-run] [--list]
  --size <s>         VQ-VAE size: small (default), medium, or large.
  --vqvae <substr>   Filter by sc-config substring (e.g. sc341).
  --label <label>    Filter by NSP arch label (e.g. s13).
  --dry-run          Print actions without submitting.
  --list             Print the 15-combo grid (for the chosen --size) and exit.
EOF
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${SIZE}" in
    small|medium|large) ;;
    *) echo "Invalid --size '${SIZE}' (must be small, medium, or large)" >&2; exit 1 ;;
esac

if [ "${LIST_ONLY}" = true ]; then
    echo "Robust-recipe scaling sweep — 15 (VQ × NSP) combos  (size=${SIZE}):"
    echo ""
    printf "  %-18s %-6s %3s %5s %4s\n" "VQ"  "label" "L" "d" "h"
    printf "  %-18s %-6s %3s %5s %4s\n" "--"  "-----" "-" "-" "-"
    for spec in "${TASKS[@]}"; do
        IFS=':' read -r sc l L d h <<< "${spec}"
        printf "  %-18s %-6s %3s %5s %4s\n" "${SIZE}-${sc}" "${l}" "${L}" "${d}" "${h}"
    done
    echo ""
    echo "Robust knobs: substitution_rate=${SUBSTITUTION_RATE}, n_refine=${N_REFINE_LAYERS}"
    echo "GPUs:         4× H100-80 per job, global batch=${BATCH_SIZE} (32/GPU), LR=${LR}"
    echo "Walltime:     12h/job — resubmit to resume."
    exit 0
fi

# ---------- Filter task list ----------
# Each row in TASKS encodes the sc-config (e.g. sc341); prefix the chosen
# VQ size to form the full token-set name (e.g. medium-sc341).
SELECTED=()
for spec in "${TASKS[@]}"; do
    IFS=':' read -r sc l L d h <<< "${spec}"
    v="${SIZE}-${sc}"
    if [ -n "${FILTER_VQVAE}" ] && [[ "${v}" != *"${FILTER_VQVAE}"* ]]; then continue; fi
    if [ -n "${FILTER_LABEL}" ] && [ "${l}" != "${FILTER_LABEL}" ]; then continue; fi
    SELECTED+=("${v}:${l}:${L}:${d}:${h}")
done

if [ ${#SELECTED[@]} -eq 0 ]; then
    echo "No combos match the given filters."
    exit 1
fi

echo "=========================================="
echo "Robust-recipe NSP Scaling Sweep (Bridges)"
echo "  VQ size:          ${SIZE}"
echo "  Combos:           ${#SELECTED[@]}"
echo "  Output base:      ${AR_BASE}"
echo "  Wandb project:    ${WANDB_PROJECT}"
echo "  Substitution rate:${SUBSTITUTION_RATE}"
echo "  GPUs/job:         4× H100-80 (GPU-shared)"
echo "  Global batch:     ${BATCH_SIZE} (32/GPU)  LR=${LR}"
echo "  Walltime:         12h/job (resubmit to resume)"
echo "  Dry run:          ${DRY_RUN}"
echo "=========================================="

N_SUBMITTED=0

for spec in "${SELECTED[@]}"; do
    IFS=':' read -r VQVAE_NAME LABEL N_LAYER N_EMBD N_HEAD <<< "${spec}"

    TOKENS_PATH="${TOKENS_BASE}/${VQVAE_NAME}.npz"
    TRAIN_TOKENS="${TOKENS_PATH}"
    if [ ! -f "${TOKENS_PATH}" ] && [ "${DRY_RUN}" = false ]; then
        echo "[skip] ${VQVAE_NAME}: tokens not found at ${TOKENS_PATH}"
        continue
    fi

    RUN_NAME="${VQVAE_NAME}-nsp-${LABEL}"
    CHECKPOINT_DIR="${AR_BASE}/${RUN_NAME}"
    LOG_DIR="${AR_BASE}/logs"
    if [ "${DRY_RUN}" = false ]; then
        mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${WANDB_BASE}"
    fi

    SC_CFG="${VQVAE_NAME#*-}"
    WANDB_GROUP="${SC_CFG}-scaling"

    RESUME_FLAG=""
    if [ -f "${CHECKPOINT_DIR}/training_state.json" ]; then
        RESUME_FLAG="--resume"
    fi
    WANDB_ID=$(get_or_create_wandb_id "${CHECKPOINT_DIR}")

    TMPFILE="$(mktemp /tmp/nspscaling_${RUN_NAME}_XXXXXX.sbatch)"
    cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:4
#SBATCH -t 12:00:00
#SBATCH -o ${LOG_DIR}/${RUN_NAME}-%j.out
#SBATCH -e ${LOG_DIR}/${RUN_NAME}-%j.err

set -euo pipefail

cd "${REPODIR}"
source "${VENV}/bin/activate"
module load cuda/12.6.1

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:           \${SLURM_JOB_ID}"
echo "Node:          \$(hostname)"
echo "Started:       \$(date)"
echo "GPUs:          4× H100-80 (GPU-shared)"
echo "Run:           ${RUN_NAME}"
echo "Arch:          ${N_LAYER}L, n_embd=${N_EMBD}, n_head=${N_HEAD}, refine=${N_REFINE_LAYERS}"
echo "Tokens:        ${TOKENS_PATH}"
echo "Train tokens:  ${TRAIN_TOKENS}  (per-position mask + substitution pool)"
echo "Substitution:  ${SUBSTITUTION_RATE}"
echo "Ckpt dir:      ${CHECKPOINT_DIR}"
echo "Batch:         ${BATCH_SIZE}  (32/GPU)  LR=${LR}"
echo "Wandb:         ${WANDB_PROJECT} / ${RUN_NAME}  (id=${WANDB_ID}, group=${WANDB_GROUP})"
echo "Resume:        ${RESUME_FLAG:-no}"
echo "=========================================="

python train_nsp.py \\
    --tokens_path "${TOKENS_PATH}" \\
    --train_tokens_path "${TRAIN_TOKENS}" \\
    --substitution_rate ${SUBSTITUTION_RATE} \\
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
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_name ${RUN_NAME} \\
    --wandb_group ${WANDB_GROUP} \\
    --wandb_dir "${WANDB_BASE}" \\
    --wandb_id ${WANDB_ID} \\
    ${RESUME_FLAG}

echo "=========================================="
echo "Finished:      \$(date)"
echo "=========================================="
SBATCH_EOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] ${RUN_NAME}  (L=${N_LAYER}, d=${N_EMBD}, h=${N_HEAD})  wandb_id=${WANDB_ID}  resume=${RESUME_FLAG:-no}"
    else
        echo "Submitting ${RUN_NAME}  (L=${N_LAYER}, d=${N_EMBD}, h=${N_HEAD})  wandb_id=${WANDB_ID}  resume=${RESUME_FLAG:-no}..."
        NEW_JOBID=$(sbatch --parsable "${TMPFILE}")
        echo "  -> ${NEW_JOBID}"
    fi
    rm -f "${TMPFILE}"
    N_SUBMITTED=$((N_SUBMITTED + 1))
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
