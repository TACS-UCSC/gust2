#!/bin/bash
# Single-step evaluation sweep for the robust-recipe NSP scaling runs.
# One Slurm job per network: eval_single_step.py over 2000 consecutive
# (t0, t1) val pairs at T=1.0, decoding predictions to pixels and
# producing the same TKE/enstrophy spectra + pixel histogram + EMD/RSE
# metrics that analyze_rollout produces, but on single-step (not AR)
# predictions.
#
# Walks the same 15-row (sc × NSP-arch) grid as sweep_robust_scaling.sh
# and reuses --size {small,medium,large} to pick the VQ tokenizer.
#
# Usage:
#   ./scripts/bridges/sweep_eval_single_step.sh                       # default size=small, all 15
#   ./scripts/bridges/sweep_eval_single_step.sh --size medium
#   ./scripts/bridges/sweep_eval_single_step.sh --size large --vqvae sc341
#   ./scripts/bridges/sweep_eval_single_step.sh --vqvae sc917 --label s34
#   ./scripts/bridges/sweep_eval_single_step.sh --dry-run
#   ./scripts/bridges/sweep_eval_single_step.sh --list

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
TOKENS_BASE="${OCEAN}/experiments/tokens"
VQVAE_BASE="${OCEAN}/experiments/vqvae"
AR_BASE="${OCEAN}/experiments/ar-robust-scaling"
EVAL_BASE="${OCEAN}/experiments/eval-scaling"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Eval config ----------
MAX_PAIRS=2000
TEMPERATURE=1.0                   # never greedy — see project memory
SEED=42
BATCH_SIZE=64
WANDB_PROJECT="gust2-analysis-bridges-scaling"

# ---------- Sweep grid (matches sweep_robust_scaling.sh) ----------
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
Usage: $0 [--size {small,medium,large}] [--vqvae <substr>] [--label <substr>] [--dry-run] [--list]
  --size <s>         VQ-VAE size: small (default), medium, or large.
  --vqvae <substr>   Filter by sc-config substring (e.g. sc341).
  --label <substr>   Filter by NSP arch label (e.g. s13).
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
    echo "Single-step eval sweep — 15 combos  (size=${SIZE}):"
    echo ""
    printf "  %-18s %-6s %3s %5s %4s\n" "VQ"  "label" "L" "d" "h"
    printf "  %-18s %-6s %3s %5s %4s\n" "--"  "-----" "-" "-" "-"
    for spec in "${TASKS[@]}"; do
        IFS=':' read -r sc l L d h <<< "${spec}"
        printf "  %-18s %-6s %3s %5s %4s\n" "${SIZE}-${sc}" "${l}" "${L}" "${d}" "${h}"
    done
    echo ""
    echo "Eval:         ${MAX_PAIRS} pairs @ T=${TEMPERATURE}"
    echo "Wandb:        ${WANDB_PROJECT}, group=<size>-<sc>-eval"
    echo "GPUs:         1× H100-80 per job, walltime 2h"
    exit 0
fi

# ---------- Filter task list ----------
SELECTED=()
for spec in "${TASKS[@]}"; do
    IFS=':' read -r sc l L d h <<< "${spec}"
    v="${SIZE}-${sc}"
    if [ -n "${FILTER_VQVAE}" ] && [[ "${v}" != *"${FILTER_VQVAE}"* ]]; then continue; fi
    if [ -n "${FILTER_LABEL}" ] && [[ "${l}" != *"${FILTER_LABEL}"* ]]; then continue; fi
    SELECTED+=("${v}:${l}:${L}:${d}:${h}")
done

if [ ${#SELECTED[@]} -eq 0 ]; then
    echo "No combos match the given filters."
    exit 1
fi

echo "=========================================="
echo "Single-step eval sweep (Bridges scaling)"
echo "  VQ size:          ${SIZE}"
echo "  Combos:           ${#SELECTED[@]}"
echo "  Eval base:        ${EVAL_BASE}"
echo "  Wandb project:    ${WANDB_PROJECT}"
echo "  Pairs × T:        ${MAX_PAIRS} × T=${TEMPERATURE}"
echo "  GPUs/job:         1× H100-80 (GPU-shared), walltime 2h"
echo "  Dry run:          ${DRY_RUN}"
echo "=========================================="

N_SUBMITTED=0

for spec in "${SELECTED[@]}"; do
    IFS=':' read -r VQVAE_NAME LABEL N_LAYER N_EMBD N_HEAD <<< "${spec}"

    SC_CFG="${VQVAE_NAME#*-}"
    RUN_NAME="${VQVAE_NAME}-nsp-${LABEL}"
    CHECKPOINT_DIR="${AR_BASE}/${RUN_NAME}"
    VAL_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}-val.npz"
    VQVAE_DIR="${VQVAE_BASE}/${VQVAE_NAME}"
    OUTPUT_DIR="${EVAL_BASE}/${RUN_NAME}"
    LOG_DIR="${EVAL_BASE}/logs"
    WANDB_NAME="${RUN_NAME}-eval"
    WANDB_GROUP="${SIZE}-${SC_CFG}-eval"

    if [ ! -f "${CHECKPOINT_DIR}/training_state.json" ] && [ "${DRY_RUN}" = false ]; then
        echo "[skip] ${RUN_NAME}: no NSP checkpoint at ${CHECKPOINT_DIR}"
        continue
    fi
    if [ ! -f "${VAL_TOKENS}" ] && [ "${DRY_RUN}" = false ]; then
        echo "[skip] ${RUN_NAME}: no val tokens at ${VAL_TOKENS}"
        continue
    fi
    if [ ! -f "${VQVAE_DIR}/training_state.json" ] && [ "${DRY_RUN}" = false ]; then
        echo "[skip] ${RUN_NAME}: no VQ-VAE checkpoint at ${VQVAE_DIR}"
        continue
    fi
    if [ -f "${OUTPUT_DIR}/eval_single_step.json" ]; then
        echo "[skip] ${RUN_NAME}: eval already complete (eval_single_step.json exists)"
        continue
    fi

    if [ "${DRY_RUN}" = false ]; then
        mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${WANDB_BASE}"
    fi

    TMPFILE="$(mktemp /tmp/evalscale_${RUN_NAME}_XXXXXX.sbatch)"
    cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J es-${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t 2:00:00
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
echo "Run:          ${RUN_NAME}"
echo "VQ-VAE:       ${VQVAE_DIR}"
echo "NSP ckpt:     ${CHECKPOINT_DIR}"
echo "Val tokens:   ${VAL_TOKENS}"
echo "Output:       ${OUTPUT_DIR}"
echo "Wandb:        ${WANDB_PROJECT} / ${WANDB_NAME}  (group=${WANDB_GROUP})"
echo "Eval cfg:     ${MAX_PAIRS} pairs, T=${TEMPERATURE}, seed=${SEED}"
echo "=========================================="

python eval_single_step.py \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --tokens_path "${VAL_TOKENS}" \\
    --vqvae_dir "${VQVAE_DIR}" \\
    --data_path "${DATA_PATH}" \\
    --output_dir "${OUTPUT_DIR}" \\
    --batch_size ${BATCH_SIZE} \\
    --max_pairs ${MAX_PAIRS} \\
    --seed ${SEED} \\
    --temperature ${TEMPERATURE} \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_name "${WANDB_NAME}" \\
    --wandb_group "${WANDB_GROUP}" \\
    --wandb_dir "${WANDB_BASE}"

echo "=========================================="
echo "Finished:     \$(date)"
echo "=========================================="
SBATCH_EOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] ${RUN_NAME}  (L=${N_LAYER}, d=${N_EMBD}, h=${N_HEAD})  -> ${OUTPUT_DIR}"
    else
        echo "Submitting ${RUN_NAME}  (L=${N_LAYER}, d=${N_EMBD}, h=${N_HEAD})..."
        NEW_JOBID=$(sbatch --parsable "${TMPFILE}")
        echo "  -> ${NEW_JOBID}"
    fi
    rm -f "${TMPFILE}"
    N_SUBMITTED=$((N_SUBMITTED + 1))
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
