#!/bin/bash
# 15-job robust-recipe NSP scaling sweep.
#
# Five parameter counts per VQ config, geometrically spaced around the
# domain-specific Chinchilla optimum (D_uniq / P ≈ 0.54 anchored at
# small-sc341 × micro). All jobs use the robust training recipe:
#   * per-position loss mask (--train_tokens_path)
#   * same-position substitution noise (--substitution_rate 0.1)
#
# Submission pattern mirrors sweep_nsp.sh: one PBS job per combo, 12h
# walltime, auto-resume via training_state.json + persistent wandb id.
# Resubmit the script (or a single combo) to continue past wall clock.
#
# Usage:
#   ./scripts/derecho/sweep_robust_scaling.sh                   # all 15
#   ./scripts/derecho/sweep_robust_scaling.sh --vqvae sc341     # 5 jobs
#   ./scripts/derecho/sweep_robust_scaling.sh --label s13       # all VQs at s13
#   ./scripts/derecho/sweep_robust_scaling.sh --vqvae sc917 --label s34
#   ./scripts/derecho/sweep_robust_scaling.sh --dry-run
#   ./scripts/derecho/sweep_robust_scaling.sh --list

set -euo pipefail

# ---------- Paths ----------
REPODIR="${HOME}/gust2"
VENV="${SCRATCH}/.venvs/gust2"
TOKENS_BASE="${SCRATCH}/experiments/tokens"
AR_BASE="${SCRATCH}/experiments/ar-robust-scaling"
WANDB_BASE="${SCRATCH}/wandb"
ACCOUNT="UCSC0009"

# ---------- Shared training config ----------
N_REFINE_LAYERS=2
BATCH_SIZE=64
EPOCHS=400
LR=1e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
SAVE_EVERY=5
SEED=42
SUBSTITUTION_RATE=0.1
WANDB_PROJECT="gust2-nsp-robust-scaling-derecho"

# ---------- Sweep grid ----------
# Five sizes per VQ, bracketing the projected D/P ≈ 0.54 optimum:
#   sc341  optimum ~12.65 M  (anchor = s13)
#   sc917  optimum ~34.08 M  (anchor = s34)
#   sc1941 optimum ~72.18 M  (anchor = s73)
# Format per row: vq:label:L:d:h
TASKS=(
    # sc341 — anchor s13 (existing micro arch)
    "small-sc341:s06:2:256:4"
    "small-sc341:s09:1:384:6"
    "small-sc341:s13:3:384:6"
    "small-sc341:s18:6:384:6"
    "small-sc341:s24:4:512:8"

    # sc917 — anchor s34
    "small-sc917:s13:3:384:6"
    "small-sc917:s22:8:384:6"
    "small-sc917:s34:5:576:9"
    "small-sc917:s50:9:576:9"
    "small-sc917:s74:3:1024:16"

    # sc1941 — anchor s73
    "small-sc1941:s31:4:576:9"
    "small-sc1941:s48:6:640:10"
    "small-sc1941:s73:7:768:12"
    "small-sc1941:s113:6:1024:16"
    "small-sc1941:s139:8:1024:8"
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --vqvae) FILTER_VQVAE="$2"; shift 2 ;;
        --label) FILTER_LABEL="$2"; shift 2 ;;
        --list)
            echo "Robust-recipe scaling sweep — 15 (VQ × NSP) combos:"
            echo ""
            printf "  %-14s %-6s %3s %5s %4s\n" "VQ"  "label" "L" "d" "h"
            printf "  %-14s %-6s %3s %5s %4s\n" "--"  "-----" "-" "-" "-"
            for spec in "${TASKS[@]}"; do
                IFS=':' read -r v l L d h <<< "${spec}"
                printf "  %-14s %-6s %3s %5s %4s\n" "${v}" "${l}" "${L}" "${d}" "${h}"
            done
            echo ""
            echo "Robust knobs: substitution_rate=${SUBSTITUTION_RATE}, n_refine=${N_REFINE_LAYERS}"
            echo "Walltime: 12h/job — resubmit to resume."
            exit 0
            ;;
        --help|-h)
            cat <<EOF
Usage: $0 [--vqvae <substr>] [--label <label>] [--dry-run] [--list]
  --vqvae <substr>   Filter by VQ-VAE token name (e.g. sc341).
  --label <label>    Filter by NSP arch label (e.g. s13).
  --dry-run          Print actions without submitting.
  --list             Print the 15-combo grid and exit.
EOF
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------- Filter task list ----------
SELECTED=()
for spec in "${TASKS[@]}"; do
    IFS=':' read -r v l L d h <<< "${spec}"
    if [ -n "${FILTER_VQVAE}" ] && [[ "${v}" != *"${FILTER_VQVAE}"* ]]; then continue; fi
    if [ -n "${FILTER_LABEL}" ] && [ "${l}" != "${FILTER_LABEL}" ]; then continue; fi
    SELECTED+=("${spec}")
done

if [ ${#SELECTED[@]} -eq 0 ]; then
    echo "No combos match the given filters."
    exit 1
fi

echo "=========================================="
echo "Robust-recipe NSP Scaling Sweep (Derecho)"
echo "  Combos:           ${#SELECTED[@]}"
echo "  Output base:      ${AR_BASE}"
echo "  Wandb project:    ${WANDB_PROJECT}"
echo "  Substitution rate:${SUBSTITUTION_RATE}"
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

    VQVAE_SIZE="${VQVAE_NAME%%-*}"
    SC_CFG="${VQVAE_NAME#*-}"
    WANDB_GROUP="${SC_CFG}-scaling"

    RESUME_FLAG=""
    if [ -f "${CHECKPOINT_DIR}/training_state.json" ]; then
        RESUME_FLAG="--resume"
    fi
    WANDB_ID=$(get_or_create_wandb_id "${CHECKPOINT_DIR}")

    TMPFILE="$(mktemp /tmp/nspscaling_${RUN_NAME}_XXXXXX.pbs)"
    cat > "${TMPFILE}" << PBS_EOF
#!/bin/bash
#PBS -N ${RUN_NAME}
#PBS -A ${ACCOUNT}
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=64:ngpus=4:mpiprocs=1:mem=480GB
#PBS -j oe
#PBS -o ${LOG_DIR}/${RUN_NAME}.log

set -euo pipefail

cd "${REPODIR}"

export TMPDIR="\${SCRATCH}/\${USER}/tmpdir"
mkdir -p "\${TMPDIR}"

module purge
module load ncarenv

source "${VENV}/bin/activate"

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:           \${PBS_JOBID}"
echo "Node:          \$(hostname)"
echo "Started:       \$(date)"
echo "GPUs:          4x A100-40 (node-exclusive)"
echo "Run:           ${RUN_NAME}"
echo "Arch:          ${N_LAYER}L, n_embd=${N_EMBD}, n_head=${N_HEAD}, refine=${N_REFINE_LAYERS}"
echo "Tokens:        ${TOKENS_PATH}"
echo "Train tokens:  ${TRAIN_TOKENS}  (per-position mask + substitution pool)"
echo "Substitution:  ${SUBSTITUTION_RATE}"
echo "Ckpt dir:      ${CHECKPOINT_DIR}"
echo "Batch:         ${BATCH_SIZE}  (16/GPU)"
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
PBS_EOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] ${RUN_NAME}  (L=${N_LAYER}, d=${N_EMBD}, h=${N_HEAD})  wandb_id=${WANDB_ID}  resume=${RESUME_FLAG:-no}"
    else
        echo "Submitting ${RUN_NAME}  (L=${N_LAYER}, d=${N_EMBD}, h=${N_HEAD})  wandb_id=${WANDB_ID}  resume=${RESUME_FLAG:-no}..."
        NEW_JOBID=$(qsub "${TMPFILE}")
        echo "  -> ${NEW_JOBID}"
    fi
    rm -f "${TMPFILE}"
    N_SUBMITTED=$((N_SUBMITTED + 1))
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
