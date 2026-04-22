#!/bin/bash
# Single-step NSP evaluation sweep on Derecho.
#
# Derecho `main` is node-exclusive (1 job = full 4-GPU A100-40 node). Each
# eval_single_step.py task is single-GPU, so we bundle 9 tasks (3 VQ-VAE
# sizes x 3 NSP sizes for one scale config) into one PBS job and pool them
# across the node's 4 GPUs via CUDA_VISIBLE_DEVICES — same pattern as
# scripts/derecho/tokenize_all.sh.
#
# Each task:
#   - writes <output_dir>/eval_single_step.json (scalar metrics)
#   - writes <output_dir>/eval_per_timestep.npz, rollout_tokens.npz
#   - writes <output_dir>/{tke_spectrum,enstrophy_spectrum,pixel_histogram}.png
#   - writes <output_dir>/snapshots/t{t:04d}.png for a few sample timesteps
#   - logs to wandb project gust2-eval-derecho
#
# Usage:
#   ./scripts/derecho/sweep_eval.sh sc341                    All 9 sc341 combos
#   ./scripts/derecho/sweep_eval.sh sc917 small              sc917, small NSP only (3)
#   ./scripts/derecho/sweep_eval.sh sc341 --vqvae small      Only small VQ-VAE sources (3)
#   ./scripts/derecho/sweep_eval.sh sc341 --dry-run
#   ./scripts/derecho/sweep_eval.sh --list
#   ./scripts/derecho/sweep_eval.sh --help

set -euo pipefail

# ---------- Paths ----------
REPODIR="${HOME}/gust2"
VENV="${SCRATCH}/.venvs/gust2"
DATA_PATH="${SCRATCH}/turb2d_long/output.h5"
VQVAE_BASE="${SCRATCH}/experiments/vqvae"
TOKENS_BASE="${SCRATCH}/experiments/tokens"
AR_BASE="${SCRATCH}/experiments/ar-refine"
EVAL_BASE="${SCRATCH}/experiments/eval-refine"
WANDB_BASE="${SCRATCH}/wandb"
ACCOUNT="UCSC0009"

WANDB_PROJECT="gust2-eval-derecho"

# ---------- Parse args ----------
SC=""
FILTER_NSP=""
FILTER_VQVAE=""
DRY_RUN=false

print_help() {
    cat <<EOF
Usage: $0 <sc341|sc917|sc1941> [small|medium|large] [--vqvae <size>] [--dry-run] [--list]

Positional:
  <sc341|sc917|sc1941>          Scale config (required).
  [small|medium|large]          Optional NSP size filter.

Options:
  --vqvae <small|medium|large>  Only run combos whose VQ-VAE size matches.
  --dry-run                     Print the PBS script without submitting.
  --list                        Show the task list for the given SC and exit.
  --help, -h                    This message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        sc341|sc917|sc1941) SC="$1"; shift ;;
        small|medium|large) FILTER_NSP="$1"; shift ;;
        --vqvae) FILTER_VQVAE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --list) LIST_ONLY=true; shift ;;
        --help|-h) print_help; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; print_help; exit 1 ;;
    esac
done

if [ -z "${SC}" ]; then
    echo "Error: must provide scale config (sc341|sc917|sc1941)." >&2
    print_help
    exit 1
fi

# ---------- Build task list ----------
VQVAE_SIZES=(small medium large)
if [ -n "${FILTER_VQVAE}" ]; then
    VQVAE_SIZES=("${FILTER_VQVAE}")
fi

NSP_SIZES=(small medium large)
if [ -n "${FILTER_NSP}" ]; then
    NSP_SIZES=("${FILTER_NSP}")
fi

TASKS=()
for VQSIZE in "${VQVAE_SIZES[@]}"; do
    VQVAE_NAME="${VQSIZE}-${SC}"
    for NSP_SIZE in "${NSP_SIZES[@]}"; do
        TASKS+=("${VQVAE_NAME}:${NSP_SIZE}")
    done
done

if [ "${#TASKS[@]}" -eq 0 ]; then
    echo "No tasks to run."
    exit 0
fi

echo "=========================================="
echo "Single-Step Eval Sweep (Derecho, ${SC})"
echo "  VQ-VAE sizes:  ${VQVAE_SIZES[*]}"
echo "  NSP sizes:     ${NSP_SIZES[*]}"
echo "  Total tasks:   ${#TASKS[@]}"
echo "  Dry run:       ${DRY_RUN}"
echo "=========================================="

if [ "${LIST_ONLY:-false}" = true ]; then
    for t in "${TASKS[@]}"; do echo "  ${t/:/-nsp-}"; done
    exit 0
fi

LOG_DIR="${EVAL_BASE}/logs"
if [ "${DRY_RUN}" = false ]; then
    mkdir -p "${EVAL_BASE}" "${LOG_DIR}"
fi

# Filter to only tasks that have prerequisites and aren't already done.
FILTERED_TASKS=()
for t in "${TASKS[@]}"; do
    VQ="${t%%:*}"
    NSP="${t##*:}"
    RUN="${VQ}-nsp-${NSP}"
    CKPT="${AR_BASE}/${RUN}"
    VAL="${TOKENS_BASE}/${VQ}-val.npz"
    VQDIR="${VQVAE_BASE}/${VQ}"
    OUT="${EVAL_BASE}/${RUN}"

    if [ "${DRY_RUN}" = false ]; then
        if [ ! -f "${CKPT}/training_state.json" ]; then
            echo "[skip] ${RUN}: no NSP checkpoint at ${CKPT}"
            continue
        fi
        if [ ! -f "${VAL}" ]; then
            echo "[skip] ${RUN}: no val tokens at ${VAL}"
            continue
        fi
        if [ ! -f "${VQDIR}/training_state.json" ]; then
            echo "[skip] ${RUN}: no VQ-VAE checkpoint at ${VQDIR}"
            continue
        fi
        if [ -f "${OUT}/eval_single_step.json" ]; then
            echo "[skip] ${RUN}: eval already exists"
            continue
        fi
    fi
    FILTERED_TASKS+=("${t}")
done

if [ "${#FILTERED_TASKS[@]}" -eq 0 ]; then
    echo "Nothing to do."
    exit 0
fi

TASK_STR="${FILTERED_TASKS[*]}"

echo ""
echo "Tasks to run (${#FILTERED_TASKS[@]}):"
for t in "${FILTERED_TASKS[@]}"; do echo "  ${t/:/-nsp-}"; done
echo ""

JOB_NAME="eval-${SC}"
TMPFILE="$(mktemp /tmp/sweep_eval_${SC}_XXXXXX.pbs)"
cat > "${TMPFILE}" << PBS_EOF
#!/bin/bash
#PBS -N ${JOB_NAME}
#PBS -A ${ACCOUNT}
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=64:ngpus=4:mpiprocs=1:mem=480GB
#PBS -j oe
#PBS -o ${LOG_DIR}/${JOB_NAME}.log

set -euo pipefail

cd "${REPODIR}"

export TMPDIR="\${SCRATCH}/\${USER}/tmpdir"
mkdir -p "\${TMPDIR}"

module purge
module load ncarenv

source "${VENV}/bin/activate"

# JAX is installed with its own cuda13/cudnn via pip (driver is CUDA 13).
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:       \${PBS_JOBID}"
echo "Node:      \$(hostname)"
echo "Started:   \$(date)"
echo "GPUs:      4x A100-40 (node-exclusive)"
echo "Scale:     ${SC}"
echo "Tasks:     ${TASK_STR}"
echo "=========================================="

run_task() {
    local gpu=\$1
    local spec=\$2
    local vq="\${spec%%:*}"
    local nsp="\${spec##*:}"
    local run="\${vq}-nsp-\${nsp}"

    local ckpt="${AR_BASE}/\${run}"
    local val="${TOKENS_BASE}/\${vq}-val.npz"
    local vqdir="${VQVAE_BASE}/\${vq}"
    local out="${EVAL_BASE}/\${run}"
    local log="${LOG_DIR}/eval-\${run}.log"

    mkdir -p "\${out}"
    echo "[gpu\${gpu}] START \${run}  -> \${log}"
    CUDA_VISIBLE_DEVICES=\${gpu} python eval_single_step.py \\
        --checkpoint_dir "\${ckpt}" \\
        --tokens_path "\${val}" \\
        --vqvae_dir "\${vqdir}" \\
        --data_path "${DATA_PATH}" \\
        --output_dir "\${out}" \\
        --batch_size 64 \\
        --wandb_project ${WANDB_PROJECT} \\
        --wandb_name "\${run}" \\
        --wandb_group "\${nsp}" \\
        --wandb_dir "${WANDB_BASE}" > "\${log}" 2>&1
    echo "[gpu\${gpu}] DONE  \${run}"
}

run_pool() {
    local tasks=("\$@")
    local pid_for_gpu=("" "" "" "")
    local idx=0

    while [ \${idx} -lt \${#tasks[@]} ] || \\
          [ -n "\${pid_for_gpu[0]}" ] || [ -n "\${pid_for_gpu[1]}" ] || \\
          [ -n "\${pid_for_gpu[2]}" ] || [ -n "\${pid_for_gpu[3]}" ]; do

        for g in 0 1 2 3; do
            if [ -z "\${pid_for_gpu[\${g}]}" ] && [ \${idx} -lt \${#tasks[@]} ]; then
                run_task \${g} "\${tasks[\${idx}]}" &
                pid_for_gpu[\${g}]=\$!
                idx=\$((idx + 1))
            fi
        done

        wait -n || true
        for g in 0 1 2 3; do
            if [ -n "\${pid_for_gpu[\${g}]}" ] && ! kill -0 "\${pid_for_gpu[\${g}]}" 2>/dev/null; then
                wait "\${pid_for_gpu[\${g}]}" || true
                pid_for_gpu[\${g}]=""
            fi
        done
    done
}

TASKS=(${TASK_STR})
run_pool "\${TASKS[@]}"

echo ""
echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="

echo ""
echo "Outputs:"
ls -d "${EVAL_BASE}"/*-${SC}-nsp-* 2>/dev/null || echo "  (none)"
PBS_EOF

if [ "${DRY_RUN}" = true ]; then
    echo ""
    echo "[dry-run] PBS script:"
    echo "  ${TMPFILE}"
    echo ""
    cat "${TMPFILE}"
else
    echo "Submitting ${JOB_NAME}..."
    qsub "${TMPFILE}"
    rm -f "${TMPFILE}"
fi
