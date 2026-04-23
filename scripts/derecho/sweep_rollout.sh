#!/bin/bash
# Autoregressive rollout sweep on Derecho: 2000 steps from val frame 0, greedy.
#
# Bundles all (VQ-VAE x NSP) combos for a given scale into one PBS job and
# pools tasks across 4 A100-40 GPUs (same pattern as tokenize_all.sh).
#
# Output: <rollouts-refine>/<run>/rollout_tokens.npz, later consumed by
# sweep_analysis.sh. Rollout itself does not log to wandb.
#
# Usage:
#   ./scripts/derecho/sweep_rollout.sh sc341                All 9 sc341 combos
#   ./scripts/derecho/sweep_rollout.sh sc917 small          sc917, small NSP (3)
#   ./scripts/derecho/sweep_rollout.sh sc341 --vqvae medium Only medium VQ-VAE (3)
#   ./scripts/derecho/sweep_rollout.sh sc341 --dry-run

set -euo pipefail

# ---------- Paths ----------
REPODIR="${HOME}/gust2"
VENV="${SCRATCH}/.venvs/gust2"
TOKENS_BASE="${SCRATCH}/experiments/tokens"
AR_BASE="${SCRATCH}/experiments/ar-refine"
ROLLOUT_BASE="${SCRATCH}/experiments/rollouts-refine"
ACCOUNT="UCSC0009"

N_STEPS=2000
START_FRAME=0

# ---------- Parse args ----------
SC=""
FILTER_NSP=""
FILTER_VQVAE=""
DRY_RUN=false
LIST_ONLY=false
DEPEND_JOBID=""
FORCE=false
TEMPERATURE="1.0"

print_help() {
    cat <<EOF
Usage: $0 <sc341|sc917|sc1941> [nano|micro|mini|small|medium|large]
         [--vqvae <size>] [--temperature <T>] [--force]
         [--depend <jobid>] [--dry-run] [--list]

Positional:
  <sc341|sc917|sc1941>          Scale config (required).
  [nsp-size]                    Optional NSP size filter (default:
                                small,medium,large). Use nano/micro/mini
                                for the sub-small ablation fleet.

Options:
  --vqvae <small|medium|large>  Only run combos whose VQ-VAE size matches.
  --temperature <T>             Sampling temperature (default 1.0; pass 0.0
                                for greedy, but greedy is discouraged on this
                                project — see project memory).
  --force                       Overwrite existing rollout_tokens.npz
                                instead of skipping.
  --depend <jobid>              Submit with PBS afterok dependency on <jobid>.
  --dry-run                     Print the PBS script without submitting.
  --list                        Show the task list for the given SC and exit.
  --help, -h                    This message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        sc341|sc917|sc1941) SC="$1"; shift ;;
        nano|micro|mini|small|medium|large) FILTER_NSP="$1"; shift ;;
        --vqvae) FILTER_VQVAE="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        --depend) DEPEND_JOBID="$2"; shift 2 ;;
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

echo "=========================================="
echo "NSP Rollout Sweep (Derecho, ${SC}, ${N_STEPS} steps, greedy)"
echo "  VQ-VAE sizes:  ${VQVAE_SIZES[*]}"
echo "  NSP sizes:     ${NSP_SIZES[*]}"
echo "  Total tasks:   ${#TASKS[@]}"
echo "  Dry run:       ${DRY_RUN}"
echo "=========================================="

if [ "${LIST_ONLY}" = true ]; then
    for t in "${TASKS[@]}"; do echo "  ${t/:/-nsp-}"; done
    exit 0
fi

LOG_DIR="${ROLLOUT_BASE}/logs"
if [ "${DRY_RUN}" = false ]; then
    mkdir -p "${ROLLOUT_BASE}" "${LOG_DIR}"
fi

FILTERED_TASKS=()
for t in "${TASKS[@]}"; do
    VQ="${t%%:*}"
    NSP="${t##*:}"
    RUN="${VQ}-nsp-${NSP}"
    CKPT="${AR_BASE}/${RUN}"
    VAL="${TOKENS_BASE}/${VQ}-val.npz"
    OUT="${ROLLOUT_BASE}/${RUN}"

    if [ "${DRY_RUN}" = false ]; then
        if [ ! -f "${CKPT}/training_state.json" ]; then
            echo "[skip] ${RUN}: no NSP checkpoint at ${CKPT}"
            continue
        fi
        if [ ! -f "${VAL}" ]; then
            echo "[skip] ${RUN}: no val tokens at ${VAL}"
            continue
        fi
        if [ "${FORCE}" = false ] && [ -f "${OUT}/rollout_tokens.npz" ]; then
            echo "[skip] ${RUN}: rollout already exists (pass --force to overwrite)"
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

JOB_NAME="rollout-${SC}"
TMPFILE="$(mktemp /tmp/sweep_rollout_${SC}_XXXXXX.pbs)"
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

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:       \${PBS_JOBID}"
echo "Node:      \$(hostname)"
echo "Started:   \$(date)"
echo "GPUs:      4x A100-40 (node-exclusive)"
echo "Scale:     ${SC}"
echo "Steps:     ${N_STEPS}  (start_frame=${START_FRAME})"
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
    local out="${ROLLOUT_BASE}/\${run}"
    local log="${LOG_DIR}/rollout-\${run}.log"

    mkdir -p "\${out}"
    echo "[gpu\${gpu}] START \${run}  -> \${log}"
    CUDA_VISIBLE_DEVICES=\${gpu} python rollout_nsp.py \\
        --checkpoint_dir "\${ckpt}" \\
        --tokens_path "\${val}" \\
        --start_frame ${START_FRAME} \\
        --n_steps ${N_STEPS} \\
        --temperature ${TEMPERATURE} \\
        --output_dir "\${out}" > "\${log}" 2>&1
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
echo "Rollout files:"
ls "${ROLLOUT_BASE}"/*-${SC}-nsp-*/rollout_tokens.npz 2>/dev/null || echo "  (none)"
PBS_EOF

if [ "${DRY_RUN}" = true ]; then
    echo ""
    echo "[dry-run] PBS script:"
    echo "  ${TMPFILE}"
    echo ""
    cat "${TMPFILE}"
else
    QSUB_ARGS=()
    if [ -n "${DEPEND_JOBID}" ]; then
        QSUB_ARGS+=(-W "depend=afterok:${DEPEND_JOBID}")
    fi
    echo "Submitting ${JOB_NAME}..."
    JOBID=$(qsub "${QSUB_ARGS[@]}" "${TMPFILE}")
    rm -f "${TMPFILE}"
    echo "Submitted: ${JOBID}"
    # Parseable marker so orchestration scripts (e.g. sweep_multistep.sh)
    # can chain an analysis job with -W depend=afterok:<jobid>.
    echo "JOBID=${JOBID}"
fi
