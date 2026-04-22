#!/bin/bash
# Spectral + histogram + snapshot analysis sweep on Derecho.
#
# Consumes <rollouts-refine>/<run>/rollout_tokens.npz produced by
# sweep_rollout.sh. Writes time-averaged TKE/enstrophy spectra, pixel
# histograms, scalar metrics, and snapshot PNGs at
# t = 1, 2, 5, 10, 50, 100, 250, 500, 1000, 1500, 2000 (see analyze_rollout.py).
#
# Run AFTER sweep_rollout.sh has finished for the same <sc>.
#
# Usage:
#   ./scripts/derecho/sweep_analysis.sh sc341                     All 9 sc341 combos
#   ./scripts/derecho/sweep_analysis.sh sc917 small               sc917, small NSP (3)
#   ./scripts/derecho/sweep_analysis.sh sc341 --vqvae large       Only large VQ-VAE (3)
#   ./scripts/derecho/sweep_analysis.sh sc341 --dry-run

set -euo pipefail

# ---------- Paths ----------
REPODIR="${HOME}/gust2"
VENV="${SCRATCH}/.venvs/gust2"
DATA_PATH="${SCRATCH}/turb2d_long/output.h5"
VQVAE_BASE="${SCRATCH}/experiments/vqvae"
ROLLOUT_BASE="${SCRATCH}/experiments/rollouts-refine"
ANALYSIS_BASE="${SCRATCH}/experiments/analysis-refine"
WANDB_BASE="${SCRATCH}/wandb"
ACCOUNT="UCSC0009"

WANDB_PROJECT="gust2-analysis-derecho"

# ---------- Parse args ----------
SC=""
FILTER_NSP=""
FILTER_VQVAE=""
DRY_RUN=false
LIST_ONLY=false
DEPEND_JOBID=""

print_help() {
    cat <<EOF
Usage: $0 <sc341|sc917|sc1941> [small|medium|large] [--vqvae <size>] [--depend <jobid>] [--dry-run] [--list]

Positional:
  <sc341|sc917|sc1941>          Scale config (required).
  [small|medium|large]          Optional NSP size filter.

Options:
  --vqvae <small|medium|large>  Only run combos whose VQ-VAE size matches.
  --depend <jobid>              Submit with PBS afterok dependency on <jobid>.
                                Useful for chaining after sweep_rollout.sh.
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
echo "Rollout Analysis Sweep (Derecho, ${SC})"
echo "  VQ-VAE sizes:  ${VQVAE_SIZES[*]}"
echo "  NSP sizes:     ${NSP_SIZES[*]}"
echo "  Total tasks:   ${#TASKS[@]}"
echo "  Dry run:       ${DRY_RUN}"
echo "=========================================="

if [ "${LIST_ONLY}" = true ]; then
    for t in "${TASKS[@]}"; do echo "  ${t/:/-nsp-}"; done
    exit 0
fi

LOG_DIR="${ANALYSIS_BASE}/logs"
if [ "${DRY_RUN}" = false ]; then
    mkdir -p "${ANALYSIS_BASE}" "${LOG_DIR}"
fi

FILTERED_TASKS=()
for t in "${TASKS[@]}"; do
    VQ="${t%%:*}"
    NSP="${t##*:}"
    RUN="${VQ}-nsp-${NSP}"
    ROLLOUT_DIR="${ROLLOUT_BASE}/${RUN}"
    VQDIR="${VQVAE_BASE}/${VQ}"
    OUT="${ANALYSIS_BASE}/${RUN}"

    if [ "${DRY_RUN}" = false ]; then
        # When chained via --depend, rollout_tokens.npz will only exist after
        # the dependency job runs; skip the on-disk check in that case.
        if [ -z "${DEPEND_JOBID}" ] && [ ! -f "${ROLLOUT_DIR}/rollout_tokens.npz" ]; then
            echo "[skip] ${RUN}: no rollout at ${ROLLOUT_DIR}"
            continue
        fi
        if [ ! -f "${VQDIR}/training_state.json" ]; then
            echo "[skip] ${RUN}: no VQ-VAE checkpoint at ${VQDIR}"
            continue
        fi
        if [ -f "${OUT}/metrics.json" ]; then
            echo "[skip] ${RUN}: analysis already exists"
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

JOB_NAME="analysis-${SC}"
TMPFILE="$(mktemp /tmp/sweep_analysis_${SC}_XXXXXX.pbs)"
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
echo "Tasks:     ${TASK_STR}"
echo "=========================================="

run_task() {
    local gpu=\$1
    local spec=\$2
    local vq="\${spec%%:*}"
    local nsp="\${spec##*:}"
    local run="\${vq}-nsp-\${nsp}"

    local rollout="${ROLLOUT_BASE}/\${run}"
    local vqdir="${VQVAE_BASE}/\${vq}"
    local out="${ANALYSIS_BASE}/\${run}"
    local log="${LOG_DIR}/analysis-\${run}.log"
    local vqsize="\${vq%%-*}"

    mkdir -p "\${out}"
    echo "[gpu\${gpu}] START \${run}  -> \${log}"
    CUDA_VISIBLE_DEVICES=\${gpu} python analyze_rollout.py \\
        --rollout_dir "\${rollout}" \\
        --vqvae_dir "\${vqdir}" \\
        --data_path "${DATA_PATH}" \\
        --output_dir "\${out}" \\
        --batch_size 64 \\
        --wandb_project ${WANDB_PROJECT} \\
        --wandb_name "\${run}" \\
        --wandb_group "\${vqsize}-nsp-\${nsp}" \\
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
echo "Analysis outputs:"
ls "${ANALYSIS_BASE}"/*-${SC}-nsp-*/metrics.json 2>/dev/null || echo "  (none)"
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
    echo "JOBID=${JOBID}"
fi
