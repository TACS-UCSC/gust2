#!/bin/bash
# Tokenize all 9 VQ-VAE checkpoints on Derecho: both training and validation.
#
# Since Derecho `main` is node-exclusive (1 job = full 4-GPU node), we submit
# a single PBS job that runs 4 tokenizations in parallel across the node's
# 4 A100-40GB GPUs. Phase 1: 9 training tokenize tasks (frames 0..20000).
# Phase 2: 9 validation tokenize tasks (frames 20000..22000), each using
# `--fit_from <train>.npz` to share the training compact index mapping.
#
# Usage:
#   ./scripts/derecho/tokenize_all.sh              Submit 1 job
#   ./scripts/derecho/tokenize_all.sh --dry-run    Print script, do not submit
#   ./scripts/derecho/tokenize_all.sh --force      Re-tokenize even if .npz exists

set -euo pipefail

# ---------- Paths ----------
REPODIR="${HOME}/gust2"
VENV="${SCRATCH}/.venvs/gust2"
DATA_PATH="${SCRATCH}/turb2d_long/output.h5"
VQVAE_BASE="${SCRATCH}/experiments/vqvae"
TOKENS_BASE="${SCRATCH}/experiments/tokens"
ACCOUNT="UCSC0009"

# ---------- Validation window ----------
# Training uses frames 0..20000; validation uses the next 2k frames.
VAL_SAMPLE_START=20000
VAL_SAMPLE_STOP=22000

# ---------- Configurations ----------
VQVAE_SIZES=(small medium large)
SCALE_CONFIGS=(sc341 sc917 sc1941)

# ---------- Parse args ----------
DRY_RUN=false
FORCE=false
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=true ;;
        --force) FORCE=true ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--force]"
            exit 0
            ;;
    esac
done

LOG_DIR="${VQVAE_BASE}/logs"
mkdir -p "${TOKENS_BASE}" "${LOG_DIR}"

# ---------- Build task lists ----------
TRAIN_TASKS=()
VAL_TASKS=()
for SIZE in "${VQVAE_SIZES[@]}"; do
    for SC in "${SCALE_CONFIGS[@]}"; do
        RUN_NAME="${SIZE}-${SC}"
        CHECKPOINT_DIR="${VQVAE_BASE}/${RUN_NAME}"
        TRAIN_OUT="${TOKENS_BASE}/${RUN_NAME}.npz"
        VAL_OUT="${TOKENS_BASE}/${RUN_NAME}-val.npz"

        if [ ! -f "${CHECKPOINT_DIR}/training_state.json" ]; then
            echo "[skip] ${RUN_NAME}: no checkpoint"
            continue
        fi

        if [ -f "${TRAIN_OUT}" ] && [ "${FORCE}" = false ]; then
            echo "[skip-train] ${RUN_NAME}: ${TRAIN_OUT} exists"
        else
            TRAIN_TASKS+=("${RUN_NAME}")
        fi

        if [ -f "${VAL_OUT}" ] && [ "${FORCE}" = false ]; then
            echo "[skip-val]   ${RUN_NAME}: ${VAL_OUT} exists"
        else
            VAL_TASKS+=("${RUN_NAME}")
        fi
    done
done

echo ""
echo "Train tasks: ${#TRAIN_TASKS[@]}"
echo "Val tasks:   ${#VAL_TASKS[@]}"

if [ "${#TRAIN_TASKS[@]}" -eq 0 ] && [ "${#VAL_TASKS[@]}" -eq 0 ]; then
    echo "Nothing to do."
    exit 0
fi

# Serialize task lists for embedding in the PBS script.
TRAIN_TASKS_STR="${TRAIN_TASKS[*]:-}"
VAL_TASKS_STR="${VAL_TASKS[*]:-}"

TMPFILE="$(mktemp /tmp/tokenize_all_XXXXXX.pbs)"
cat > "${TMPFILE}" << PBS_EOF
#!/bin/bash
#PBS -N tokenize-all
#PBS -A ${ACCOUNT}
#PBS -q main
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=64:ngpus=4:mpiprocs=1:mem=480GB
#PBS -j oe
#PBS -o ${LOG_DIR}/tokenize-all.log

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
echo "Train:     ${TRAIN_TASKS_STR}"
echo "Val:       ${VAL_TASKS_STR}"
echo "=========================================="

# ---------- Worker: run one tokenize task on a specific GPU ----------
# Args: <gpu_id> <mode:train|val> <run_name>
run_task() {
    local gpu=\$1
    local mode=\$2
    local run=\$3

    local ckpt="${VQVAE_BASE}/\${run}"
    local train_out="${TOKENS_BASE}/\${run}.npz"
    local log="${LOG_DIR}/tok-\${mode}-\${run}.log"

    echo "[gpu\${gpu}] START \${mode} \${run}  -> \${log}"
    if [ "\${mode}" = "train" ]; then
        CUDA_VISIBLE_DEVICES=\${gpu} python tokenizer.py save \\
            --checkpoint_dir "\${ckpt}" \\
            --data_path "${DATA_PATH}" \\
            --output "\${train_out}" \\
            --sample_stop 20000 \\
            --batch_size 128 > "\${log}" 2>&1
    else
        local val_out="${TOKENS_BASE}/\${run}-val.npz"
        CUDA_VISIBLE_DEVICES=\${gpu} python tokenizer.py save \\
            --checkpoint_dir "\${ckpt}" \\
            --data_path "${DATA_PATH}" \\
            --output "\${val_out}" \\
            --sample_start ${VAL_SAMPLE_START} \\
            --sample_stop ${VAL_SAMPLE_STOP} \\
            --batch_size 128 \\
            --fit_from "\${train_out}" > "\${log}" 2>&1
    fi
    echo "[gpu\${gpu}] DONE  \${mode} \${run}"
}

# ---------- Pool: dispatch tasks across 4 GPUs ----------
# Keeps 4 tasks running concurrently; starts a new one whenever a slot frees.
run_pool() {
    local mode=\$1
    shift
    local tasks=("\$@")

    # pid_for_gpu[g] holds the current worker pid on gpu g (empty = free).
    local pid_for_gpu=("" "" "" "")
    local idx=0

    while [ \${idx} -lt \${#tasks[@]} ] || \\
          [ -n "\${pid_for_gpu[0]}" ] || [ -n "\${pid_for_gpu[1]}" ] || \\
          [ -n "\${pid_for_gpu[2]}" ] || [ -n "\${pid_for_gpu[3]}" ]; do

        # Fill any free slot with the next task.
        for g in 0 1 2 3; do
            if [ -z "\${pid_for_gpu[\${g}]}" ] && [ \${idx} -lt \${#tasks[@]} ]; then
                run_task \${g} "\${mode}" "\${tasks[\${idx}]}" &
                pid_for_gpu[\${g}]=\$!
                idx=\$((idx + 1))
            fi
        done

        # Wait for any single child to exit, then mark its slot free.
        wait -n || true
        for g in 0 1 2 3; do
            if [ -n "\${pid_for_gpu[\${g}]}" ] && ! kill -0 "\${pid_for_gpu[\${g}]}" 2>/dev/null; then
                wait "\${pid_for_gpu[\${g}]}" || true
                pid_for_gpu[\${g}]=""
            fi
        done
    done
}

# ---------- Phase 1: training tokens ----------
TRAIN_TASKS=(${TRAIN_TASKS_STR})
if [ \${#TRAIN_TASKS[@]} -gt 0 ]; then
    echo ""
    echo "--- Phase 1: training tokens (\${#TRAIN_TASKS[@]} tasks) ---"
    run_pool train "\${TRAIN_TASKS[@]}"
fi

# ---------- Phase 2: validation tokens (requires train .npz to exist) ----------
VAL_TASKS=(${VAL_TASKS_STR})
if [ \${#VAL_TASKS[@]} -gt 0 ]; then
    echo ""
    echo "--- Phase 2: validation tokens (\${#VAL_TASKS[@]} tasks) ---"
    run_pool val "\${VAL_TASKS[@]}"
fi

echo ""
echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="

# Summary
echo ""
echo "Output files:"
ls -la "${TOKENS_BASE}"/*.npz 2>/dev/null || echo "  (none)"
PBS_EOF

if [ "${DRY_RUN}" = true ]; then
    echo ""
    echo "[dry-run] PBS script:"
    echo "  ${TMPFILE}"
    echo ""
    cat "${TMPFILE}"
else
    echo ""
    echo "Submitting tokenize-all..."
    qsub "${TMPFILE}"
    rm -f "${TMPFILE}"
fi
