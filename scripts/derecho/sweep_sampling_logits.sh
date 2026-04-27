#!/bin/bash
# Re-run the same sampling sweep as sweep_sampling.sh, but this time with
# --log_topk so we capture per-emission top-K logits + indices (post
# scale_mask, pre temperature/top_k/top_p truncation). Then per cfg run
# analyze_logits.py to produce per-rollout diagnostic plots.
#
# Tokens come out identical to the original sweep (same seed/temperature),
# only rollout_logits.npz is new alongside the existing rollout_tokens.npz.
#
# Submit:
#   qsub scripts/derecho/sweep_sampling_logits.sh
#
# Outputs (per cfg):
#   $SCRATCH/experiments/sampling-sweep/<RUN>/<cfg>/rollout/rollout_logits.npz
#   $SCRATCH/experiments/sampling-sweep/<RUN>/<cfg>/logits/diagnostics.{png,npz}
#
#PBS -N logits-sweep
#PBS -A UCSC0009
#PBS -q main
#PBS -l select=1:ncpus=64:ngpus=4:mpiprocs=1:mem=480GB
#PBS -l walltime=02:30:00
#PBS -j oe
#PBS -o /glade/derecho/scratch/anishs/sampling_sweep_logits.out

set -euo pipefail

cd $HOME/gust2

# ----------------------------- config ---------------------------------
RUN=small-sc341-nsp-micro
VQ=small-sc341
N_TRAJECTORIES=25
N_STEPS=2000
START_FRAME=0
SEED=0
LOG_TOPK=64

VENV=$SCRATCH/.venvs/gust2
TOKENS=$SCRATCH/experiments/tokens/${VQ}-val.npz
AR_DIR=$SCRATCH/experiments/ar-refine/$RUN

SWEEP_BASE=$SCRATCH/experiments/sampling-sweep/$RUN
LOG_DIR=$SCRATCH/sampling_sweep_logits_logs/$(date +%Y%m%d-%H%M%S)
mkdir -p $LOG_DIR $SWEEP_BASE

# Sweep configs — identical to sweep_sampling.sh.
TASKS=(
  "T10:--temperature 1.0"
  "T09:--temperature 0.9"
  "T08:--temperature 0.8"
  "T07:--temperature 0.7"
  "T10_topp95:--temperature 1.0 --top_p 0.95"
  "T10_topp90:--temperature 1.0 --top_p 0.90"
  "T10_topk50:--temperature 1.0 --top_k 50"
  "T09_topp95:--temperature 0.9 --top_p 0.95"
)

# ----------------------------- env ------------------------------------
export TMPDIR=$SCRATCH/$USER/tmpdir
mkdir -p $TMPDIR

module purge
module load ncarenv
source $VENV/bin/activate

NVIDIA_LIBS=$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=$(find $NVIDIA_LIBS -name "lib" -type d | tr '\n' ':'):${LD_LIBRARY_PATH:-}
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false

echo "=========================================="
echo "Job:        ${PBS_JOBID:-interactive}"
echo "Node:       $(hostname)"
echo "Started:    $(date)"
echo "Run:        ${RUN}"
echo "Configs:    ${#TASKS[@]}"
echo "log_topk:   ${LOG_TOPK}"
echo "Trajectories per config: ${N_TRAJECTORIES}"
echo "Steps:      ${N_STEPS}"
echo "Outputs:    ${SWEEP_BASE}"
echo "Logs:       ${LOG_DIR}"
echo "=========================================="

# ----------------------------- task -----------------------------------
run_task() {
    local gpu=$1
    local spec=$2
    local name=${spec%%:*}
    local args=${spec#*:}
    local roll=$SWEEP_BASE/$name/rollout
    local diag=$SWEEP_BASE/$name/logits
    local log=$LOG_DIR/${name}.log
    mkdir -p "$roll" "$diag"

    echo "[gpu${gpu}] START ${name}  (${args})  -> ${log}"
    {
        echo "=== rollout (--log_topk ${LOG_TOPK}) ==="
        CUDA_VISIBLE_DEVICES=${gpu} python rollout_nsp.py \
            --checkpoint_dir "$AR_DIR" \
            --tokens_path  "$TOKENS" \
            --start_frame ${START_FRAME} \
            --n_steps ${N_STEPS} \
            --n_trajectories ${N_TRAJECTORIES} \
            --seed ${SEED} \
            --log_topk ${LOG_TOPK} \
            ${args} \
            --output_dir "$roll"

        echo "=== analyze_logits ==="
        CUDA_VISIBLE_DEVICES=${gpu} python analyze_logits.py \
            --rollout_dir "$roll" \
            --output_dir  "$diag"
    } > "$log" 2>&1
    local status=$?
    if [ $status -ne 0 ]; then
        echo "[gpu${gpu}] FAILED ${name} (exit ${status}, see ${log})"
    else
        echo "[gpu${gpu}] DONE   ${name}"
    fi
}

# ----------------------------- pool -----------------------------------
run_pool() {
    local tasks=("$@")
    local pid_for_gpu=("" "" "" "")
    local idx=0

    while [ $idx -lt ${#tasks[@]} ] || \
          [ -n "${pid_for_gpu[0]}" ] || [ -n "${pid_for_gpu[1]}" ] || \
          [ -n "${pid_for_gpu[2]}" ] || [ -n "${pid_for_gpu[3]}" ]; do

        for g in 0 1 2 3; do
            if [ -z "${pid_for_gpu[$g]}" ] && [ $idx -lt ${#tasks[@]} ]; then
                run_task $g "${tasks[$idx]}" &
                pid_for_gpu[$g]=$!
                idx=$((idx + 1))
            fi
        done

        wait -n || true
        for g in 0 1 2 3; do
            if [ -n "${pid_for_gpu[$g]}" ] && ! kill -0 "${pid_for_gpu[$g]}" 2>/dev/null; then
                wait "${pid_for_gpu[$g]}" || true
                pid_for_gpu[$g]=""
            fi
        done
    done
}

run_pool "${TASKS[@]}"

# ----------------------------- summary --------------------------------
echo ""
echo "=========================================="
echo "All configs finished at $(date)"
echo ""
echo "=== outputs ==="
for d in $SWEEP_BASE/*/logits; do
    name=$(basename $(dirname $d))
    if [ -f "$d/diagnostics.png" ]; then
        size=$(stat -c%s "$d/../rollout/rollout_logits.npz" 2>/dev/null || echo 0)
        printf "  %-14s  %s  (logits.npz: %.1f GB)\n" \
            "$name" "$d/diagnostics.png" "$(echo "$size / 1073741824" | bc -l)"
    else
        printf "  %-14s  MISSING\n" "$name"
    fi
done
echo "=========================================="
