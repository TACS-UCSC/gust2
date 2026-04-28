#!/bin/bash
# Causal test of the per-position-OOD hypothesis:
# rerun the same 8-config sampling sweep as sweep_sampling.sh, but force
# rollout_nsp.py to AND a per-position vocabulary mask (built from the
# training tokens) with the per-scale scale_mask at every emission. Each
# position is then restricted to tokens ever observed at that exact
# (scale, row, col) during training -- closing the "scale-legal but
# position-unseen" gap that the position_ood diagnostic identified.
#
# If survival improves dramatically vs the no-mask baseline, position-OOD
# emission is causally important and we have a free rollout-time fix.
# If it doesn't move, the marginal-per-position view isn't enough and
# the trigger lives in joint token statistics.
#
# Submit:
#   qsub scripts/derecho/sweep_sampling_posmask.sh
#
# Outputs:
#   $SCRATCH/experiments/sampling-sweep-posmask/<RUN>/<cfg_name>/{rollout,analysis}/
#   wandb project: gust2-sampling-posmask-derecho
#                  (group=<RUN>, name=<RUN>-<cfg>)
#
#PBS -N sampling-posmask
#PBS -A UCSC0009
#PBS -q main
#PBS -l select=1:ncpus=64:ngpus=4:mpiprocs=1:mem=480GB
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /glade/derecho/scratch/anishs/sampling_sweep_posmask.out

set -euo pipefail

cd $HOME/gust2

# ----------------------------- config ---------------------------------
RUN=small-sc341-nsp-micro
VQ=small-sc341
N_TRAJECTORIES=25
N_STEPS=2000
START_FRAME=0
SEED=0

VENV=$SCRATCH/.venvs/gust2
DATA_PATH=$SCRATCH/turb2d_long/output.h5
TOKENS=$SCRATCH/experiments/tokens/${VQ}-val.npz
TRAIN_TOKENS=$SCRATCH/experiments/tokens/${VQ}.npz   # train; for pos-mask
AR_DIR=$SCRATCH/experiments/ar-refine/$RUN
VQVAE_DIR=$SCRATCH/experiments/vqvae/$VQ
WANDB_BASE=$SCRATCH/wandb
WANDB_PROJECT=gust2-sampling-posmask-derecho

SWEEP_BASE=$SCRATCH/experiments/sampling-sweep-posmask/$RUN
LOG_DIR=$SCRATCH/sampling_sweep_posmask_logs/$(date +%Y%m%d-%H%M%S)
mkdir -p $LOG_DIR $SWEEP_BASE

# Sweep configs -- identical to sweep_sampling.sh so survival outcomes are
# directly comparable cfg-by-cfg.
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

if [ ! -f "$TRAIN_TOKENS" ]; then
    echo "ERROR: train tokens not found at $TRAIN_TOKENS"
    echo "       (--train_tokens_path is required to build per-position mask)"
    exit 1
fi

echo "=========================================="
echo "Job:        ${PBS_JOBID:-interactive}"
echo "Node:       $(hostname)"
echo "Started:    $(date)"
echo "Run:        ${RUN}"
echo "Configs:    ${#TASKS[@]}"
echo "Trajectories per config: ${N_TRAJECTORIES}"
echo "Steps:      ${N_STEPS}"
echo "Train tok:  ${TRAIN_TOKENS} (per-position mask source)"
echo "Wandb:      ${WANDB_PROJECT}"
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
    local ana=$SWEEP_BASE/$name/analysis
    local log=$LOG_DIR/${name}.log
    mkdir -p "$roll" "$ana"

    echo "[gpu${gpu}] START ${name}  (${args})  -> ${log}"
    {
        echo "=== rollout (per-position mask ON) ==="
        CUDA_VISIBLE_DEVICES=${gpu} python rollout_nsp.py \
            --checkpoint_dir "$AR_DIR" \
            --tokens_path  "$TOKENS" \
            --train_tokens_path "$TRAIN_TOKENS" \
            --start_frame ${START_FRAME} \
            --n_steps ${N_STEPS} \
            --n_trajectories ${N_TRAJECTORIES} \
            --seed ${SEED} \
            ${args} \
            --output_dir "$roll"

        echo "=== analysis ==="
        CUDA_VISIBLE_DEVICES=${gpu} python analyze_rollout.py \
            --rollout_dir "$roll" \
            --vqvae_dir   "$VQVAE_DIR" \
            --data_path   "$DATA_PATH" \
            --output_dir  "$ana" \
            --batch_size 64 \
            --wandb_project ${WANDB_PROJECT} \
            --wandb_name   "${RUN}-${name}" \
            --wandb_group  "${RUN}" \
            --wandb_dir    "$WANDB_BASE"

        echo "=== snapshot grid ==="
        CUDA_VISIBLE_DEVICES=${gpu} python multitraj_snapshot_grid.py \
            --rollout_dir "$roll" \
            --vqvae_dir   "$VQVAE_DIR" \
            --data_path   "$DATA_PATH" \
            --output_path "$ana/multitraj_grid.png"
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
echo "=== summary (sorted by collapse_rate) ==="
python - <<EOF
import json, os
import glob

base = "$SWEEP_BASE"
rows = []
for d in sorted(glob.glob(os.path.join(base, "*"))):
    name = os.path.basename(d)
    mp = os.path.join(d, "analysis", "metrics.json")
    if not os.path.isfile(mp):
        continue
    m = json.load(open(mp))
    rows.append((m["collapse_rate"], name, m))

rows.sort()
print(f"{'cfg':<14}  {'collapse':>8}  {'emd_mean':>9}  {'emd_std':>8}  "
      f"{'emd_max':>8}  {'tke_rse':>8}  {'ens_rse':>8}")
for cr, name, m in rows:
    print(f"{name:<14}  {cr:>7.1%}  {m['emd_nsp_mean']:>9.3f}  "
          f"{m['emd_nsp_std']:>8.3f}  {m['emd_nsp_max']:>8.3f}  "
          f"{m['tke_rse_nsp_mean']:>8.3f}  {m['enstrophy_rse_nsp_mean']:>8.3f}")
EOF
echo "=========================================="
