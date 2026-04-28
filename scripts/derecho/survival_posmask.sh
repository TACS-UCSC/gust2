#!/bin/bash
# One-shot multitraj_survival.py over the posmask sampling sweep.
# Decodes every rollout trajectory through the VQ-VAE, computes windowed
# pixel-EMD vs raw GT at probe times, and writes survival curves +
# explosion times. Mirrors the prior survival run on the no-mask sweep,
# so the two output dirs can be overlaid head-to-head.
#
# Submit:
#   qsub scripts/derecho/survival_posmask.sh
#
# Outputs:
#   $SCRATCH/experiments/sampling-sweep-posmask/<RUN>/survival/
#     survival.json, survival_data.npz, survival_curves.png, emd_traces.png
#
#PBS -N survival-posmask
#PBS -A UCSC0009
#PBS -q main
#PBS -l select=1:ncpus=64:ngpus=4:mpiprocs=1:mem=480GB
#PBS -l walltime=00:45:00
#PBS -j oe
#PBS -o /glade/derecho/scratch/anishs/survival_posmask.out

set -euo pipefail

cd $HOME/gust2

RUN=small-sc341-nsp-micro
VQ=small-sc341
SWEEP_BASE=$SCRATCH/experiments/sampling-sweep-posmask/$RUN
VQVAE_DIR=$SCRATCH/experiments/vqvae/$VQ
DATA_PATH=$SCRATCH/turb2d_long/output.h5
OUTPUT_DIR=$SWEEP_BASE/survival
VENV=$SCRATCH/.venvs/gust2

mkdir -p $OUTPUT_DIR

export TMPDIR=$SCRATCH/$USER/tmpdir
mkdir -p $TMPDIR

module purge
module load ncarenv
source $VENV/bin/activate

NVIDIA_LIBS=$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=$(find $NVIDIA_LIBS -name "lib" -type d | tr '\n' ':'):${LD_LIBRARY_PATH:-}
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Started: $(date)"
echo "Sweep:   $SWEEP_BASE"
echo "Outputs: $OUTPUT_DIR"
echo "=========================================="

python multitraj_survival.py \
    --sweep_root  "$SWEEP_BASE" \
    --vqvae_dir   "$VQVAE_DIR" \
    --data_path   "$DATA_PATH" \
    --output_dir  "$OUTPUT_DIR"

echo "Done: $(date)"
