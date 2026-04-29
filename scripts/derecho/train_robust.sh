#!/bin/bash
# One-off retrain of small-sc341-nsp-micro with robustness changes:
# (1) per-position vocabulary mask in the loss (replacing the per-scale
#     mask), and (2) same-position-token substitution noise on the input
#     (replacing the prior zero-token context dropout, which is now
#     removed entirely from the codebase).
#
# Architecture and base hyperparameters match sweep_nsp.sh's nsp-micro
# config so the new model is directly comparable to the existing
# ar-refine/small-sc341-nsp-micro checkpoint used in the sampling sweep.
#
# Submit:
#   qsub scripts/derecho/train_robust.sh
#
# Resubmit (auto-resumes via training_state.json):
#   qsub scripts/derecho/train_robust.sh
#
# Outputs:
#   $SCRATCH/experiments/ar-robust/small-sc341-nsp-micro/
#   wandb project: gust2-nsp-robust-derecho (group=small-sc341-nsp-micro)
#
#PBS -N train-robust
#PBS -A UCSC0009
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=64:ngpus=4:mpiprocs=1:mem=480GB
#PBS -j oe
#PBS -o /glade/derecho/scratch/anishs/train_robust.log

set -euo pipefail

cd $HOME/gust2

# ----------------------------- config ---------------------------------
# Match sweep_nsp.sh nsp-micro so we can compare head-to-head against the
# existing ar-refine/small-sc341-nsp-micro checkpoint.
RUN_NAME=small-sc341-nsp-micro
VQ=small-sc341

# nsp-micro architecture (from sweep_nsp.sh:set_nsp_size).
N_LAYER=3
N_EMBD=384
N_HEAD=6
N_REFINE_LAYERS=2

# Training hyperparameters mirror sweep_nsp.sh exactly.
BATCH_SIZE=64
EPOCHS=400
LR=1e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
SAVE_EVERY=5
SEED=42

# Robustness knobs.
SUBSTITUTION_RATE=0.1     # per-token Bernoulli rate; tunable

VENV=$SCRATCH/.venvs/gust2
TOKENS_PATH=$SCRATCH/experiments/tokens/${VQ}.npz
TRAIN_TOKENS=$SCRATCH/experiments/tokens/${VQ}.npz
CHECKPOINT_DIR=$SCRATCH/experiments/ar-robust/${RUN_NAME}
WANDB_BASE=$SCRATCH/wandb
WANDB_PROJECT=gust2-nsp-robust-derecho
WANDB_GROUP=${RUN_NAME}
WANDB_NAME=${RUN_NAME}-r${SUBSTITUTION_RATE}

# Persistent wandb id (first submit creates, later submits reuse).
ID_FILE=${CHECKPOINT_DIR}/wandb_id.txt
mkdir -p ${CHECKPOINT_DIR}
if [ -f "${ID_FILE}" ]; then
    WANDB_ID=$(cat ${ID_FILE})
else
    WANDB_ID=$(head /dev/urandom | tr -dc 'a-z0-9' | head -c 8)
    echo ${WANDB_ID} > ${ID_FILE}
fi

# Auto-resume if a checkpoint already exists.
RESUME_FLAG=""
if [ -f "${CHECKPOINT_DIR}/training_state.json" ]; then
    RESUME_FLAG="--resume"
fi

# ----------------------------- env ------------------------------------
export TMPDIR=$SCRATCH/$USER/tmpdir
mkdir -p $TMPDIR

module purge
module load ncarenv
source $VENV/bin/activate

NVIDIA_LIBS=$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=$(find $NVIDIA_LIBS -name "lib" -type d | tr '\n' ':'):${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:                ${PBS_JOBID:-interactive}"
echo "Node:               $(hostname)"
echo "Started:            $(date)"
echo "Run:                ${RUN_NAME}"
echo "Arch:               ${N_LAYER}L, n_embd=${N_EMBD}, n_head=${N_HEAD}, refine=${N_REFINE_LAYERS}"
echo "Tokens:             ${TOKENS_PATH}"
echo "Train tokens (mask): ${TRAIN_TOKENS}"
echo "Substitution rate:  ${SUBSTITUTION_RATE}"
echo "Batch:              ${BATCH_SIZE}  (16/GPU)"
echo "Epochs:             ${EPOCHS}"
echo "Ckpt dir:           ${CHECKPOINT_DIR}"
echo "Wandb:              ${WANDB_PROJECT} / ${WANDB_NAME} (id=${WANDB_ID})"
echo "Resume:             ${RESUME_FLAG:-no}"
echo "=========================================="

python train_nsp.py \
    --tokens_path "${TOKENS_PATH}" \
    --train_tokens_path "${TRAIN_TOKENS}" \
    --substitution_rate ${SUBSTITUTION_RATE} \
    --n_layer ${N_LAYER} \
    --n_head ${N_HEAD} \
    --n_embd ${N_EMBD} \
    --n_refine_layers ${N_REFINE_LAYERS} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --grad_clip ${GRAD_CLIP} \
    --save_every ${SAVE_EVERY} \
    --seed ${SEED} \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_name ${WANDB_NAME} \
    --wandb_group ${WANDB_GROUP} \
    --wandb_dir "${WANDB_BASE}" \
    --wandb_id ${WANDB_ID} \
    ${RESUME_FLAG}

echo "=========================================="
echo "Finished:  $(date)"
echo "=========================================="
