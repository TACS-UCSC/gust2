#!/bin/bash
# Submit a single training job on Bridges2.
#
# Usage:
#   ./scripts/train_single.sh <run_name> <n_gpus> [extra train.py args...]
#
# Examples:
#   ./scripts/train_single.sh test-small 1 --epochs 2 --batch_size 4
#   ./scripts/train_single.sh train-medium 2 --encoder_depth 10 --decoder_depth 10
#   ./scripts/train_single.sh train-large 4 --encoder_depth 20 --decoder_depth 20

set -euo pipefail

RUN_NAME="${1:?Usage: $0 <run_name> <n_gpus> [extra args...]}"
N_GPUS="${2:?Usage: $0 <run_name> <n_gpus> [extra args...]}"
shift 2

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
CHECKPOINT_DIR="${OCEAN}/checkpoints/${RUN_NAME}"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- GPU partition ----------
if [ "${N_GPUS}" -le 4 ]; then
    PARTITION="GPU-shared"
else
    PARTITION="GPU"
fi

# ---------- Auto-detect resume ----------
RESUME_FLAG=""
if [ -f "${CHECKPOINT_DIR}/training_state.json" ]; then
    RESUME_FLAG="--resume"
fi

mkdir -p "${OCEAN}/checkpoints/logs"

sbatch << SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:${N_GPUS}
#SBATCH -t 2-00:00:00
#SBATCH -o ${OCEAN}/checkpoints/logs/${RUN_NAME}-%j.out
#SBATCH -e ${OCEAN}/checkpoints/logs/${RUN_NAME}-%j.err

cd "${REPODIR}"
source /ocean/projects/mth260004p/sambamur/.venvs/gust/bin/activate
module load cuda/12.6.1

# cuDNN and NVIDIA libs from pip packages
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\$LD_LIBRARY_PATH

mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${WANDB_BASE}"

echo "=========================================="
echo "Job:       \${SLURM_JOB_ID}"
echo "Node:      \$(hostname)"
echo "Started:   \$(date)"
echo "GPUs:      ${N_GPUS}"
echo "Run:       ${RUN_NAME}"
echo "Ckpt dir:  ${CHECKPOINT_DIR}"
echo "Resume:    ${RESUME_FLAG:-no}"
echo "=========================================="

python train.py \\
    --data_path "${DATA_PATH}" \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --wandb_project gust2-vqvae \\
    --wandb_name ${RUN_NAME} \\
    --wandb_dir "${WANDB_BASE}" \\
    ${RESUME_FLAG} \\
    $@

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
SBATCH_EOF

echo "Submitted ${RUN_NAME} on ${N_GPUS} GPU(s) (${PARTITION})"
