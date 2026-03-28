#!/bin/bash
# Single run: beta=0 test with best config (cd512-K4096, decay=0.85)

set -euo pipefail

OCEAN="/ocean/projects/mth260004p/sambamur"
CHECKPOINT_DIR="${OCEAN}/sweeps/dynamics/d0.85-b0.0"

mkdir -p "${CHECKPOINT_DIR}" "${OCEAN}/sweeps/dynamics/logs" "${OCEAN}/wandb"

sbatch << 'SBATCH_EOF'
#!/bin/bash
#SBATCH -J d0.85-b0.0
#SBATCH -A mth260004p
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 12:00:00
#SBATCH -o /ocean/projects/mth260004p/sambamur/sweeps/dynamics/logs/d0.85-b0.0-%j.out
#SBATCH -e /ocean/projects/mth260004p/sambamur/sweeps/dynamics/logs/d0.85-b0.0-%j.err

cd /ocean/projects/mth260004p/sambamur/gust
source /ocean/projects/mth260004p/sambamur/.venvs/gust/bin/activate
module load cuda/12.6.1
NVIDIA_LIBS=$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=$(find $NVIDIA_LIBS -name "lib" -type d | tr '\n' ':'):$LD_LIBRARY_PATH

python train.py \
    --data_path /ocean/projects/mth260004p/sambamur/data_lowres/output.h5 \
    --checkpoint_dir /ocean/projects/mth260004p/sambamur/sweeps/dynamics/d0.85-b0.0 \
    --d_model 512 --n_heads 8 --mlp_dim 1024 \
    --encoder_depth 5 --decoder_depth 5 \
    --codebook_dim 512 --codebook_size 4096 \
    --scales 1,2,4,8,16,32 \
    --batch_size 64 --epochs 50 --lr 1e-4 \
    --beta 0.0 --ema_decay 0.85 \
    --wandb_project gust2-vqvae --wandb_name d0.85-b0.0 \
    --wandb_group dynamics-sweep \
    --wandb_dir /ocean/projects/mth260004p/sambamur/wandb
SBATCH_EOF
