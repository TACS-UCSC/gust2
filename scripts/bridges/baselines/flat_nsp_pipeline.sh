#!/bin/bash
# Flat-token NSP dynamics — does the B3 recon parity hold under long AR?
#
# The 2026-07-07 recon-floor verdict: flat-1024 ties sc917 on every val recon
# metric. This pipeline tests the DYNAMICS half: a degenerate single-scale
# NSP on flat-sc1024 tokens predicts all 1024 tokens of t+1 in ONE shot from
# t0's hidden states (nsp_model's scale_idx==0 path + shared refinement
# stack; truncated t1 is empty, so teacher forcing == generation compute).
# The matrix cell this fills: latent-discrete-flat vs our latent-discrete-
# multi-scale, i.e. the classify-analogue of B2's next-ViT.
#
# Reading the outcome:
#   flat holds over 2000 steps  -> M6 reduces further (interface = speed +
#                                  calibration hooks only)
#   flat collapses / cannot be temperature-rescued (no scale axis for the
#   fine-heat schedule)         -> direct evidence FOR the multi-scale
#                                  interface claim
#
# Chain: tokenize (train+val, skip-guarded) -> 2 NSP trainings (s13/s34-class
# arch match to the sc917 comparators, substitution 0.1 = E2 canonical) ->
# per-(arch x temp) rollout+analysis fan. All steps skip-resume.
#
# Usage:
#   ./scripts/bridges/baselines/flat_nsp_pipeline.sh              # everything
#   ./scripts/bridges/baselines/flat_nsp_pipeline.sh --only s34   # substring
#   ./scripts/bridges/baselines/flat_nsp_pipeline.sh --rollout-only
#   ./scripts/bridges/baselines/flat_nsp_pipeline.sh --dry-run

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
VQVAE_DIR="${OCEAN}/experiments/vqvae/small-flat-sc1024"
TOKENS_BASE="${OCEAN}/experiments/tokens"
AR_BASE="${OCEAN}/experiments/ar"
ROLLOUT_BASE="${OCEAN}/experiments/rollouts"
ANALYSIS_BASE="${OCEAN}/experiments/analysis"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

TRAIN_TOKENS="${TOKENS_BASE}/small-flat-sc1024.npz"
VAL_TOKENS="${TOKENS_BASE}/small-flat-sc1024-val.npz"

# ---------- NSP training (canonical recipe: sweep_nsp_sc917.sh) ----------
BATCH_SIZE=32
EPOCHS=400
SUBSTITUTION_RATE=0.1
SEED=42
# Arch grid: name:n_layer:n_embd:n_head — s13/s34-class, matching the sc917
# comparator arms (param count will differ slightly via the smaller vocab).
ARCHS=(
    "s13:3:384:6"
    "s34:5:576:9"
)

# ---------- Rollout fan ----------
TEMPS="0.8 1.2 1.6 2.0"
N_STEPS=2000
N_TRAJ=16
START_FRAME=0
ANALYSIS_BATCH=64
WANDB_GROUP="flat-nsp"

# ---------- Parse args ----------
DRY_RUN=false
ONLY=""
ROLLOUT_ONLY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --rollout-only) ROLLOUT_ONLY=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--only <substr>] [--rollout-only] [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

LOG_DIR="${AR_BASE}/logs"
[ "${DRY_RUN}" = false ] && mkdir -p "${LOG_DIR}" "${TOKENS_BASE}" "${WANDB_BASE}"

echo "=========================================="
echo "Flat-NSP pipeline (small-flat-sc1024)"
echo "  archs: ${ARCHS[*]}   sub=${SUBSTITUTION_RATE} batch=${BATCH_SIZE} epochs=${EPOCHS}"
echo "  rollout fan: T in {${TEMPS}}  ${N_TRAJ} traj x ${N_STEPS} steps"
echo "  Dry run: ${DRY_RUN}   Rollout-only: ${ROLLOUT_ONLY}"
echo "=========================================="

N_SUBMITTED=0

# ---------- Job 1: tokenize (train + val, one job) ----------
TOK_JOB_ID=""
if [ -f "${TRAIN_TOKENS}" ] && [ -f "${VAL_TOKENS}" ]; then
    echo "Skip tokenize (both token files exist)"
elif [ "${ROLLOUT_ONLY}" = true ]; then
    echo "Skip tokenize (rollout-only)"
else
    TMPFILE="$(mktemp /tmp/flatnsp_tok_XXXXXX.sbatch)"
    cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J tok-flat-sc1024
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 2:00:00
#SBATCH -o ${LOG_DIR}/tok-flat-sc1024-%j.out
#SBATCH -e ${LOG_DIR}/tok-flat-sc1024-%j.err

set -euo pipefail
cd "${REPODIR}"
source "${OCEAN}/.venvs/gust/bin/activate"
module load cuda/12.6.1
export PYTHONUNBUFFERED=1
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

if [ ! -f "${TRAIN_TOKENS}" ]; then
    python tokenizer.py save \\
        --checkpoint_dir "${VQVAE_DIR}" \\
        --data_path "${DATA_PATH}" \\
        --output "${TRAIN_TOKENS}" \\
        --batch_size 128
fi
if [ ! -f "${VAL_TOKENS}" ]; then
    python tokenizer.py save \\
        --checkpoint_dir "${VQVAE_DIR}" \\
        --data_path "${DATA_PATH}" \\
        --output "${VAL_TOKENS}" \\
        --sample_start 20000 --sample_stop 22000 \\
        --fit_from "${TRAIN_TOKENS}" \\
        --batch_size 128
fi
echo "Done."
SBATCH_EOF
    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] tokenize small-flat-sc1024 (train + val)"
    else
        TOK_JOB_ID=$(sbatch --parsable "${TMPFILE}")
        echo "tokenize -> job ${TOK_JOB_ID}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
    rm -f "${TMPFILE}"
fi

# ---------- Per arch: training + rollout fan ----------
for arch in "${ARCHS[@]}"; do
    IFS=: read -r LABEL N_LAYER N_EMBD N_HEAD <<< "${arch}"
    RUN_NAME="small-flat-sc1024-nsp-${LABEL}"
    if [ -n "${ONLY}" ] && [[ "${RUN_NAME}" != *"${ONLY}"* ]]; then continue; fi

    CKPT_DIR="${AR_BASE}/${RUN_NAME}"

    # Training completeness / resume detection
    STATE_FILE="${CKPT_DIR}/training_state.json"
    RESUME_FLAG=""
    TRAIN_DONE=false
    if [ -f "${STATE_FILE}" ]; then
        RESUME_FLAG="--resume"
        DONE_EPOCH=$(grep -o '"epoch": *[0-9]*' "${STATE_FILE}" | grep -o '[0-9]*' || echo 0)
        if [ "${DONE_EPOCH:-0}" -ge "${EPOCHS}" ]; then TRAIN_DONE=true; fi
    fi

    TRAIN_JOB_ID=""
    if [ "${ROLLOUT_ONLY}" = false ] && [ "${TRAIN_DONE}" = false ]; then
        DEP_FLAG=""
        [ -n "${TOK_JOB_ID}" ] && DEP_FLAG="--dependency=afterok:${TOK_JOB_ID}"
        TMPFILE="$(mktemp /tmp/flatnsp_train_${LABEL}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 2-00:00:00
#SBATCH -o ${LOG_DIR}/${RUN_NAME}-%j.out
#SBATCH -e ${LOG_DIR}/${RUN_NAME}-%j.err

set -euo pipefail
cd "${REPODIR}"
source "${OCEAN}/.venvs/gust/bin/activate"
module load cuda/12.6.1
export PYTHONUNBUFFERED=1
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

python train_nsp.py \\
    --tokens_path "${TRAIN_TOKENS}" \\
    --train_tokens_path "${TRAIN_TOKENS}" \\
    --n_layer ${N_LAYER} \\
    --n_embd ${N_EMBD} \\
    --n_head ${N_HEAD} \\
    --batch_size ${BATCH_SIZE} \\
    --epochs ${EPOCHS} \\
    --substitution_rate ${SUBSTITUTION_RATE} \\
    --seed ${SEED} \\
    --checkpoint_dir "${CKPT_DIR}" \\
    --wandb_project gust2-nsp \\
    --wandb_name ${RUN_NAME} \\
    --wandb_id flatnsp-${LABEL} \\
    --wandb_group ${WANDB_GROUP} \\
    --wandb_dir "${WANDB_BASE}" \\
    ${RESUME_FLAG}

echo "Finished: \$(date)"
SBATCH_EOF
        if [ "${DRY_RUN}" = true ]; then
            echo "[dry-run] train ${RUN_NAME} (${N_LAYER}L d${N_EMBD} h${N_HEAD}, dep: ${TOK_JOB_ID:-none})"
        else
            TRAIN_JOB_ID=$(sbatch --parsable ${DEP_FLAG} "${TMPFILE}")
            echo "train ${RUN_NAME} -> job ${TRAIN_JOB_ID}"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        fi
        rm -f "${TMPFILE}"
    else
        echo "Skip train ${RUN_NAME} (done=${TRAIN_DONE}, rollout-only=${ROLLOUT_ONLY})"
    fi

    # ---------- Rollout + analysis fan ----------
    for TEMP in ${TEMPS}; do
        TP="${TEMP/./p}"
        ROUT="${ROLLOUT_BASE}/${RUN_NAME}-T${TP}"
        AOUT="${ANALYSIS_BASE}/${RUN_NAME}-T${TP}"
        if [ -f "${AOUT}/metrics.json" ]; then
            echo "Skip rollout ${RUN_NAME}-T${TP} (metrics.json exists)"
            continue
        fi
        DEP_FLAG=""
        [ -n "${TRAIN_JOB_ID}" ] && DEP_FLAG="--dependency=afterok:${TRAIN_JOB_ID}"

        TMPFILE="$(mktemp /tmp/flatnsp_ra_${LABEL}_T${TP}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_NAME}-T${TP}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 2:00:00
#SBATCH -o ${LOG_DIR}/${RUN_NAME}-T${TP}-%j.out
#SBATCH -e ${LOG_DIR}/${RUN_NAME}-T${TP}-%j.err

set -euo pipefail
cd "${REPODIR}"
source "${OCEAN}/.venvs/gust/bin/activate"
module load cuda/12.6.1
export PYTHONUNBUFFERED=1
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

if [ ! -f "${ROUT}/rollout_tokens.npz" ]; then
    python rollout_nsp.py \\
        --checkpoint_dir "${CKPT_DIR}" \\
        --tokens_path "${VAL_TOKENS}" \\
        --train_tokens_path "${TRAIN_TOKENS}" \\
        --start_frame ${START_FRAME} \\
        --n_steps ${N_STEPS} \\
        --n_trajectories ${N_TRAJ} \\
        --seed ${SEED} \\
        --temperature ${TEMP} \\
        --output_dir "${ROUT}"
fi

python analyze_rollout.py \\
    --rollout_dir "${ROUT}" \\
    --vqvae_dir "${VQVAE_DIR}" \\
    --data_path "${DATA_PATH}" \\
    --output_dir "${AOUT}" \\
    --batch_size ${ANALYSIS_BATCH} \\
    --seed ${SEED} \\
    --wandb_project gust2-analysis \\
    --wandb_name "${RUN_NAME}-T${TP}" \\
    --wandb_group ${WANDB_GROUP} \\
    --wandb_dir "${WANDB_BASE}"

echo "Finished: \$(date)"
SBATCH_EOF
        if [ "${DRY_RUN}" = true ]; then
            echo "[dry-run] rollout+analysis ${RUN_NAME}-T${TP} (dep: ${TRAIN_JOB_ID:-none})"
        else
            RA_JOB_ID=$(sbatch --parsable ${DEP_FLAG} "${TMPFILE}")
            echo "rollout+analysis ${RUN_NAME}-T${TP} -> job ${RA_JOB_ID}"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        fi
        rm -f "${TMPFILE}"
    done
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
