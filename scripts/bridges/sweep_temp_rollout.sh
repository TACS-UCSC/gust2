#!/bin/bash
# Temperature sweep (rollouts) for the robust-scaling anchor models.
#
# Diffusive-collapse diagnostic: long autoregressive rollouts relax to an
# over-diffuse steady state. We've ruled out single-step modeling (sc341/
# sc917/sc1941 all sit at the tokenizer's single-step ceiling), so this
# probes the rollout/sampling dynamics by sweeping the global sampling
# temperature and asking whether more dispersion refills the high-k tail.
#
# Repurposed from sweep_sampling_rollout.sh, but targets the robust-scaling
# layout (experiments/ar-robust-scaling, run names <vq>-nsp-<label>) and a
# fixed two-model list instead of the old <vq>-nsp-<size> grid. Writes to
# experiments/rollouts-temp-sweep/ to stay isolated. No wandb here — the
# companion sweep_temp_analysis.sh decodes these rollouts and logs spectra.
#
# Usage:
#   ./scripts/bridges/sweep_temp_rollout.sh              Submit the full grid
#   ./scripts/bridges/sweep_temp_rollout.sh --dry-run
#   ./scripts/bridges/sweep_temp_rollout.sh --vqvae sc917   Only sc917 model

set -euo pipefail

OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
TOKENS_BASE="${OCEAN}/experiments/tokens"
AR_BASE="${OCEAN}/experiments/ar-robust-scaling"
ROLLOUT_BASE="${OCEAN}/experiments/rollouts-temp-sweep"
ACCOUNT="mth260004p"

N_STEPS=2000
START_FRAME=0

TEMPERATURES=(0.8 1.0 1.2 1.4 1.6)
SEEDS=(0)

# "<vqvae_name>:<full_run_name>" — robust-scaling anchors.
MODELS=(
    "large-sc917:large-sc917-nsp-s34"
    "large-sc1941:large-sc1941-nsp-s73"
)

# ---------- Parse args ----------
DRY_RUN=false
FILTER_VQVAE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --vqvae) FILTER_VQVAE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--vqvae <substr>] [--dry-run]"
            echo "  Models:       ${MODELS[*]}"
            echo "  Temperatures: ${TEMPERATURES[*]}"
            echo "  Seeds:        ${SEEDS[*]}"
            exit 0
            ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

if [ -n "${FILTER_VQVAE}" ]; then
    FILTERED=()
    for m in "${MODELS[@]}"; do
        [[ "${m}" == *"${FILTER_VQVAE}"* ]] && FILTERED+=("${m}")
    done
    MODELS=("${FILTERED[@]}")
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No models match the given filter."
    exit 1
fi

echo "=========================================="
echo "NSP Temperature Sweep — Rollouts (${N_STEPS} steps)"
echo "  Models:       ${#MODELS[@]}"
echo "  Temperatures: ${TEMPERATURES[*]}"
echo "  Seeds:        ${SEEDS[*]}"
echo "  AR base:      ${AR_BASE}"
echo "  Output:       ${ROLLOUT_BASE}"
echo "  Dry run:      ${DRY_RUN}"
echo "=========================================="
echo ""

N_SUBMITTED=0

for entry in "${MODELS[@]}"; do
    IFS=':' read -r VQVAE_NAME RUN_NAME <<< "${entry}"
    VAL_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}-val.npz"
    CHECKPOINT_DIR="${AR_BASE}/${RUN_NAME}"

    if [ ! -f "${VAL_TOKENS}" ] && [ "${DRY_RUN}" = false ]; then
        echo "[skip] ${VQVAE_NAME}: val tokens not found"
        continue
    fi
    if [ ! -f "${CHECKPOINT_DIR}/training_state.json" ] && [ "${DRY_RUN}" = false ]; then
        echo "[skip] ${RUN_NAME}: no NSP checkpoint"
        continue
    fi

    for TEMP in "${TEMPERATURES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            RUN_TAG="${RUN_NAME}-T${TEMP}-s${SEED}"
            OUTPUT_DIR="${ROLLOUT_BASE}/${RUN_TAG}"

            if [ -f "${OUTPUT_DIR}/rollout_tokens.npz" ]; then
                echo "[skip] ${RUN_TAG}: rollout already exists"
                continue
            fi

            if [ "${DRY_RUN}" = false ]; then
                mkdir -p "${OUTPUT_DIR}" "${ROLLOUT_BASE}/logs"
            fi

            TMPFILE="$(mktemp /tmp/temp_roll_${RUN_TAG//\//_}_XXXXXX.sbatch)"
            cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J temp-${RUN_TAG}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t 12:00:00
#SBATCH -o ${ROLLOUT_BASE}/logs/${RUN_TAG}-%j.out
#SBATCH -e ${ROLLOUT_BASE}/logs/${RUN_TAG}-%j.err

cd "${REPODIR}"
source /ocean/projects/mth260004p/sambamur/.venvs/gust/bin/activate
module load cuda/12.6.1
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\$LD_LIBRARY_PATH

echo "=========================================="
echo "Job:       \${SLURM_JOB_ID}"
echo "Node:      \$(hostname)"
echo "Started:   \$(date)"
echo "Run:       ${RUN_TAG}"
echo "Tokens:    ${VAL_TOKENS}"
echo "Ckpt:      ${CHECKPOINT_DIR}"
echo "Output:    ${OUTPUT_DIR}"
echo "Steps:     ${N_STEPS}"
echo "Temp:      ${TEMP}"
echo "Seed:      ${SEED}"
echo "=========================================="

python rollout_nsp.py \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --tokens_path "${VAL_TOKENS}" \\
    --start_frame ${START_FRAME} \\
    --n_steps ${N_STEPS} \\
    --temperature ${TEMP} \\
    --seed ${SEED} \\
    --output_dir "${OUTPUT_DIR}"

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
SBATCH_EOF

            if [ "${DRY_RUN}" = true ]; then
                echo "[dry-run] ${RUN_TAG}"
            else
                echo "Submitting ${RUN_TAG}..."
                sbatch "${TMPFILE}"
            fi
            rm -f "${TMPFILE}"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        done
    done
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
