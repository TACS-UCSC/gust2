#!/bin/bash
# Inference-sampler sweep: parameter-free, data-anchored samplers vs the
# swept-temperature yardstick. The PAYOFF experiment for paper section B
# (I1/I2/I3): does a self-calibrating sampler land in the non-collapsed range
# OUT OF THE BOX, within ballpark of the per-config temperature optimum, with
# NO per-config tuning?
#
# This is the sampler analogue of sweep_scaling_tempopt_n128.sh, but the
# 5-temperature axis is REPLACED by a sampler-ARM axis (one run per arm — no
# per-config sweep, that's the whole point). The swept-T optima are already in
# hand (plots/scaling_tempopt_n128/scaling_tempopt.csv) and are NOT rerun; the
# comparison is done offline by plot_inference_samplers.py.
#
# Arms (run = <cell>-<arm>):
#   etpool   entropy_target, anchor=pooled-marginal     [HEADLINE homeostat]
#   etpos    entropy_target, anchor=mean-per-position   [anchor-stat ablation]
#   tophabs  top_h absolute (per-position anchor)        [truncation-only control:
#                                                         expected to stay stuck
#                                                         in mean-field collapse]
#   typical  locally typical (tau=0.95)                  [I2 cheap baseline]
#   minp     min_p=0.05                                  [I3 negative control]
#   invedt   inverted_edt strength=0.5                   [I3 negative control]
#
# Per (cell, arm): build the per-scale entropy anchor from the cell's TRAIN
# tokens (cheap, pure-numpy; built inside the arm dir so concurrent arms never
# race), then rollout_nsp.py (N=128, posmask ON, --ic_chunk) -> analyze_rollout.py
# (spectra, pooled EMD, collapse_rate). Honest metrics land in wandb
# gust2-inference-samplers-{small,medium,large}, group=<sc>, run=<cell>-<arm>.
#
# Success (ranked by EMD / PDF / OOD — NEVER TKE RSE, which the over-diffusive
# mode games): the homeostat arms are non-collapsed (EMD < 2x VQ-VAE floor) and
# within ballpark of the cell's best-T EMD; the entropy-shrinking controls
# (minp, invedt) and truncation-only tophabs collapse.
#
# Usage:
#   ./scripts/bridges/sweep_inference_samplers.sh                 # full grid x 6 arms
#   ./scripts/bridges/sweep_inference_samplers.sh --flagship      # 1 arch per (size,sc)
#   ./scripts/bridges/sweep_inference_samplers.sh --size small --vqvae sc917
#   ./scripts/bridges/sweep_inference_samplers.sh --arm etpool,etpos
#   ./scripts/bridges/sweep_inference_samplers.sh --dry-run
#   ./scripts/bridges/sweep_inference_samplers.sh --list

set -euo pipefail

# ---------- Paths (match sweep_scaling_tempopt_n128.sh) ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
TOKENS_BASE="${OCEAN}/experiments/tokens"
VQVAE_BASE="${OCEAN}/experiments/vqvae"
AR_BASE="${OCEAN}/experiments/ar-robust-scaling"
SAMPLERS_BASE="${OCEAN}/experiments/inference-samplers"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"
WANDB_PROJECT_PREFIX="gust2-inference-samplers"

# ---------- Rollout / analysis config ----------
N_STEPS=2000
START_FRAME=0
N_TRAJ=128
SEED=42
BATCH_SIZE=64

ic_chunk_for() {
    case "$1" in
        sc341)  echo "0"  ;;
        sc917)  echo "64" ;;
        sc1941) echo "32" ;;
        *)      echo "32" ;;
    esac
}

# Each arm = one 2000-step rollout + analyze (same cost as one tempopt temp job).
walltime_for() {
    case "$1" in
        sc341)  echo "1:00:00" ;;
        sc917)  echo "1:30:00" ;;
        sc1941) echo "6:00:00" ;;
        *)      echo "1:30:00" ;;
    esac
}

# ---------- Sampler arms: "label|needs_anchor|flags" ----------
# Non-ancestral samplers ignore --temperature, but we pass --temperature 1.0
# everywhere to respect the project's "never invoke rollout greedy" convention.
ARMS=(
    "etpool|1|--sampler entropy_target --anchor_stat pooled"
    "etpos|1|--sampler entropy_target --anchor_stat per_position"
    "tophabs|1|--sampler top_h --top_h_mode absolute --anchor_stat per_position"
    "typical|0|--sampler typical --typical_tau 0.95"
    "minp|0|--sampler min_p --min_p 0.05"
    "invedt|0|--sampler inverted_edt --inverted_edt_strength 0.5"
)

# ---------- Sweep grid (matches sweep_scaling_tempopt_n128.sh) ----------
SIZES_ALL=(small medium large)
TASKS=(
    "sc341:s06:2:256:4"
    "sc341:s09:1:384:6"
    "sc341:s13:3:384:6"
    "sc341:s18:6:384:6"
    "sc341:s24:4:512:8"

    "sc917:s13:3:384:6"
    "sc917:s22:8:384:6"
    "sc917:s34:5:576:9"
    "sc917:s50:9:576:9"
    "sc917:s74:3:1024:16"

    "sc1941:s31:4:576:9"
    "sc1941:s48:6:640:10"
    "sc1941:s73:7:768:12"
    "sc1941:s113:6:1024:16"
    "sc1941:s139:8:1024:8"
)

# Flagship arch per sc (largest params) for the --flagship quick pass.
flagship_label_for() {
    case "$1" in
        sc341)  echo "s24"  ;;
        sc917)  echo "s74"  ;;
        sc1941) echo "s139" ;;
        *)      echo "" ;;
    esac
}

# ---------- Parse args ----------
DRY_RUN=false
FILTER_SIZE=""
FILTER_VQVAE=""
FILTER_LABEL=""
FILTER_ARM=""
FLAGSHIP=false
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --size) FILTER_SIZE="$2"; shift 2 ;;
        --vqvae) FILTER_VQVAE="$2"; shift 2 ;;
        --label) FILTER_LABEL="$2"; shift 2 ;;
        --arm) FILTER_ARM="$2"; shift 2 ;;
        --flagship) FLAGSHIP=true; shift ;;
        --list) LIST_ONLY=true; shift ;;
        --help|-h)
            cat <<EOF
Usage: $0 [--size <s>] [--vqvae <substr>] [--label <substr>] [--arm <csv>] [--flagship] [--dry-run] [--list]
  --size <s>         Only this VQ size (small|medium|large). Default: all three.
  --vqvae <substr>   Filter by sc-config substring (e.g. sc1941).
  --label <substr>   Filter by NSP arch label (e.g. s73).
  --arm <csv>        Only these arms (e.g. etpool,etpos). Default: all 6.
  --flagship         Only the flagship arch per (size, sc) — sc341:s24, sc917:s74, sc1941:s139.
  --dry-run          Print actions without submitting.
  --list             Print the grid + arms and exit.
EOF
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SIZES=("${SIZES_ALL[@]}")
if [ -n "${FILTER_SIZE}" ]; then SIZES=("${FILTER_SIZE}"); fi

arm_selected() {  # $1 = arm label
    [ -z "${FILTER_ARM}" ] && return 0
    IFS=',' read -ra want <<< "${FILTER_ARM}"
    for a in "${want[@]}"; do [ "$a" = "$1" ] && return 0; done
    return 1
}

if [ "${LIST_ONLY}" = true ]; then
    echo "Inference-sampler sweep — N=${N_TRAJ}, posmask ON, sampler-arm axis:"
    echo "  Arms:"
    for armspec in "${ARMS[@]}"; do
        IFS='|' read -r ARM NEEDS FLAGS <<< "${armspec}"
        printf "    %-9s needs_anchor=%s  %s\n" "${ARM}" "${NEEDS}" "${FLAGS}"
    done
    echo ""
    echo "  Brackets per sc:"
    for sc in sc341 sc917 sc1941; do
        printf "    %-7s ic_chunk=%s  walltime=%s  flagship=%s\n" \
            "${sc}" "$(ic_chunk_for ${sc})" "$(walltime_for ${sc})" "$(flagship_label_for ${sc})"
    done
    echo ""
    n_cells=0
    for SIZE in "${SIZES[@]}"; do for spec in "${TASKS[@]}"; do n_cells=$((n_cells+1)); done; done
    n_arms=0; for armspec in "${ARMS[@]}"; do IFS='|' read -r ARM _ _ <<< "${armspec}"; arm_selected "${ARM}" && n_arms=$((n_arms+1)); done
    echo "  Cells:      ${n_cells}  (sizes: ${SIZES[*]})  x arms: ${n_arms}"
    echo "  Wandb:      ${WANDB_PROJECT_PREFIX}-{small,medium,large}, group=<sc>, run=<cell>-<arm>"
    echo "  Output base:${SAMPLERS_BASE}"
    exit 0
fi

echo "=========================================="
echo "Inference-sampler sweep (N=${N_TRAJ})"
echo "  Sizes:       ${SIZES[*]}"
echo "  posmask:     ON   N_traj: ${N_TRAJ}   steps: ${N_STEPS}"
echo "  Axis:        sampler arms (no per-config temperature sweep)"
echo "  Flagship:    ${FLAGSHIP}   arm filter: ${FILTER_ARM:-<all>}"
echo "  Output base: ${SAMPLERS_BASE}"
echo "  Wandb:       ${WANDB_PROJECT_PREFIX}-{small,medium,large}"
echo "  Dry run:     ${DRY_RUN}"
echo "=========================================="

N_SUBMITTED=0

for SIZE in "${SIZES[@]}"; do
    case "${SIZE}" in
        small|medium|large) ;;
        *) echo "Invalid size '${SIZE}'" >&2; exit 1 ;;
    esac

    for spec in "${TASKS[@]}"; do
        IFS=':' read -r SC LABEL N_LAYER N_EMBD N_HEAD <<< "${spec}"

        if [ -n "${FILTER_VQVAE}" ] && [[ "${SC}" != *"${FILTER_VQVAE}"* ]]; then continue; fi
        if [ -n "${FILTER_LABEL}" ] && [[ "${LABEL}" != *"${FILTER_LABEL}"* ]]; then continue; fi
        if [ "${FLAGSHIP}" = true ] && [ "${LABEL}" != "$(flagship_label_for ${SC})" ]; then continue; fi

        VQVAE_NAME="${SIZE}-${SC}"
        RUN_NAME="${VQVAE_NAME}-nsp-${LABEL}"
        CHECKPOINT_DIR="${AR_BASE}/${RUN_NAME}"
        VAL_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}-val.npz"
        TRAIN_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}.npz"
        VQVAE_DIR="${VQVAE_BASE}/${VQVAE_NAME}"
        CELL_DIR="${SAMPLERS_BASE}/${RUN_NAME}"
        LOG_DIR="${SAMPLERS_BASE}/logs"
        WANDB_PROJECT="${WANDB_PROJECT_PREFIX}-${SIZE}"
        WANDB_GROUP="${SC}"
        WALLTIME="$(walltime_for ${SC})"
        IC_CHUNK="$(ic_chunk_for ${SC})"

        if [ "${DRY_RUN}" = false ]; then
            if [ ! -f "${CHECKPOINT_DIR}/training_state.json" ]; then
                echo "[skip] ${RUN_NAME}: no NSP checkpoint"; continue; fi
            if [ ! -f "${VAL_TOKENS}" ] || [ ! -f "${TRAIN_TOKENS}" ]; then
                echo "[skip] ${RUN_NAME}: missing val or train tokens"; continue; fi
            if [ ! -f "${VQVAE_DIR}/training_state.json" ]; then
                echo "[skip] ${RUN_NAME}: no VQ-VAE checkpoint"; continue; fi
            mkdir -p "${CELL_DIR}" "${LOG_DIR}" "${WANDB_BASE}"
        fi

        for armspec in "${ARMS[@]}"; do
            IFS='|' read -r ARM NEEDS_ANCHOR ARM_FLAGS <<< "${armspec}"
            arm_selected "${ARM}" || continue

            ARM_DIR="${CELL_DIR}/${ARM}"
            ROUT="${ARM_DIR}/rollout"
            AOUT="${ARM_DIR}/analysis"
            ANCHOR="${ARM_DIR}/anchor.json"

            ANCHOR_FLAG=""
            if [ "${NEEDS_ANCHOR}" = "1" ]; then
                ANCHOR_FLAG="--entropy_anchor_path ${ANCHOR}"
            fi

            TMPFILE="$(mktemp /tmp/samp_${RUN_NAME}_${ARM}_XXXXXX.sbatch)"
            cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J samp-${RUN_NAME}-${ARM}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t ${WALLTIME}
#SBATCH -o ${LOG_DIR}/${RUN_NAME}-${ARM}-%j.out
#SBATCH -e ${LOG_DIR}/${RUN_NAME}-${ARM}-%j.err

set -euo pipefail

cd "${REPODIR}"
source "${OCEAN}/.venvs/gust/bin/activate"
module load cuda/12.6.1
export PYTHONUNBUFFERED=1

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:    \${SLURM_JOB_ID}   Node: \$(hostname)   Started: \$(date)"
echo "Cell:   ${RUN_NAME}   Arm: ${ARM}   (N=${N_TRAJ}, ic_chunk=${IC_CHUNK}, posmask ON)"
echo "Flags:  ${ARM_FLAGS} ${ANCHOR_FLAG}"
echo "Wandb:  ${WANDB_PROJECT} / group=${WANDB_GROUP} / run=${RUN_NAME}-${ARM}"
echo "=========================================="

mkdir -p "${ARM_DIR}"

if [ -f "${AOUT}/metrics.json" ]; then
    echo "[skip] ${ARM}: analysis already complete"
else
    # Build the per-scale entropy anchor from the cell's TRAIN tokens (same
    # tokenizer as the checkpoint). Idempotent; arm-local so no cross-arm race.
    if [ "${NEEDS_ANCHOR}" = "1" ] && [ ! -f "${ANCHOR}" ]; then
        echo "---- building anchor (${ANCHOR}) ----"
        python measure_tokenizer_entropy.py \\
            --tokens_path "${TRAIN_TOKENS}" \\
            --output "${ANCHOR}"
    fi

    if [ -f "${ROUT}/rollout_tokens.npz" ]; then
        echo "[reuse] ${ARM}: rollout exists, running analysis only"
    else
        python rollout_nsp.py \\
            --checkpoint_dir "${CHECKPOINT_DIR}" \\
            --tokens_path "${VAL_TOKENS}" \\
            --train_tokens_path "${TRAIN_TOKENS}" \\
            --start_frame ${START_FRAME} \\
            --n_steps ${N_STEPS} \\
            --n_trajectories ${N_TRAJ} \\
            --ic_chunk ${IC_CHUNK} \\
            --seed ${SEED} \\
            --temperature 1.0 \\
            ${ARM_FLAGS} ${ANCHOR_FLAG} \\
            --output_dir "${ROUT}"
    fi

    python analyze_rollout.py \\
        --rollout_dir "${ROUT}" \\
        --vqvae_dir "${VQVAE_DIR}" \\
        --data_path "${DATA_PATH}" \\
        --output_dir "${AOUT}" \\
        --batch_size ${BATCH_SIZE} \\
        --seed ${SEED} \\
        --wandb_project ${WANDB_PROJECT} \\
        --wandb_name "${RUN_NAME}-${ARM}" \\
        --wandb_group "${WANDB_GROUP}" \\
        --wandb_dir "${WANDB_BASE}"
fi

echo "Finished: \$(date)"
SBATCH_EOF

            if [ "${DRY_RUN}" = true ]; then
                echo "[dry-run] ${RUN_NAME} arm=${ARM}  N=${N_TRAJ} ic_chunk=${IC_CHUNK} walltime=${WALLTIME}  flags='${ARM_FLAGS} ${ANCHOR_FLAG}'"
            else
                echo "Submitting ${RUN_NAME} arm=${ARM} (N=${N_TRAJ})..."
                JOBID=$(sbatch --parsable "${TMPFILE}")
                echo "  -> ${JOBID}"
                N_SUBMITTED=$((N_SUBMITTED + 1))
            fi
            rm -f "${TMPFILE}"
        done
    done
done

echo "=========================================="
echo "Submitted ${N_SUBMITTED} arm jobs."
echo "When analysis runs have logged, compare vs the swept-T yardstick:"
echo "  ~/llm/bin/python plot_inference_samplers.py"
echo "=========================================="
