#!/bin/bash
# Calibration-temperature GATE — the decisive CHEAP test before any rollout
# compute on the per-scale calibration sampler (paper section B, Recipe 1(ii)).
#
# WHAT IT TESTS. The `etmodel` arm anchored a sampler to the model's PREDICTIVE
# entropy H (the model's own, overconfident, ~1-nat confidence) -> T_k~=1
# everywhere -> nails cold sc341 but collapses the warm configs. The new anchor
# is the per-scale CROSS-ENTROPY / validation NLL  CE = H + KL(p_data||p_model):
# strictly larger by the calibration gap, which is data-measured to GROW with
# token count. Solving H(softmax(z/T_k)) = CE_k gives a per-scale temperature
# schedule that *should* land cold on sc341 and progressively warmer on
# sc917/sc1941 with NO per-config tuning. This job MEASURES that schedule.
#
# This is NOT a rollout. Each cell is one teacher-forced pass over 256 (t0,t1)
# val pairs (minutes), reusing eval_single_step.make_predict_and_loss with a
# temperature grid; measure_calibration_temp.py inverts the entropy-vs-T curve
# on the REAL logits to get T_k. One Slurm job loops all selected cells, then
# runs analyze_calibration_gate.py to print the GO/NO-GO verdict in the log.
#
# GO/NO-GO (analyze_calibration_gate.py): GO to the rollout sweep only if the
# solved T_eff is COLD on sc341 (~0.9-1.2), WARM on sc917 (~1.5-2.0), WARMEST on
# sc1941 (~1.6-2.2), monotone, and sc1941 > sc917. NO-GO (report negative
# result, save the rollout compute) if sc341 over-warms (>1.3), sc1941
# under-warms (<1.4), or anything explodes (>2.5 -> over-diffusive).
#
# Usage:
#   ./scripts/bridges/sweep_calibration_gate.sh                 # small+medium x 3 configs (flagship arch)
#   ./scripts/bridges/sweep_calibration_gate.sh --size small
#   ./scripts/bridges/sweep_calibration_gate.sh --all-arches    # every arch (still cheap)
#   ./scripts/bridges/sweep_calibration_gate.sh --dry-run
#   ./scripts/bridges/sweep_calibration_gate.sh --list

set -euo pipefail

# ---------- Paths (match sweep_inference_samplers.sh) ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
TOKENS_BASE="${OCEAN}/experiments/tokens"
AR_BASE="${OCEAN}/experiments/ar-robust-scaling"
GATE_BASE="${OCEAN}/experiments/calibration-gate"
ACCOUNT="mth260004p"

MAX_PAIRS=256

# ---------- Grid (matches sweep_inference_samplers.sh) ----------
SIZES_DEFAULT=(small medium)        # the relevant VQ tiers (large adds no recon power)
TASKS=(
    "sc341:s06"  "sc341:s09"  "sc341:s13"  "sc341:s18"  "sc341:s24"
    "sc917:s13"  "sc917:s22"  "sc917:s34"  "sc917:s50"  "sc917:s74"
    "sc1941:s31" "sc1941:s48" "sc1941:s73" "sc1941:s113" "sc1941:s139"
)
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
ALL_ARCHES=false
LIST_ONLY=false
FILTER_SIZE=""
FILTER_VQVAE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --all-arches) ALL_ARCHES=true; shift ;;
        --size)       FILTER_SIZE="$2"; shift 2 ;;
        --vqvae)      FILTER_VQVAE="$2"; shift 2 ;;
        --list)       LIST_ONLY=true; shift ;;
        --help|-h)
            cat <<EOF
Usage: $0 [--size <s>] [--vqvae <substr>] [--all-arches] [--dry-run] [--list]
  --size <s>        Only this VQ size (small|medium|large). Default: small medium.
  --vqvae <substr>  Filter by sc-config (e.g. sc1941).
  --all-arches      Every NSP arch (default: flagship arch per config).
  --dry-run         Print the cell list without submitting.
  --list            Print the plan and exit.
EOF
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SIZES=("${SIZES_DEFAULT[@]}")
if [ -n "${FILTER_SIZE}" ]; then SIZES=("${FILTER_SIZE}"); fi

# ---------- Build the cell list ----------
CELLS=()
for SIZE in "${SIZES[@]}"; do
    for spec in "${TASKS[@]}"; do
        IFS=':' read -r SC LABEL <<< "${spec}"
        if [ -n "${FILTER_VQVAE}" ] && [[ "${SC}" != *"${FILTER_VQVAE}"* ]]; then continue; fi
        if [ "${ALL_ARCHES}" = false ] && [ "${LABEL}" != "$(flagship_label_for ${SC})" ]; then continue; fi
        CELLS+=("${SIZE}:${SC}:${LABEL}")
    done
done

if [ "${LIST_ONLY}" = true ]; then
    echo "Calibration-temperature gate — teacher-forced only (NO rollout):"
    echo "  Sizes:   ${SIZES[*]}"
    echo "  Cells:   ${#CELLS[@]}  (flagship arch unless --all-arches)"
    for c in "${CELLS[@]}"; do echo "    ${c}"; done
    echo "  Output:  ${GATE_BASE}/<size>-<sc>-<arch>-calib.json"
    echo "  Verdict: analyze_calibration_gate.py (printed in the job log)"
    exit 0
fi

if [ "${#CELLS[@]}" -eq 0 ]; then
    echo "No cells selected." >&2; exit 1
fi

echo "=========================================="
echo "Calibration-temperature GATE (teacher-forced, ${MAX_PAIRS} pairs/cell, NO rollout)"
echo "  Sizes: ${SIZES[*]}    cells: ${#CELLS[@]}"
echo "  Output: ${GATE_BASE}"
echo "  Dry run: ${DRY_RUN}"
echo "=========================================="

# Emit one job that loops all cells then prints the verdict.
LOG_DIR="${GATE_BASE}/logs"
CELL_ARGS=""
for c in "${CELLS[@]}"; do CELL_ARGS+=" ${c}"; done

TMPFILE="$(mktemp /tmp/calib_gate_XXXXXX.sbatch)"
cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J calib-gate
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t 1:00:00
#SBATCH -o ${LOG_DIR}/calib-gate-%j.out
#SBATCH -e ${LOG_DIR}/calib-gate-%j.err

set -euo pipefail
cd "${REPODIR}"
source "${OCEAN}/.venvs/gust/bin/activate"
module load cuda/12.6.1
export PYTHONUNBUFFERED=1
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

mkdir -p "${GATE_BASE}" "${LOG_DIR}"
echo "Job \${SLURM_JOB_ID} on \$(hostname)  started \$(date)"

for cell in ${CELL_ARGS}; do
    IFS=':' read -r SIZE SC LABEL <<< "\${cell}"
    RUN_NAME="\${SIZE}-\${SC}-nsp-\${LABEL}"
    CKPT="${AR_BASE}/\${RUN_NAME}"
    VAL_TOKENS="${TOKENS_BASE}/\${SIZE}-\${SC}-val.npz"
    OUT="${GATE_BASE}/\${SIZE}-\${SC}-\${LABEL}-calib.json"

    if [ ! -f "\${CKPT}/training_state.json" ]; then echo "[skip] \${RUN_NAME}: no checkpoint"; continue; fi
    if [ ! -f "\${VAL_TOKENS}" ]; then echo "[skip] \${RUN_NAME}: no val tokens"; continue; fi
    if [ -f "\${OUT}" ]; then echo "[reuse] \${OUT}"; continue; fi

    echo "==== \${RUN_NAME} ===="
    python measure_calibration_temp.py \\
        --checkpoint_dir "\${CKPT}" \\
        --tokens_path "\${VAL_TOKENS}" \\
        --output "\${OUT}" \\
        --max_pairs ${MAX_PAIRS}
done

echo ""
echo "================ GATE VERDICT ================"
python analyze_calibration_gate.py --gate_dir "${GATE_BASE}" || true
echo "Finished \$(date)"
SBATCH_EOF

if [ "${DRY_RUN}" = true ]; then
    echo "[dry-run] would submit one job over ${#CELLS[@]} cells:"
    for c in "${CELLS[@]}"; do echo "    ${c}"; done
    echo "[dry-run] sbatch script at ${TMPFILE} (kept for inspection)"
else
    JOBID=$(sbatch --parsable "${TMPFILE}")
    echo "Submitted calibration gate -> ${JOBID}"
    echo "Verdict will be in ${LOG_DIR}/calib-gate-${JOBID}.out"
    rm -f "${TMPFILE}"
fi
