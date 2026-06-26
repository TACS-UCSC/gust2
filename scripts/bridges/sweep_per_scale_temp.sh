#!/bin/bash
# Horizon-marginal temperature calibration — per-scale + per-cell + static controls.
#
# Section B's positive-result attempt after THREE single-step negatives (calib
# gate, data_mix, drift_warm). Those all anchored to a SINGLE-STEP teacher-forced
# statistic, which provably solves to T<=1 while the rollout needs T~=2. The fix
# (the user's correction): the single-step conditional is the WRONG target — at
# coarse scales H(X_k|past)~=0 yet the DATA marginal over time is large, and
# mean-field collapse IS the rollout's horizon emitted-token marginal H_roll
# shrinking below the data marginal H_data. So we calibrate a FIXED temperature
# (per-scale, and per-cell) whose rollout marginal MATCHES the data marginal:
#     per-scale arm:  T_k*  s.t. H_roll,k(T_k)   = H_data,k    (pooled marginal)
#     per-cell  arm:  T_ij* s.t. H_roll,ij(T_ij) = H_data,ij   (per-position)
# Curves H_roll(T) are read off the ALREADY-EXISTING global-T rollouts
# (scaling-tempopt-n128) — no new GPU for the curve — and inverted by
# solve_calib_schedule.py. A static coarse-warm vs fine-warm control independently
# localizes where warmth moves EMD even if the calibrated schedule disappoints.
#
# Judge by EMD / PDF / per-position OOD / spectra — NEVER entropy-match (that is
# the knob, not the metric) and NEVER TKE RSE. Yardstick (small-sc1941-s139):
# swept best_T=2.0 EMD 0.196 (= 2.40x VQ floor 0.0816); etmodel ~73x floor.
#
# Arms:
#   calib   one job: build data target (train tokens) -> measure reused-rollout
#           curves -> solve -> per-scale rollout+analyze -> per-cell rollout+analyze.
#   static  one job per warm value: hand-set per-scale anchor (coarse-warm /
#           fine-warm) -> rollout+analyze.
#
# Usage:
#   ./scripts/bridges/sweep_per_scale_temp.sh                      # sc1941 flagship, calib+static
#   ./scripts/bridges/sweep_per_scale_temp.sh --arms calib
#   ./scripts/bridges/sweep_per_scale_temp.sh --arms static --coarse-warms "2.0" --fine-warms "2.0"
#   ./scripts/bridges/sweep_per_scale_temp.sh --size medium --vqvae sc917
#   ./scripts/bridges/sweep_per_scale_temp.sh --supp-temps "1.0 1.2 2.5 3.0"   # extend the curve
#   ./scripts/bridges/sweep_per_scale_temp.sh --dry-run
#   ./scripts/bridges/sweep_per_scale_temp.sh --list

set -euo pipefail

# ---------- Paths (match sweep_drift_warm.sh / sweep_inference_samplers.sh) ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
TOKENS_BASE="${OCEAN}/experiments/tokens"
VQVAE_BASE="${OCEAN}/experiments/vqvae"
AR_BASE="${OCEAN}/experiments/ar-robust-scaling"
TEMPOPT_BASE="${OCEAN}/experiments/scaling-tempopt-n128"   # reused global-T rollouts
PST_BASE="${OCEAN}/experiments/per-scale-temp"             # new output base
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"
WANDB_PROJECT_PREFIX="gust2-inference-samplers"

# ---------- Rollout / analysis config (match the global-T sweep) ----------
N_STEPS=2000
START_FRAME=0
N_TRAJ=128
SEED=42
BATCH_SIZE=64
BIAS_CORRECT="miller_madow"        # MM on both sides; no-op for pooled, corrects per-cell
PER_SCALE_TARGET="pooled_marginal_nats"
COARSE_WARMS="1.5 2.0 2.5"
FINE_WARMS="1.5 2.0 3.0"
N_FINE=2                           # # of finest trainable scales the static controls warm
SUPP_TEMPS=""                      # extra global-T points to fold into the curve (if rolled out)

ic_chunk_for() { case "$1" in sc341) echo 0;; sc917) echo 64;; sc1941) echo 32;; *) echo 32;; esac; }
flagship_label_for() { case "$1" in sc341) echo s24;; sc917) echo s74;; sc1941) echo s139;; *) echo "";; esac; }
reused_temps_for() {
    case "$1" in
        sc341)  echo "0.8 0.9 1.0 1.1 1.2" ;;
        sc917)  echo "1.2 1.4 1.6 1.8 2.0" ;;
        sc1941) echo "1.4 1.6 1.8 2.0 2.2" ;;
        *)      echo "" ;;
    esac
}
# one rollout ~ drift_warm walltime; the calib job runs TWO rollouts back to back.
walltime_calib_for()  { case "$1" in sc341) echo "2:00:00";; sc917) echo "3:00:00";; sc1941) echo "12:00:00";; *) echo "3:00:00";; esac; }
walltime_static_for() { case "$1" in sc341) echo "1:00:00";; sc917) echo "1:30:00";; sc1941) echo "6:00:00";;  *) echo "1:30:00";; esac; }

# ---------- Parse args ----------
DRY_RUN=false; LIST_ONLY=false; SIZE="small"; SC="sc1941"; ARMS="calib static"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --list)         LIST_ONLY=true; shift ;;
        --size)         SIZE="$2"; shift 2 ;;
        --vqvae)        SC="$2"; shift 2 ;;
        --arms)         ARMS="$2"; shift 2 ;;
        --coarse-warms) COARSE_WARMS="$2"; shift 2 ;;
        --fine-warms)   FINE_WARMS="$2"; shift 2 ;;
        --n-fine)       N_FINE="$2"; shift 2 ;;
        --supp-temps)   SUPP_TEMPS="$2"; shift 2 ;;
        --per-scale-target) PER_SCALE_TARGET="$2"; shift 2 ;;
        --bias-correct) BIAS_CORRECT="$2"; shift 2 ;;
        --help|-h)
            cat <<EOF
Usage: $0 [--size <s>] [--vqvae <sc>] [--arms "calib static"] [--coarse-warms "..."] [--fine-warms "..."] [--supp-temps "..."] [--dry-run] [--list]
  --size <s>          VQ size (small|medium|large). Default: small.
  --vqvae <sc>        sc-config (sc341|sc917|sc1941). Default: sc1941.
  --arms "..."        Which arms to submit: calib, static, or both. Default: "calib static".
  --coarse-warms "..."  Static coarse-warm temperatures. Default: ${COARSE_WARMS}.
  --fine-warms "..."    Static fine-warm temperatures. Default: ${FINE_WARMS}.
  --n-fine N          # finest trainable scales the static controls warm. Default: ${N_FINE}.
  --supp-temps "..."  Extra global-T points to fold into the calibration curve
                      (their scaling-tempopt-n128 rollouts must exist). Default: none.
  --per-scale-target  pooled_marginal_nats | mean_per_position_nats. Default: ${PER_SCALE_TARGET}.
  --bias-correct      none | miller_madow. Default: ${BIAS_CORRECT}.
  --dry-run           Print actions without submitting.
  --list              Print the plan and exit.
EOF
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

LABEL="$(flagship_label_for ${SC})"
if [ -z "${LABEL}" ]; then echo "No flagship arch for ${SC}" >&2; exit 1; fi

VQVAE_NAME="${SIZE}-${SC}"
RUN_NAME="${VQVAE_NAME}-nsp-${LABEL}"
CHECKPOINT_DIR="${AR_BASE}/${RUN_NAME}"
VAL_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}-val.npz"
TRAIN_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}.npz"
VQVAE_DIR="${VQVAE_BASE}/${VQVAE_NAME}"
CELL_DIR="${PST_BASE}/${RUN_NAME}"
CALIB_DIR="${CELL_DIR}/_calib"          # shared: data target, curves, solved anchors
LOG_DIR="${PST_BASE}/logs"
WANDB_PROJECT="${WANDB_PROJECT_PREFIX}-${SIZE}"
WANDB_GROUP="${SC}"
IC_CHUNK="$(ic_chunk_for ${SC})"
REUSED_TEMPS="$(reused_temps_for ${SC}) ${SUPP_TEMPS}"
TEMPOPT_RUN="${TEMPOPT_BASE}/${RUN_NAME}"
WT_CALIB="$(walltime_calib_for ${SC})"
WT_STATIC="$(walltime_static_for ${SC})"

if [ "${LIST_ONLY}" = true ]; then
    echo "horizon-marginal per-scale-temp calibration:"
    echo "  Cell:        ${RUN_NAME}"
    echo "  Arms:        ${ARMS}"
    echo "  Data target: ${TRAIN_TOKENS}  (bias_correct=${BIAS_CORRECT}, target=${PER_SCALE_TARGET})"
    echo "  Curve:       reused global-T rollouts T={${REUSED_TEMPS}} under ${TEMPOPT_RUN}/T<tp>/rollout/"
    echo "  Static:      coarse-warm {${COARSE_WARMS}} | fine-warm {${FINE_WARMS}} (n_fine=${N_FINE})"
    echo "  Rollout:     N=${N_TRAJ} steps=${N_STEPS} ic_chunk=${IC_CHUNK} posmask ON"
    echo "  Walltime:    calib=${WT_CALIB}  static=${WT_STATIC}"
    echo "  Wandb:       ${WANDB_PROJECT} / group=${WANDB_GROUP} / run=${RUN_NAME}-<arm>"
    echo "  Output:      ${CELL_DIR}/<arm>/"
    exit 0
fi

echo "=========================================="
echo "per-scale-temp calibration (N=${N_TRAJ})"
echo "  Cell:  ${RUN_NAME}    Arms: ${ARMS}"
echo "  Curve: reused T={${REUSED_TEMPS}}   target=${PER_SCALE_TARGET}  bias=${BIAS_CORRECT}"
echo "  Wandb: ${WANDB_PROJECT} / ${WANDB_GROUP}    Dry run: ${DRY_RUN}"
echo "=========================================="

if [ "${DRY_RUN}" = false ]; then
    [ -f "${CHECKPOINT_DIR}/training_state.json" ] || { echo "[abort] no NSP checkpoint at ${CHECKPOINT_DIR}" >&2; exit 1; }
    [ -f "${VAL_TOKENS}" ]   || { echo "[abort] no val tokens at ${VAL_TOKENS}" >&2; exit 1; }
    [ -f "${TRAIN_TOKENS}" ] || { echo "[abort] no train tokens at ${TRAIN_TOKENS}" >&2; exit 1; }
    [ -f "${VQVAE_DIR}/training_state.json" ] || { echo "[abort] no VQ-VAE checkpoint at ${VQVAE_DIR}" >&2; exit 1; }
    [ -d "${TEMPOPT_RUN}" ] || echo "[warn] no reused-rollout dir ${TEMPOPT_RUN}; the calib arm will have no curve."
    mkdir -p "${CELL_DIR}" "${CALIB_DIR}" "${LOG_DIR}" "${WANDB_BASE}"
fi

# Shared sbatch header + environment (matches sweep_drift_warm.sh).
SBATCH_HEADER() {
    local jobname="$1" walltime="$2"
    cat <<EOF
#!/bin/bash
#SBATCH -J ${jobname}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t ${walltime}
#SBATCH -o ${LOG_DIR}/${jobname}-%j.out
#SBATCH -e ${LOG_DIR}/${jobname}-%j.err

set -euo pipefail
cd "${REPODIR}"
source "${OCEAN}/.venvs/gust/bin/activate"
module load cuda/12.6.1
export PYTHONUNBUFFERED=1
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\n' ':'):\${LD_LIBRARY_PATH:-}
echo "Job \${SLURM_JOB_ID} on \$(hostname)  started \$(date)"
EOF
}

# Emit the prep step (data target + curve measurement + solve). Writes
# ${CALIB_DIR}/{data-target.json, marg-T<tp>.json, calib-perscale.json, calib-percell.json}.
# Idempotent (file-existence guards) so a resubmit / concurrent arm is race-safe.
PREP_BLOCK() {
    cat <<EOF
DATA_JSON="${CALIB_DIR}/data-target.json"
if [ ! -f "\${DATA_JSON}" ]; then
    echo "---- data target (train tokens, bias=${BIAS_CORRECT}) ----"
    python measure_tokenizer_entropy.py --tokens_path "${TRAIN_TOKENS}" \\
        --output "\${DATA_JSON}" --bias_correct ${BIAS_CORRECT}
fi
POINTS=""
for T in ${REUSED_TEMPS}; do
    TP=\${T/./p}
    RTOK="${TEMPOPT_RUN}/T\${TP}/rollout/rollout_tokens.npz"
    MJSON="${CALIB_DIR}/marg-T\${TP}.json"
    if [ -f "\${RTOK}" ]; then
        if [ ! -f "\${MJSON}" ]; then
            echo "---- rollout marginal T=\${T} ----"
            python measure_rollout_marginal.py --rollout "\${RTOK}" \\
                --output "\${MJSON}" --data_ref "\${DATA_JSON}" --bias_correct ${BIAS_CORRECT}
        fi
        POINTS="\${POINTS} --point \${T} \${MJSON}"
    else
        echo "[warn] reused rollout missing: \${RTOK} (T=\${T} dropped from curve)"
    fi
done
if [ -z "\${POINTS}" ]; then echo "[abort] no curve points found under ${TEMPOPT_RUN}" >&2; exit 1; fi
if [ ! -f "${CALIB_DIR}/calib-perscale.json" ] || [ ! -f "${CALIB_DIR}/calib-percell.json" ]; then
    echo "---- solve calibrated schedules ----"
    python solve_calib_schedule.py --data "\${DATA_JSON}" \${POINTS} \\
        --per_scale_target ${PER_SCALE_TARGET} \\
        --out_per_scale "${CALIB_DIR}/calib-perscale.json" \\
        --out_per_cell  "${CALIB_DIR}/calib-percell.json"
fi
EOF
}

# Emit a rollout + analyze block for one arm (ANCHOR_FLAG selects per-scale/per-cell/static).
ROLLOUT_ANALYZE_BLOCK() {
    local arm="$1" anchor_flag="$2"
    local rout="${CELL_DIR}/${arm}/rollout" aout="${CELL_DIR}/${arm}/analysis"
    cat <<EOF
echo "==== arm ${arm} ===="
mkdir -p "${CELL_DIR}/${arm}"
if [ -f "${aout}/metrics.json" ]; then
    echo "[skip] ${arm}: analysis already complete"
else
    if [ -f "${rout}/rollout_tokens.npz" ]; then
        echo "[reuse] ${arm}: rollout exists, analysis only"
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
            --sampler per_scale_temp ${anchor_flag} \\
            --output_dir "${rout}"
    fi
    python analyze_rollout.py \\
        --rollout_dir "${rout}" \\
        --vqvae_dir "${VQVAE_DIR}" \\
        --data_path "${DATA_PATH}" \\
        --output_dir "${aout}" \\
        --batch_size ${BATCH_SIZE} \\
        --seed ${SEED} \\
        --wandb_project ${WANDB_PROJECT} \\
        --wandb_name "${RUN_NAME}-${arm}" \\
        --wandb_group "${WANDB_GROUP}" \\
        --wandb_dir "${WANDB_BASE}"
fi
EOF
}

submit() {  # submit <tmpfile> <label>
    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] would submit ${2}"
    else
        echo "Submitting ${2}..."
        echo "  -> $(sbatch --parsable "$1")"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
    rm -f "$1"
}

N_SUBMITTED=0

# ---------- calib arm: one job, prep + per-scale + per-cell ----------
if [[ " ${ARMS} " == *" calib "* ]]; then
    TMP="$(mktemp /tmp/pst_calib_${RUN_NAME}_XXXXXX.sbatch)"
    {
        SBATCH_HEADER "pst-${RUN_NAME}-calib" "${WT_CALIB}"
        PREP_BLOCK
        ROLLOUT_ANALYZE_BLOCK "calibPS" "--entropy_anchor_path ${CALIB_DIR}/calib-perscale.json"
        ROLLOUT_ANALYZE_BLOCK "calibPC" "--percell_anchor_path ${CALIB_DIR}/calib-percell.json"
        echo 'echo "Finished $(date)"'
    } > "${TMP}"
    submit "${TMP}" "${RUN_NAME} calib (per-scale + per-cell)"
fi

# ---------- static controls: one job per warm value ----------
if [[ " ${ARMS} " == *" static "* ]]; then
    DATA_JSON_STATIC="${CALIB_DIR}/data-target.json"
    for spec in $(for w in ${COARSE_WARMS}; do echo "coarse:${w}"; done; \
                  for w in ${FINE_WARMS};   do echo "fine:${w}";   done); do
        GROUP="${spec%%:*}"; WARM="${spec##*:}"; WTP="${WARM/./p}"
        ARM="${GROUP}T${WTP}"
        ANCHOR="${CELL_DIR}/${ARM}/static-anchor.json"
        TMP="$(mktemp /tmp/pst_${ARM}_${RUN_NAME}_XXXXXX.sbatch)"
        {
            SBATCH_HEADER "pst-${RUN_NAME}-${ARM}" "${WT_STATIC}"
            cat <<EOF
mkdir -p "${CELL_DIR}/${ARM}"
# Build a light data target only for the trainable-scale layout, then hand-set
# the ${GROUP}-warm schedule (warm value ${WARM} on the ${GROUP} scales, 1.0 else).
if [ ! -f "${DATA_JSON_STATIC}" ]; then
    python measure_tokenizer_entropy.py --tokens_path "${TRAIN_TOKENS}" \\
        --output "${DATA_JSON_STATIC}" --bias_correct ${BIAS_CORRECT}
fi
python - <<PY
import json
d = json.load(open("${DATA_JSON_STATIC}"))
tr = d["trainable_scale_indices"]; n = len(tr); nf = min(${N_FINE}, n)
T = float(${WARM})
sched = ([T]*(n-nf) + [1.0]*nf) if "${GROUP}" == "coarse" else ([1.0]*(n-nf) + [T]*nf)
out = {"source": "static_control_${GROUP}", "warm": T, "n_fine": nf,
       "scales": d["scales"], "first_trainable_scale": d["first_trainable_scale"],
       "trainable_scale_indices": tr,
       "per_trainable_pooled_marginal_nats": sched,
       "per_trainable_mean_per_position_nats": sched}
json.dump(out, open("${ANCHOR}", "w"), indent=2)
print("static ${GROUP}-warm schedule:", sched)
PY
EOF
            ROLLOUT_ANALYZE_BLOCK "${ARM}" "--entropy_anchor_path ${ANCHOR}"
            echo 'echo "Finished $(date)"'
        } > "${TMP}"
        submit "${TMP}" "${RUN_NAME} ${ARM} (${GROUP}-warm ${WARM})"
    done
fi

echo "=========================================="
echo "Submitted ${N_SUBMITTED} job(s) for ${RUN_NAME}."
echo "Judge EMD/PDF/OOD/spectra vs the yardstick (NEVER entropy-match, NEVER TKE RSE):"
echo "  best_T=2.0 EMD 0.196 (2.40x floor 0.0816); etmodel ~73x floor."
echo "Calibration shape (the answer): inspect ${CALIB_DIR}/calib-perscale.json T_k +"
echo "  solve_status; clamped_lo/saturated_hi => rerun with --supp-temps to extend the curve."
echo "=========================================="
