#!/usr/bin/env bash
# run_drift_sweep.sh — per-scale token-DISTRIBUTION drift over the N=128
# scaling-tempopt rollout tree, ALL 3 token counts. Pure numpy + wandb (no GPU):
# fans out one CPU (RM-shared) job per VQ size for parallel I/O. Each job logs
# wandb gust2-drift-<size> (group=<sc>, name=<size>-<sc>-nsp-<arch>-T<tp>) and a
# per-size CSV. Pull/plot locally with plot_scale_drift.py.
#
#   ./scripts/bridges/run_drift_sweep.sh            # submit all 3 sizes
#   ./scripts/bridges/run_drift_sweep.sh --list     # print what would submit
#   SIZES="large" ./scripts/bridges/run_drift_sweep.sh
set -euo pipefail

OCEAN="/ocean/projects/mth260004p/sambamur"
CODE="${OCEAN}/gust"
TEMPOPT_BASE="${OCEAN}/experiments/scaling-tempopt-n128"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"
LOG_DIR="${TEMPOPT_BASE}/logs"
SIZES="${SIZES:-small medium large}"
WALLTIME="${WALLTIME:-2:00:00}"          # pure-numpy I/O sweep over ~75 npz/size
mkdir -p "${LOG_DIR}"

if [[ "${1:-}" == "--list" ]]; then
    for SIZE in ${SIZES}; do
        echo "=== ${SIZE} ==="
        ls -d ${TEMPOPT_BASE}/${SIZE}-sc*-nsp-s*/T*/rollout/rollout_tokens.npz 2>/dev/null | wc -l
    done
    exit 0
fi

for SIZE in ${SIZES}; do
    sbatch <<EOF
#!/usr/bin/env bash
#SBATCH -J drift-${SIZE}
#SBATCH -A ${ACCOUNT}
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node 4
#SBATCH -t ${WALLTIME}
#SBATCH -o ${LOG_DIR}/drift-${SIZE}-%j.out
#SBATCH -e ${LOG_DIR}/drift-${SIZE}-%j.err
set -euo pipefail
source "${OCEAN}/.venvs/gust/bin/activate"
cd "${CODE}"
python -u measure_drift_sweep.py \\
    --sweep_root "${TEMPOPT_BASE}" \\
    --sizes ${SIZE} \\
    --wandb_project_prefix gust2-drift \\
    --wandb_dir "${WANDB_BASE}" \\
    --csv "${TEMPOPT_BASE}/drift_sweep_${SIZE}.csv"
EOF
    echo "submitted drift sweep for ${SIZE}"
done
