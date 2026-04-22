#!/bin/bash
# Master submitter for multistep analysis on Derecho: submits sweep_rollout.sh
# and then sweep_analysis.sh chained via PBS `-W depend=afterok:<jobid>`.
#
# Both child jobs are node-exclusive (4 A100-40), 12h walltime each. The
# analysis job is queued immediately but held by PBS until the rollout job
# completes successfully, so you can fire-and-forget with one command.
#
# All extra args are forwarded to BOTH child scripts (e.g. an NSP filter or
# --vqvae applies to both stages). --dry-run runs both children in dry-run
# mode without submitting.
#
# Usage:
#   ./scripts/derecho/sweep_multistep.sh sc341                    All 9 sc341 combos
#   ./scripts/derecho/sweep_multistep.sh sc917 small              sc917, small NSP only
#   ./scripts/derecho/sweep_multistep.sh sc341 --vqvae medium     Only medium VQ-VAE (3)
#   ./scripts/derecho/sweep_multistep.sh sc341 --dry-run

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROLLOUT_SCRIPT="${DIR}/sweep_rollout.sh"
ANALYSIS_SCRIPT="${DIR}/sweep_analysis.sh"

# Detect --dry-run anywhere in the args so we can handle it specially.
DRY_RUN=false
for arg in "$@"; do
    if [ "${arg}" = "--dry-run" ]; then
        DRY_RUN=true
        break
    fi
done

echo "=========================================="
echo "Multistep Sweep (rollout -> analysis)"
echo "  Forwarded args: $*"
echo "  Dry run:        ${DRY_RUN}"
echo "=========================================="

if [ "${DRY_RUN}" = true ]; then
    echo ""
    echo "--- Rollout (dry-run) ---"
    "${ROLLOUT_SCRIPT}" "$@"
    echo ""
    echo "--- Analysis (dry-run, without --depend) ---"
    "${ANALYSIS_SCRIPT}" "$@"
    exit 0
fi

echo ""
echo "--- Stage 1: submitting rollout ---"
# Tee the rollout submission output so the user sees it and we can grep for
# the parseable JOBID=<id> marker emitted by sweep_rollout.sh.
ROLLOUT_LOG="$(mktemp /tmp/sweep_multistep_rollout_XXXXXX.log)"
trap 'rm -f "${ROLLOUT_LOG}"' EXIT
set +e
"${ROLLOUT_SCRIPT}" "$@" | tee "${ROLLOUT_LOG}"
ROLLOUT_RC=${PIPESTATUS[0]}
set -e

if [ ${ROLLOUT_RC} -ne 0 ]; then
    echo ""
    echo "Rollout script exited with rc=${ROLLOUT_RC}. Not submitting analysis." >&2
    exit ${ROLLOUT_RC}
fi

ROLLOUT_JOBID=$(grep -E '^JOBID=' "${ROLLOUT_LOG}" | tail -1 | cut -d= -f2- || true)

echo ""
if [ -z "${ROLLOUT_JOBID}" ]; then
    # sweep_rollout.sh exits 0 with no JOBID when there is nothing to submit
    # (all rollouts already exist or prerequisites missing). In that case we
    # still want analysis to run on whatever IS available — just without a
    # PBS dependency.
    echo "No rollout job was submitted (nothing to do or all skipped)."
    echo "--- Stage 2: submitting analysis (no dependency) ---"
    "${ANALYSIS_SCRIPT}" "$@"
else
    echo "--- Stage 2: submitting analysis with --depend ${ROLLOUT_JOBID} ---"
    "${ANALYSIS_SCRIPT}" "$@" --depend "${ROLLOUT_JOBID}"
fi

echo ""
echo "=========================================="
echo "Multistep submission complete."
echo "=========================================="
