#!/usr/bin/env bash
#
# Download VQ-VAE codebook artifacts from Bridges-2 to a local mirror so
# analyze_codebooks.py can run locally with no GPU dependency.
#
# Follows PSC's documented rsync pattern for Bridges-2:
#     rsync -rltpDv -e 'ssh -l <PSC-username>' \
#           data.bridges2.psc.edu:<source>  <target>
# `data.bridges2.psc.edu` is PSC's dedicated data-transfer host (not a login node).
#
# Usage:
#   PSC_USER=sambamur ./scripts/local/download_codebook_artifacts.sh
#   ./scripts/local/download_codebook_artifacts.sh sambamur
#
# What gets downloaded (~525 MB total):
#   codebook_artifacts/vqvae/{small,medium,large}-sc{341,917,1941}/{ema_state.npz, config.txt}
#   codebook_artifacts/tokens/{small,medium,large}-sc{341,917,1941}{,-val}.npz

set -euo pipefail

PSC_USER="${1:-${PSC_USER:-}}"
if [[ -z "$PSC_USER" ]]; then
    echo "Error: pass your PSC username as arg or set \$PSC_USER" >&2
    echo "  example: PSC_USER=sambamur $0" >&2
    exit 1
fi

DATA_HOST="data.bridges2.psc.edu"
REMOTE_ROOT="/ocean/projects/mth260004p/sambamur/experiments"
LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)/codebook_artifacts"

# Flags from PSC docs (rltpDv), plus --partial --info=progress2 for resume
# safety and a live throughput readout.
RSYNC_FLAGS=(-rltpDv --partial --info=progress2)
SSH_AUTH=("-e" "ssh -l $PSC_USER")

echo "PSC user: $PSC_USER"
echo "Remote:   $DATA_HOST:$REMOTE_ROOT"
echo "Local:    $LOCAL_ROOT"
mkdir -p "$LOCAL_ROOT/vqvae" "$LOCAL_ROOT/tokens"

echo
echo "==> Stage 1/2: pulling ema_state.npz + config.txt for 9 VQ-VAEs..."
rsync "${RSYNC_FLAGS[@]}" "${SSH_AUTH[@]}" \
    --include='*/' \
    --include='ema_state.npz' \
    --include='config.txt' \
    --exclude='*' \
    "$DATA_HOST:$REMOTE_ROOT/vqvae/" \
    "$LOCAL_ROOT/vqvae/"

echo
echo "==> Stage 2/2: pulling token .npz files (train + val) for 9 VQ-VAEs..."
rsync "${RSYNC_FLAGS[@]}" "${SSH_AUTH[@]}" \
    --include='*-sc341.npz' --include='*-sc341-val.npz' \
    --include='*-sc917.npz' --include='*-sc917-val.npz' \
    --include='*-sc1941.npz' --include='*-sc1941-val.npz' \
    --exclude='*' \
    "$DATA_HOST:$REMOTE_ROOT/tokens/" \
    "$LOCAL_ROOT/tokens/"

echo
echo "==> Done. Summary:"
du -sh "$LOCAL_ROOT/vqvae" "$LOCAL_ROOT/tokens" 2>/dev/null || true
ls -1 "$LOCAL_ROOT/vqvae" 2>/dev/null | sed 's/^/  vqvae:  /' || true
ls -1 "$LOCAL_ROOT/tokens" 2>/dev/null | sed 's/^/  tokens: /' || true
echo
echo "Run the analysis with:"
echo "  ~/llm/bin/python analyze_codebooks.py"
