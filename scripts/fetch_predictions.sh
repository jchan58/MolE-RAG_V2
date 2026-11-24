#!/usr/bin/env bash
# Download and extract the saved per-molecule prediction files used to reproduce
# the paper's exact numbers (and inspect the model's selections per molecule).
# See docs/PREDICTIONS.md for the record schema and how to re-score.
set -euo pipefail

TAG="${MOLERAG_PRED_TAG:-predictions-v1}"
REPO="${MOLERAG_REPO:-jchan58/MolE-RAG_V2}"
ASSET="molerag_predictions.tar.gz"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"

DEST="${1:-predictions}"
mkdir -p "$DEST"

echo "Downloading ${URL}"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "$DEST/$ASSET" "$URL"
else
  wget -O "$DEST/$ASSET" "$URL"
fi

echo "Extracting into ${DEST}/ ..."
tar -xzf "$DEST/$ASSET" -C "$DEST"
rm -f "$DEST/$ASSET"

echo "Done. Contents:"
ls "$DEST"
echo
echo "To reproduce the framework table cells exactly:"
echo "  export MOLERAG_RUNS=\$PWD/${DEST}/runs"
echo "  python framework/score_fw_multi.py <dataset> <model>"
