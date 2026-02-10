#!/usr/bin/env bash
# Batch-process propagators / correlators over saved gauge configs.
#
# Usage:
#   bash scripts/run_propagators_batch.sh [CONFIG_DIR] [LS] [LT]
#
# Defaults:
#   CONFIG_DIR = configs/8x8x8x20_b2.40
#   LS = 8, LT = 20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PROP_DIR="$REPO_DIR/su2/meson_correlator"

CONFIG_DIR="${1:-$REPO_DIR/configs/8x8x8x20_b2.40}"
LS="${2:-8}"
LT="${3:-20}"
MASS="0.2"
WILSON_R="0.5"

echo "=== Batch Propagator / Correlator Processing ==="
echo "  config dir = $CONFIG_DIR"
echo "  lattice    = ${LS}^3 x ${LT}"
echo "  mass       = $MASS"
echo "  wilson r   = $WILSON_R"
echo ""

COUNT=0
TOTAL=$(ls "$CONFIG_DIR"/*.pkl 2>/dev/null | wc -l)

for cfg in "$CONFIG_DIR"/*.pkl; do
    COUNT=$((COUNT + 1))
    BASENAME=$(basename "$cfg" .pkl)
    echo "[$COUNT/$TOTAL] Processing $BASENAME ..."

    cd "$PROP_DIR"
    PYTHONPATH="$REPO_DIR/su2:${PYTHONPATH:-}" python3 PropagatorModular.py \
        --input-config "$cfg" \
        --mass "$MASS" \
        --wilson-r "$WILSON_R" \
        --ls "$LS" --lt "$LT" \
        --channel all \
        --save-correlators \
        --no-plots \
        || echo "  WARNING: failed on $BASENAME"
done

echo ""
echo "Batch complete: processed $COUNT / $TOTAL configurations."
