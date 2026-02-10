#!/usr/bin/env bash
# Generate 200 thermalized 8^3 x 20 gauge configurations at beta=2.4
#
# Runtime: several hours depending on hardware.
# Run from terminal, NOT inside a notebook:
#   bash scripts/generate_8x8x8x20.sh
#
# Output: configs/8x8x8x20_b2.40/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SU2_DIR="$REPO_DIR/su2"
OUTPUT_DIR="$REPO_DIR/configs/8x8x8x20_b2.40"

mkdir -p "$OUTPUT_DIR"

echo "=== 8^3 x 20 Gauge Configuration Generation ==="
echo "  beta   = 2.4"
echo "  lattice = 8 x 8 x 8 x 20"
echo "  target  = 200 configs"
echo "  output  = $OUTPUT_DIR"
echo ""

cd "$SU2_DIR"
python3 Thermal_Generator.py \
    --mode cold \
    --beta 2.4 \
    --Ls 8 --Lt 20 \
    --trajectories 5000 \
    --save-interval 10 \
    --target-configs 200 \
    --checkpoint-interval 100 \
    --progress-interval 10 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Done! Configs saved to $OUTPUT_DIR"
echo "Count: $(ls "$OUTPUT_DIR"/*.pkl 2>/dev/null | wc -l) files"
