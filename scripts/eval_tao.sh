#!/bin/bash
# The three inference methods on one TAO subset (default LaSOT), same task-1 RandBox weights.
#   bash scripts/eval_tao.sh [SUBSET] [WEIGHTS]
set -euo pipefail
SUBSET=${1:-LaSOT}
WEIGHTS=${2:-models/randbox_t1.pth}
FRAMES=datasets/tao/frames_${SUBSET,,}.pkl

[ -f "$FRAMES" ] || python tools/prepare_tao.py --tao-root datasets/tao --subsets "$SUBSET" --out "$FRAMES"

for METHOD in ta-randbox randbox-500 randbox-10k; do
  python tools/eval_tao.py --config-file configs/t1.yaml --frames "$FRAMES" --method "$METHOD" \
    MODEL.WEIGHTS "$WEIGHTS" OUTPUT_DIR "output/tao_${SUBSET,,}_${METHOD}"
done
