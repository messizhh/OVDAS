#!/usr/bin/env bash
set -e

python3 tools/run_grounding_dino_batch.py \
  --image-dir data/processed/visdrone/images/val \
  --output-dir outputs/grounding_dino_json/val_debug \
  --prompt "pedestrian. people. bicycle. car. van. truck. bus. motor." \
  --box-threshold 0.35 \
  --text-threshold 0.25 \
  --device cuda \
  --config-file external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --checkpoint checkpoints/groundingdino_swint_ogc.pth \
  --limit 20 \
  --skip-existing
