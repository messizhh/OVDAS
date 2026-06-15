#!/usr/bin/env bash
set -e

MIN_REFINE_AREA_PX="${MIN_REFINE_AREA_PX:-0}"

python3 tools/run_sam_refine_batch.py \
  --image-dir data/processed/visdrone/images/val \
  --dino-json-dir outputs/grounding_dino_json/val_debug \
  --output-json-dir outputs/sam_refine_json/val_debug \
  --vis-output-dir results/visualizations/sam_refine_val_debug \
  --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --device cuda \
  --min-refine-area-px "${MIN_REFINE_AREA_PX}" \
  --limit 20 \
  --skip-existing
