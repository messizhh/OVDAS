#!/usr/bin/env bash
set -e

python3 tools/run_sam_refine_batch.py \
  --image-dir data/processed/visdrone/images/val \
  --dino-json-dir outputs/grounding_dino_json/val_debug \
  --output-json-dir outputs/sam_refine_json/val_debug \
  --vis-output-dir results/visualizations/sam_refine_val_debug \
  --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --device cuda \
  --limit 20 \
  --skip-existing
