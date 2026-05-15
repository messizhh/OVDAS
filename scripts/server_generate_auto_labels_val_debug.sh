#!/usr/bin/env bash
set -e

python3 tools/generate_yolo_labels_from_auto.py \
  --json-dir outputs/sam_refine_json/val_debug \
  --image-dir data/processed/visdrone/images/val \
  --out-label-dir outputs/auto_labels/sam_refine_val_debug/labels \
  --classes-config configs/classes_visdrone.yaml \
  --bbox-key refined_bbox_xyxy \
  --fallback-bbox-key bbox_xyxy \
  --score-threshold 0.35 \
  --min-box-area 4 \
  --stats-csv results/tables/auto_label_statistics_sam_refine_val_debug.csv \
  --limit 20 \
  --skip-existing
