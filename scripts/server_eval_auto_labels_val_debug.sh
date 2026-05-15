#!/usr/bin/env bash
set -e

python3 tools/evaluate_auto_labels.py \
  --gt-label-dir data/processed/visdrone/labels/val \
  --pred-label-dir outputs/auto_labels/dino_val_debug/labels \
  --image-dir data/processed/visdrone/images/val \
  --classes-config configs/classes_visdrone.yaml \
  --out-summary-csv results/tables/auto_label_quality_dino_val_debug_summary.csv \
  --out-class-csv results/tables/auto_label_quality_dino_val_debug_by_class.csv \
  --out-size-csv results/tables/auto_label_quality_dino_val_debug_by_size.csv \
  --iou-threshold 0.5

python3 tools/evaluate_auto_labels.py \
  --gt-label-dir data/processed/visdrone/labels/val \
  --pred-label-dir outputs/auto_labels/sam_refine_val_debug/labels \
  --image-dir data/processed/visdrone/images/val \
  --classes-config configs/classes_visdrone.yaml \
  --out-summary-csv results/tables/auto_label_quality_sam_refine_val_debug_summary.csv \
  --out-class-csv results/tables/auto_label_quality_sam_refine_val_debug_by_class.csv \
  --out-size-csv results/tables/auto_label_quality_sam_refine_val_debug_by_size.csv \
  --iou-threshold 0.5
