#!/usr/bin/env bash
set -e

python3 tools/convert_visdrone_to_yolo.py \
  --src-root data/raw/VisDrone/VisDrone2019-DET-train \
  --out-root data/processed/visdrone \
  --split train \
  --classes-config configs/classes_visdrone.yaml \
  --copy-images

python3 tools/convert_visdrone_to_yolo.py \
  --src-root data/raw/VisDrone/VisDrone2019-DET-val \
  --out-root data/processed/visdrone \
  --split val \
  --classes-config configs/classes_visdrone.yaml \
  --copy-images

echo "[INFO] Converted VisDrone YOLO file counts:"
echo -n "images/train: "
find data/processed/visdrone/images/train -type f | wc -l
echo -n "labels/train: "
find data/processed/visdrone/labels/train -type f | wc -l
echo -n "images/val: "
find data/processed/visdrone/images/val -type f | wc -l
echo -n "labels/val: "
find data/processed/visdrone/labels/val -type f | wc -l
