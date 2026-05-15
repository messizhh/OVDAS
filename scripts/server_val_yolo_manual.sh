#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
yolo settings datasets_dir="$(pwd)"

yolo detect val \
  model=runs/yolov8s_manual_visdrone/weights/best.pt \
  data=configs/yolo_visdrone_manual.yaml \
  imgsz=1024 \
  batch=16 \
  device=0 \
  project=results/yolo_eval \
  name=manual_val
