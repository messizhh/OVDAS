#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
yolo settings datasets_dir="$(pwd)"

yolo detect train \
  model=yolov8s.pt \
  data=configs/yolo_visdrone_auto_sam_refine.yaml \
  epochs=100 \
  imgsz=1024 \
  batch=16 \
  device=0 \
  project=runs \
  name=yolov8s_auto_sam_refine_visdrone
