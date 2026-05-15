#!/usr/bin/env bash
set -e

yolo detect val \
  model=runs/yolov8s_auto_sam_refine_visdrone/weights/best.pt \
  data=configs/yolo_visdrone_manual.yaml \
  imgsz=1024 \
  batch=16 \
  device=0 \
  project=results/yolo_eval \
  name=auto_sam_refine_val
