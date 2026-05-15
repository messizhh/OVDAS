#!/usr/bin/env bash
set -e

yolo detect train \
  model=yolov8s.pt \
  data=configs/yolo_visdrone_auto_dino.yaml \
  epochs=100 \
  imgsz=1024 \
  batch=16 \
  device=0 \
  project=runs \
  name=yolov8s_auto_dino_visdrone
