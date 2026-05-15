#!/usr/bin/env bash
set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
yolo settings datasets_dir="$(pwd)"

MODEL="${MODEL:-yolov8s.pt}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-1024}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
RUN_DINO_ONLY="${RUN_DINO_ONLY:-1}"
RUN_DINO_SAM="${RUN_DINO_SAM:-1}"

DINO_ONLY_CONFIG="configs/yolo_visdrone_auto_dino_only.yaml"
DINO_SAM_CONFIG="configs/yolo_visdrone_auto_dino_sam.yaml"
DINO_ONLY_NAME="yolov8s_auto_dino_only_visdrone"
DINO_SAM_NAME="yolov8s_auto_dino_sam_visdrone"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "[INFO] MODEL: ${MODEL}"
echo "[INFO] EPOCHS: ${EPOCHS}"
echo "[INFO] IMGSZ: ${IMGSZ}"
echo "[INFO] BATCH: ${BATCH}"
echo "[INFO] DEVICE: ${DEVICE}"
echo "[INFO] WORKERS: ${WORKERS}"
echo "[INFO] RUN_DINO_ONLY: ${RUN_DINO_ONLY}"
echo "[INFO] RUN_DINO_SAM: ${RUN_DINO_SAM}"

if [[ "${RUN_DINO_ONLY}" == "1" ]]; then
  echo "[INFO] Training auto-label YOLO model: DINO-only"
  echo "[INFO] Data config: ${DINO_ONLY_CONFIG}"
  echo "[INFO] Output run: runs/${DINO_ONLY_NAME}"
  yolo detect train \
    model="${MODEL}" \
    data="${DINO_ONLY_CONFIG}" \
    epochs="${EPOCHS}" \
    imgsz="${IMGSZ}" \
    batch="${BATCH}" \
    device="${DEVICE}" \
    workers="${WORKERS}" \
    project=runs \
    name="${DINO_ONLY_NAME}"
fi

if [[ "${RUN_DINO_SAM}" == "1" ]]; then
  echo "[INFO] Training auto-label YOLO model: DINO+SAM"
  echo "[INFO] Data config: ${DINO_SAM_CONFIG}"
  echo "[INFO] Output run: runs/${DINO_SAM_NAME}"
  yolo detect train \
    model="${MODEL}" \
    data="${DINO_SAM_CONFIG}" \
    epochs="${EPOCHS}" \
    imgsz="${IMGSZ}" \
    batch="${BATCH}" \
    device="${DEVICE}" \
    workers="${WORKERS}" \
    project=runs \
    name="${DINO_SAM_NAME}"
fi

if [[ "${RUN_DINO_ONLY}" != "1" && "${RUN_DINO_SAM}" != "1" ]]; then
  echo "[WARN] Nothing to train because RUN_DINO_ONLY and RUN_DINO_SAM are both disabled."
fi
