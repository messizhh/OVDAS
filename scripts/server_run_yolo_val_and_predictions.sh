#!/usr/bin/env bash
set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
yolo settings datasets_dir="$(pwd)"

IMGSZ="${IMGSZ:-1024}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
CONF="${CONF:-0.25}"
DATA_CONFIG="${DATA_CONFIG:-configs/yolo_visdrone_manual.yaml}"
VAL_SOURCE="${VAL_SOURCE:-data/processed/visdrone/images/val}"
PREDICTION_ROOT="results/yolo_predictions"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "[INFO] IMGSZ: ${IMGSZ}"
echo "[INFO] BATCH: ${BATCH}"
echo "[INFO] DEVICE: ${DEVICE}"
echo "[INFO] CONF: ${CONF}"
echo "[INFO] DATA_CONFIG: ${DATA_CONFIG}"
echo "[INFO] VAL_SOURCE: ${VAL_SOURCE}"

echo "[INFO] Removing stale prediction directory: ${PREDICTION_ROOT}"
rm -rf "${PREDICTION_ROOT}"

run_val_and_predict() {
  local model_name="$1"
  local weight_path="$2"
  local val_name="$3"
  local pred_name="$4"

  echo "[INFO] Running YOLO val for ${model_name}: ${weight_path}"
  yolo detect val \
    model="${weight_path}" \
    data="${DATA_CONFIG}" \
    imgsz="${IMGSZ}" \
    batch="${BATCH}" \
    device="${DEVICE}" \
    project=results/yolo_eval \
    name="${val_name}" \
    exist_ok=True

  echo "[INFO] Running YOLO predict for ${model_name}: ${weight_path}"
  yolo detect predict \
    model="${weight_path}" \
    source="${VAL_SOURCE}" \
    imgsz="${IMGSZ}" \
    conf="${CONF}" \
    device="${DEVICE}" \
    save_txt=True \
    save_conf=True \
    project="${PREDICTION_ROOT}" \
    name="${pred_name}" \
    exist_ok=True
}

run_val_and_predict \
  "manual" \
  "runs/yolov8s_manual_visdrone/weights/best.pt" \
  "manual_val" \
  "manual"

run_val_and_predict \
  "auto_dino_only" \
  "runs/yolov8s_auto_dino_only_visdrone/weights/best.pt" \
  "auto_dino_only_val" \
  "auto_dino_only"

run_val_and_predict \
  "auto_dino_sam" \
  "runs/yolov8s_auto_dino_sam_visdrone/weights/best.pt" \
  "auto_dino_sam_val" \
  "auto_dino_sam"

echo "[INFO] YOLO val and prediction export finished."
