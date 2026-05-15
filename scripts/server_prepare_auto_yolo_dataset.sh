#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LINK_MODE="${LINK_MODE:-symlink}"
CLASSES_CONFIG="${CLASSES_CONFIG:-configs/classes_visdrone.yaml}"

TRAIN_IMAGE_DIR="${TRAIN_IMAGE_DIR:-data/processed/visdrone/images/train}"
VAL_IMAGE_DIR="${VAL_IMAGE_DIR:-data/processed/visdrone/images/val}"
VAL_LABEL_DIR="${VAL_LABEL_DIR:-data/processed/visdrone/labels/val}"

DINO_LABEL_DIR="${DINO_LABEL_DIR:-outputs/auto_labels/dino_only/train}"
SAM_LABEL_DIR="${SAM_LABEL_DIR:-outputs/auto_labels/dino_sam/train}"
DINO_OUT_ROOT="${DINO_OUT_ROOT:-data/processed/visdrone_auto_yolo_dino_only}"
SAM_OUT_ROOT="${SAM_OUT_ROOT:-data/processed/visdrone_auto_yolo_dino_sam}"
DINO_CONFIG_PATH="${DINO_CONFIG_PATH:-configs/yolo_visdrone_auto_dino_only.yaml}"
SAM_CONFIG_PATH="${SAM_CONFIG_PATH:-configs/yolo_visdrone_auto_dino_sam.yaml}"

mkdir -p logs
LOG_FILE="logs/server_prepare_auto_yolo_dataset_$(date +%Y%m%d_%H%M%S).log"

{
  echo "[INFO] Project root: ${PROJECT_ROOT}"
  echo "[INFO] Link mode: ${LINK_MODE}"
  echo "[INFO] Classes config: ${CLASSES_CONFIG}"
  echo "[INFO] Train image dir: ${TRAIN_IMAGE_DIR}"
  echo "[INFO] Val image dir: ${VAL_IMAGE_DIR}"
  echo "[INFO] Val label dir: ${VAL_LABEL_DIR}"
  echo "[INFO] DINO-only train label dir: ${DINO_LABEL_DIR}"
  echo "[INFO] DINO-only output root: ${DINO_OUT_ROOT}"
  echo "[INFO] DINO-only config path: ${DINO_CONFIG_PATH}"
  echo "[INFO] DINO+SAM train label dir: ${SAM_LABEL_DIR}"
  echo "[INFO] DINO+SAM output root: ${SAM_OUT_ROOT}"
  echo "[INFO] DINO+SAM config path: ${SAM_CONFIG_PATH}"
  echo "[INFO] Log file: ${LOG_FILE}"

  echo "[INFO] Preparing DINO-only auto-label YOLO dataset."
  "${PYTHON_BIN}" tools/prepare_auto_yolo_dataset.py \
    --train-image-dir "${TRAIN_IMAGE_DIR}" \
    --train-label-dir "${DINO_LABEL_DIR}" \
    --val-image-dir "${VAL_IMAGE_DIR}" \
    --val-label-dir "${VAL_LABEL_DIR}" \
    --out-root "${DINO_OUT_ROOT}" \
    --config-path "${DINO_CONFIG_PATH}" \
    --classes-config "${CLASSES_CONFIG}" \
    --link-mode "${LINK_MODE}" \
    --replace-existing-links

  echo "[INFO] Preparing DINO+SAM auto-label YOLO dataset."
  "${PYTHON_BIN}" tools/prepare_auto_yolo_dataset.py \
    --train-image-dir "${TRAIN_IMAGE_DIR}" \
    --train-label-dir "${SAM_LABEL_DIR}" \
    --val-image-dir "${VAL_IMAGE_DIR}" \
    --val-label-dir "${VAL_LABEL_DIR}" \
    --out-root "${SAM_OUT_ROOT}" \
    --config-path "${SAM_CONFIG_PATH}" \
    --classes-config "${CLASSES_CONFIG}" \
    --link-mode "${LINK_MODE}" \
    --replace-existing-links
} 2>&1 | tee "${LOG_FILE}"
