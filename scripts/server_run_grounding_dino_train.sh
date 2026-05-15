#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
set -o pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
IMAGE_DIR="${IMAGE_DIR:-data/processed/visdrone/images/train}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grounding_dino_json/train}"
PROMPT="${PROMPT:-pedestrian. people. bicycle. car. van. truck. bus. motor.}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.35}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
DEVICE="${DEVICE:-cuda}"
DINO_CONFIG_FILE="${DINO_CONFIG_FILE:-external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-checkpoints/groundingdino_swint_ogc.pth}"

mkdir -p logs
LOG_FILE="logs/server_run_grounding_dino_train_$(date +%Y%m%d_%H%M%S).log"

{
  echo "[INFO] Project root: ${PROJECT_ROOT}"
  echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "[INFO] Input image dir: ${IMAGE_DIR}"
  echo "[INFO] Output JSON dir: ${OUTPUT_DIR}"
  echo "[INFO] Prompt: ${PROMPT}"
  echo "[INFO] Box threshold: ${BOX_THRESHOLD}"
  echo "[INFO] Text threshold: ${TEXT_THRESHOLD}"
  echo "[INFO] Device: ${DEVICE}"
  echo "[INFO] Grounding DINO config: ${DINO_CONFIG_FILE}"
  echo "[INFO] Grounding DINO checkpoint: ${DINO_CHECKPOINT}"
  echo "[INFO] Log file: ${LOG_FILE}"

  "${PYTHON_BIN}" tools/run_grounding_dino_batch.py \
    --image-dir "${IMAGE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --prompt "${PROMPT}" \
    --box-threshold "${BOX_THRESHOLD}" \
    --text-threshold "${TEXT_THRESHOLD}" \
    --device "${DEVICE}" \
    --config-file "${DINO_CONFIG_FILE}" \
    --checkpoint "${DINO_CHECKPOINT}" \
    --skip-existing
} 2>&1 | tee "${LOG_FILE}"
