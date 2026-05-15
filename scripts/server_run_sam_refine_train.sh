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
DINO_JSON_DIR="${DINO_JSON_DIR:-outputs/grounding_dino_json/train}"
OUTPUT_JSON_DIR="${OUTPUT_JSON_DIR:-outputs/sam_refine_json/train}"
VIS_OUTPUT_DIR="${VIS_OUTPUT_DIR:-results/visualizations/sam_refine_train}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_h_4b8939.pth}"
MODEL_TYPE="${MODEL_TYPE:-vit_h}"
DEVICE="${DEVICE:-cuda}"

mkdir -p logs
LOG_FILE="logs/server_run_sam_refine_train_$(date +%Y%m%d_%H%M%S).log"

{
  echo "[INFO] Project root: ${PROJECT_ROOT}"
  echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "[INFO] Input image dir: ${IMAGE_DIR}"
  echo "[INFO] Input DINO JSON dir: ${DINO_JSON_DIR}"
  echo "[INFO] Output SAM JSON dir: ${OUTPUT_JSON_DIR}"
  echo "[INFO] Visualization dir: ${VIS_OUTPUT_DIR}"
  echo "[INFO] SAM checkpoint: ${SAM_CHECKPOINT}"
  echo "[INFO] SAM model type: ${MODEL_TYPE}"
  echo "[INFO] Device: ${DEVICE}"
  echo "[INFO] Log file: ${LOG_FILE}"

  "${PYTHON_BIN}" tools/run_sam_refine_batch.py \
    --image-dir "${IMAGE_DIR}" \
    --dino-json-dir "${DINO_JSON_DIR}" \
    --output-json-dir "${OUTPUT_JSON_DIR}" \
    --vis-output-dir "${VIS_OUTPUT_DIR}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --model-type "${MODEL_TYPE}" \
    --device "${DEVICE}" \
    --skip-existing
} 2>&1 | tee "${LOG_FILE}"
