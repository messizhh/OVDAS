#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
IMAGE_DIR="${IMAGE_DIR:-data/processed/visdrone/images/train}"
LABEL_DIR="${LABEL_DIR:-data/processed/visdrone/labels/train}"
CLASSES_CONFIG="${CLASSES_CONFIG:-configs/classes_visdrone.yaml}"
NUM_IMAGES="${NUM_IMAGES:-200}"
SEED="${SEED:-42}"
SUBSET_DIR="${SUBSET_DIR:-outputs/experiment_subsets}"
OUTPUT_LIST="${OUTPUT_LIST:-${SUBSET_DIR}/visdrone_train_seed42_200.txt}"
OUTPUT_JSON="${OUTPUT_JSON:-${SUBSET_DIR}/visdrone_train_seed42_200.json}"
STATS_CSV="${STATS_CSV:-results/tables/visdrone_train_seed42_200_subset_stats.csv}"

mkdir -p logs
LOG_FILE="logs/server_prepare_tile_subset_$(date +%Y%m%d_%H%M%S).log"

{
  echo "[INFO] Start time: $(date -Is)"
  echo "[INFO] Project root: ${PROJECT_ROOT}"
  echo "[INFO] Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "[INFO] Python: $("${PYTHON_BIN}" --version)"
  "${PYTHON_BIN}" -c 'import importlib.util; spec=importlib.util.find_spec("torch"); print("[INFO] PyTorch:", "not installed" if spec is None else __import__("torch").__version__); print("[INFO] CUDA available:", "unknown" if spec is None else __import__("torch").cuda.is_available()); print("[INFO] CUDA version:", "unknown" if spec is None else __import__("torch").version.cuda)'
  echo "[INFO] Image dir: ${IMAGE_DIR}"
  echo "[INFO] Label dir: ${LABEL_DIR}"
  echo "[INFO] Classes config: ${CLASSES_CONFIG}"
  echo "[INFO] Num images: ${NUM_IMAGES}"
  echo "[INFO] Seed: ${SEED}"
  echo "[INFO] Output list: ${OUTPUT_LIST}"
  echo "[INFO] Output JSON: ${OUTPUT_JSON}"
  echo "[INFO] Stats CSV: ${STATS_CSV}"
  echo "[INFO] Log file: ${LOG_FILE}"

  "${PYTHON_BIN}" tools/create_fixed_experiment_subset.py \
    --image-dir "${IMAGE_DIR}" \
    --label-dir "${LABEL_DIR}" \
    --classes-config "${CLASSES_CONFIG}" \
    --num-images "${NUM_IMAGES}" \
    --seed "${SEED}" \
    --output-list "${OUTPUT_LIST}" \
    --output-json "${OUTPUT_JSON}" \
    --stats-csv "${STATS_CSV}"

  echo "[INFO] End time: $(date -Is)"
} 2>&1 | tee "${LOG_FILE}"
