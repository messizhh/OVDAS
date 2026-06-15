#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
IMAGE_DIR="${IMAGE_DIR:-data/processed/visdrone/images/train}"
GT_LABEL_DIR="${GT_LABEL_DIR:-data/processed/visdrone/labels/train}"
SUBSET_LIST="${SUBSET_LIST:-outputs/experiment_subsets/visdrone_train_seed42_200.txt}"
CLASSES_CONFIG="${CLASSES_CONFIG:-configs/classes_visdrone.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tile_ablation_subset}"
TABLE_DIR="${TABLE_DIR:-results/tables}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.5}"

SUMMARY_CSV="${SUMMARY_CSV:-${TABLE_DIR}/tile_ablation_subset_summary.csv}"
CLASS_CSV="${CLASS_CSV:-${TABLE_DIR}/tile_ablation_subset_by_class.csv}"
SIZE_CSV="${SIZE_CSV:-${TABLE_DIR}/tile_ablation_subset_by_size.csv}"

mkdir -p logs "${TABLE_DIR}"
LOG_FILE="logs/server_eval_tile_ablation_subset_$(date +%Y%m%d_%H%M%S).log"

eval_method() {
  local method="$1"
  local label_dir="$2"
  local metadata_json_dir="$3"
  echo "[INFO] Evaluating method=${method}"
  "${PYTHON_BIN}" tools/evaluate_auto_labels.py \
    --gt-label-dir "${GT_LABEL_DIR}" \
    --pred-label-dir "${label_dir}" \
    --image-dir "${IMAGE_DIR}" \
    --image-list "${SUBSET_LIST}" \
    --metadata-json-dir "${metadata_json_dir}" \
    --classes-config "${CLASSES_CONFIG}" \
    --out-summary-csv "${SUMMARY_CSV}" \
    --out-class-csv "${CLASS_CSV}" \
    --out-size-csv "${SIZE_CSV}" \
    --iou-threshold "${IOU_THRESHOLD}" \
    --method-name "${method}" \
    --append-csv
}

{
  echo "[INFO] Start time: $(date -Is)"
  echo "[INFO] Project root: ${PROJECT_ROOT}"
  echo "[INFO] Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "[INFO] Python: $("${PYTHON_BIN}" --version)"
  "${PYTHON_BIN}" -c 'import importlib.util; spec=importlib.util.find_spec("torch"); print("[INFO] PyTorch:", "not installed" if spec is None else __import__("torch").__version__); print("[INFO] CUDA available:", "unknown" if spec is None else __import__("torch").cuda.is_available()); print("[INFO] CUDA version:", "unknown" if spec is None else __import__("torch").version.cuda)'
  echo "[INFO] Image dir: ${IMAGE_DIR}"
  echo "[INFO] GT label dir: ${GT_LABEL_DIR}"
  echo "[INFO] Subset list: ${SUBSET_LIST}"
  echo "[INFO] Output root: ${OUTPUT_ROOT}"
  echo "[INFO] IoU threshold: ${IOU_THRESHOLD}"
  echo "[INFO] Summary CSV: ${SUMMARY_CSV}"
  echo "[INFO] Class CSV: ${CLASS_CSV}"
  echo "[INFO] Size CSV: ${SIZE_CSV}"
  echo "[INFO] Log file: ${LOG_FILE}"

  rm -f "${SUMMARY_CSV}" "${CLASS_CSV}" "${SIZE_CSV}"

  eval_method "DINO-only" \
    "${OUTPUT_ROOT}/dino_only_035/labels" \
    "${OUTPUT_ROOT}/dino_only_035/grounding_json"

  eval_method "DINO+SAM" \
    "${OUTPUT_ROOT}/dino_sam_035/labels" \
    "${OUTPUT_ROOT}/dino_sam_035/sam_json"

  eval_method "DINO-Tile-0.35" \
    "${OUTPUT_ROOT}/dino_tile_035/labels" \
    "${OUTPUT_ROOT}/dino_tile_035/grounding_json"

  eval_method "DINO-Tile-0.25" \
    "${OUTPUT_ROOT}/dino_tile_025/labels" \
    "${OUTPUT_ROOT}/dino_tile_025/grounding_json"

  eval_method "OVDAS-Tile" \
    "${OUTPUT_ROOT}/ovdas_tile/labels" \
    "${OUTPUT_ROOT}/ovdas_tile/sam_json"

  echo "[INFO] End time: $(date -Is)"
} 2>&1 | tee "${LOG_FILE}"
