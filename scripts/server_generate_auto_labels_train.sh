#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
IMAGE_DIR="${IMAGE_DIR:-data/processed/visdrone/images/train}"
CLASSES_CONFIG="${CLASSES_CONFIG:-configs/classes_visdrone.yaml}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.35}"
MIN_BOX_AREA="${MIN_BOX_AREA:-4}"
SKIP_EXISTING_LABELS="${SKIP_EXISTING_LABELS:-0}"

DINO_JSON_DIR="${DINO_JSON_DIR:-outputs/grounding_dino_json/train}"
SAM_JSON_DIR="${SAM_JSON_DIR:-outputs/sam_refine_json/train}"
DINO_LABEL_DIR="${DINO_LABEL_DIR:-outputs/auto_labels/dino_only/train}"
SAM_LABEL_DIR="${SAM_LABEL_DIR:-outputs/auto_labels/dino_sam/train}"
DINO_STATS_CSV="${DINO_STATS_CSV:-results/tables/auto_label_stats_train_dino_only.csv}"
SAM_STATS_CSV="${SAM_STATS_CSV:-results/tables/auto_label_stats_train_dino_sam.csv}"

mkdir -p logs
LOG_FILE="logs/server_generate_auto_labels_train_$(date +%Y%m%d_%H%M%S).log"

SKIP_ARGS=()
if [[ "${SKIP_EXISTING_LABELS}" == "1" ]]; then
  SKIP_ARGS+=(--skip-existing)
fi

{
  echo "[INFO] Project root: ${PROJECT_ROOT}"
  echo "[INFO] Input image dir: ${IMAGE_DIR}"
  echo "[INFO] Classes config: ${CLASSES_CONFIG}"
  echo "[INFO] Score threshold: ${SCORE_THRESHOLD}"
  echo "[INFO] Min box area: ${MIN_BOX_AREA}"
  echo "[INFO] Skip existing labels: ${SKIP_EXISTING_LABELS}"
  echo "[INFO] DINO-only JSON dir: ${DINO_JSON_DIR}"
  echo "[INFO] DINO-only label dir: ${DINO_LABEL_DIR}"
  echo "[INFO] DINO-only stats CSV: ${DINO_STATS_CSV}"
  echo "[INFO] DINO+SAM JSON dir: ${SAM_JSON_DIR}"
  echo "[INFO] DINO+SAM label dir: ${SAM_LABEL_DIR}"
  echo "[INFO] DINO+SAM stats CSV: ${SAM_STATS_CSV}"
  echo "[INFO] Log file: ${LOG_FILE}"

  echo "[INFO] Generating DINO-only train labels."
  "${PYTHON_BIN}" tools/generate_yolo_labels_from_auto.py \
    --json-dir "${DINO_JSON_DIR}" \
    --image-dir "${IMAGE_DIR}" \
    --out-label-dir "${DINO_LABEL_DIR}" \
    --classes-config "${CLASSES_CONFIG}" \
    --bbox-key bbox_xyxy \
    --fallback-bbox-key bbox_xyxy \
    --score-threshold "${SCORE_THRESHOLD}" \
    --min-box-area "${MIN_BOX_AREA}" \
    --stats-csv "${DINO_STATS_CSV}" \
    "${SKIP_ARGS[@]}" \
    --ensure-all-images

  echo "[INFO] Generating DINO+SAM train labels."
  "${PYTHON_BIN}" tools/generate_yolo_labels_from_auto.py \
    --json-dir "${SAM_JSON_DIR}" \
    --image-dir "${IMAGE_DIR}" \
    --out-label-dir "${SAM_LABEL_DIR}" \
    --classes-config "${CLASSES_CONFIG}" \
    --bbox-key refined_bbox_xyxy \
    --fallback-bbox-key bbox_xyxy \
    --score-threshold "${SCORE_THRESHOLD}" \
    --min-box-area "${MIN_BOX_AREA}" \
    --stats-csv "${SAM_STATS_CSV}" \
    "${SKIP_ARGS[@]}" \
    --ensure-all-images
} 2>&1 | tee "${LOG_FILE}"
