#!/usr/bin/env bash
set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

IMAGE_DIR="${IMAGE_DIR:-data/processed/visdrone/images/val}"
LABEL_DIR="${LABEL_DIR:-data/processed/visdrone/labels/val}"
CLASS_NAMES="${CLASS_NAMES:-configs/classes_visdrone.yaml}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.5}"
SIZE_MODE="${SIZE_MODE:-coco}"
MAX_VIS="${MAX_VIS:-50}"
SAVE_VISUALIZATIONS="${SAVE_VISUALIZATIONS:-1}"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] IMAGE_DIR: ${IMAGE_DIR}"
echo "[INFO] LABEL_DIR: ${LABEL_DIR}"
echo "[INFO] CLASS_NAMES: ${CLASS_NAMES}"
echo "[INFO] IOU_THRESHOLD: ${IOU_THRESHOLD}"
echo "[INFO] SIZE_MODE: ${SIZE_MODE}"
echo "[INFO] MAX_VIS: ${MAX_VIS}"
echo "[INFO] SAVE_VISUALIZATIONS: ${SAVE_VISUALIZATIONS}"

VIS_ARGS=()
if [[ "${SAVE_VISUALIZATIONS}" == "1" ]]; then
  VIS_ARGS+=(--save-visualizations)
fi

run_analysis() {
  local model_name="$1"
  local pred_dir="$2"
  local output_dir="$3"
  local vis_dir="$4"

  echo "[INFO] Analyzing ${model_name}"
  echo "[INFO] Prediction labels: ${pred_dir}"
  python3 tools/analyze_small_objects.py \
    --image-dir "${IMAGE_DIR}" \
    --label-dir "${LABEL_DIR}" \
    --pred-dir "${pred_dir}" \
    --class-names "${CLASS_NAMES}" \
    --output-dir "${output_dir}" \
    --vis-dir "${vis_dir}" \
    --model-name "${model_name}" \
    --iou-threshold "${IOU_THRESHOLD}" \
    --size-mode "${SIZE_MODE}" \
    --max-vis "${MAX_VIS}" \
    "${VIS_ARGS[@]}"
}

run_analysis \
  "manual" \
  "results/yolo_predictions/manual/labels" \
  "results/tables/small_object_analysis_manual" \
  "results/visualizations/small_object_failures/manual"

run_analysis \
  "dino_only" \
  "results/yolo_predictions/auto_dino_only/labels" \
  "results/tables/small_object_analysis_auto_dino_only" \
  "results/visualizations/small_object_failures/auto_dino_only"

run_analysis \
  "dino_sam" \
  "results/yolo_predictions/auto_dino_sam/labels" \
  "results/tables/small_object_analysis_auto_dino_sam" \
  "results/visualizations/small_object_failures/auto_dino_sam"

echo "[INFO] Small-object analysis finished."
