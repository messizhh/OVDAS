#!/usr/bin/env bash
set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

OUTPUT="${OUTPUT:-results/tables/yolo_weight_hash_check.csv}"

WEIGHTS=(
  "runs/yolov8s_manual_visdrone/weights/best.pt"
  "runs/yolov8s_auto_dino_only_visdrone/weights/best.pt"
  "runs/yolov8s_auto_dino_sam_visdrone/weights/best.pt"
)

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Output CSV: ${OUTPUT}"
echo "[INFO] Checking YOLO checkpoint tensor hashes..."

python3 tools/check_yolo_weight_hashes.py \
  --weights "${WEIGHTS[@]}" \
  --output "${OUTPUT}"

if grep -q "IDENTICAL_WEIGHTS" "${OUTPUT}"; then
  echo "[WARNING] At least two YOLO checkpoints have identical tensor parameters."
  echo "[WARNING] Do not trust downstream three-model comparisons until retraining is fixed."
fi
