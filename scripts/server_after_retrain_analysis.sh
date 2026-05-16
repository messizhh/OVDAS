#!/usr/bin/env bash
set -e

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

HASH_CHECK_CSV="results/tables/yolo_weight_hash_check.csv"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Step 1/4: checking YOLO checkpoint tensor hashes"
bash scripts/server_check_yolo_weights.sh

if grep -q "IDENTICAL_WEIGHTS" "${HASH_CHECK_CSV}"; then
  echo "[WARNING] Identical YOLO checkpoint tensors detected."
  echo "[WARNING] Continue only for pipeline smoke checking; do not trust three-model comparison results."
fi

echo "[INFO] Step 2/4: running YOLO val and exporting prediction txt"
bash scripts/server_run_yolo_val_and_predictions.sh

echo "[INFO] Step 3/4: running small-object analysis"
bash scripts/server_analyze_small_objects.sh

echo "[INFO] Step 4/4: refreshing report figures"
python3 tools/make_report_figures.py \
  --table-dir results/tables \
  --output-dir figures/charts

if grep -q "IDENTICAL_WEIGHTS" "${HASH_CHECK_CSV}"; then
  echo "[WARNING] Final reminder: identical checkpoint tensors were detected."
  echo "[WARNING] Do not use refreshed three-model plots as final evidence until retraining produces distinct weights."
fi

echo "[INFO] After-retrain analysis pipeline finished."
