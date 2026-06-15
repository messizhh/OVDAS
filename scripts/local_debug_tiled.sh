#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Python: $("${PYTHON_BIN}" --version)"
echo "[INFO] Checking OVDAS-Tile syntax and unit tests only; no DINO/SAM/YOLO heavy run."

"${PYTHON_BIN}" -m py_compile \
  src/open_vocab/tiled_inference.py \
  src/open_vocab/phrase_normalization.py \
  src/utils/image_lists.py \
  tools/run_grounding_dino_tiled_batch.py \
  tools/create_fixed_experiment_subset.py \
  tools/generate_yolo_labels_from_auto.py \
  tools/evaluate_auto_labels.py \
  tools/run_sam_refine_batch.py \
  tools/run_sam_refine_single.py

"${PYTHON_BIN}" tools/run_grounding_dino_tiled_batch.py --help >/dev/null
"${PYTHON_BIN}" tools/create_fixed_experiment_subset.py --help >/dev/null
"${PYTHON_BIN}" tools/generate_yolo_labels_from_auto.py --help >/dev/null
"${PYTHON_BIN}" tools/evaluate_auto_labels.py --help >/dev/null

"${PYTHON_BIN}" -m unittest discover -s tests -p "test_*.py"
