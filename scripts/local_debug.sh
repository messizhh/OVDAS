#!/usr/bin/env bash
set -e

PYTHON_BIN=python3

echo "[INFO] Using Python: ${PYTHON_BIN}"
echo "[INFO] Running project setup check..."
${PYTHON_BIN} tools/check_project_setup.py
echo "[INFO] Generating sample manifest..."
${PYTHON_BIN} tools/list_samples.py
echo "[INFO] Checking SAM refine syntax..."
${PYTHON_BIN} -m py_compile src/segmentation/sam_refine.py
${PYTHON_BIN} -m py_compile tools/run_sam_refine_single.py
${PYTHON_BIN} -m py_compile tools/run_sam_refine_batch.py
echo "[INFO] Checking SAM refine CLI help..."
${PYTHON_BIN} tools/run_sam_refine_single.py --help >/dev/null
${PYTHON_BIN} tools/run_sam_refine_batch.py --help >/dev/null
echo "[INFO] Checking auto-label generation syntax..."
${PYTHON_BIN} -m py_compile tools/generate_yolo_labels_from_auto.py
echo "[INFO] Checking auto-label generation CLI help..."
${PYTHON_BIN} tools/generate_yolo_labels_from_auto.py --help >/dev/null
echo "[INFO] Checking auto-label evaluation syntax..."
${PYTHON_BIN} -m py_compile tools/evaluate_auto_labels.py
echo "[INFO] Checking auto-label evaluation CLI help..."
${PYTHON_BIN} tools/evaluate_auto_labels.py --help >/dev/null
echo "[INFO] Checking YOLO Day 8 config and server scripts..."
test -f configs/yolo_visdrone_manual.yaml
test -f configs/yolo_visdrone_auto_sam_refine.yaml
test -f configs/yolo_visdrone_auto_dino.yaml
test -f scripts/server_prepare_visdrone_yolo_full.sh
test -f scripts/server_train_yolo_manual.sh
test -f scripts/server_train_yolo_auto_sam_refine.sh
test -f scripts/server_train_yolo_auto_dino.sh
test -f scripts/server_val_yolo_manual.sh
test -f scripts/server_val_yolo_auto_sam_refine.sh
test -f scripts/server_val_yolo_auto_dino.sh
echo "[INFO] Local debug checks passed."
