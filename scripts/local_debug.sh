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
echo "[INFO] Local debug checks passed."
