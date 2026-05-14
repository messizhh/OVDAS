#!/usr/bin/env bash
set -e

PYTHON_BIN=python3

echo "[INFO] Using Python: ${PYTHON_BIN}"
echo "[INFO] Running project setup check..."
${PYTHON_BIN} tools/check_project_setup.py
echo "[INFO] Generating sample manifest..."
${PYTHON_BIN} tools/list_samples.py
echo "[INFO] Local debug checks passed."
