#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/server_prepare_ovdas_tile_yolo_dataset.sh [--rebuild] [--dry-run]

Builds the independent OVDAS-Tile downstream YOLO dataset using copied files.

Safe modes:
  --dry-run   Print resolved parameters and the prepare command without copying files.
  --rebuild   Rebuild a previously marked OVDAS-Tile output root.
  --help      Show this help.

Environment overrides:
  PROJECT_ROOT PYTHON_BIN OMP_NUM_THREADS LOG_DIR
  TRAIN_IMAGE_DIR TRAIN_LABEL_DIR VAL_IMAGE_DIR VAL_LABEL_DIR
  OUT_ROOT CONFIG_PATH CLASSES_CONFIG IMAGE_EXTS
  EXPECTED_TRAIN_IMAGES EXPECTED_TRAIN_LABELS EXPECTED_VAL_IMAGES EXPECTED_VAL_LABELS
USAGE
}

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
cd "${PROJECT_ROOT}"

MODE="run"
REBUILD="${REBUILD:-0}"
ORIGINAL_ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      ;;
    --rebuild)
      REBUILD="1"
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_NUM_THREADS
LOG_DIR="${LOG_DIR:-logs}"

TRAIN_IMAGE_DIR="${TRAIN_IMAGE_DIR:-data/processed/visdrone/images/train}"
TRAIN_LABEL_DIR="${TRAIN_LABEL_DIR:-outputs/ovdas_tile_full/train/labels}"
VAL_IMAGE_DIR="${VAL_IMAGE_DIR:-data/processed/visdrone/images/val}"
VAL_LABEL_DIR="${VAL_LABEL_DIR:-data/processed/visdrone/labels/val}"
OUT_ROOT="${OUT_ROOT:-data/processed/visdrone_ovdas_tile_yolo}"
CONFIG_PATH="${CONFIG_PATH:-configs/yolo_visdrone_ovdas_tile.yaml}"
CLASSES_CONFIG="${CLASSES_CONFIG:-configs/classes_visdrone.yaml}"
IMAGE_EXTS="${IMAGE_EXTS:-jpg,jpeg,png,bmp,tif,tiff}"
MARKER_NAME="${MARKER_NAME:-.ovdas_tile_yolo_dataset}"
MARKER_DATASET_ID="${MARKER_DATASET_ID:-ovdas_tile_yolo}"

EXPECTED_TRAIN_IMAGES="${EXPECTED_TRAIN_IMAGES:-6471}"
EXPECTED_TRAIN_LABELS="${EXPECTED_TRAIN_LABELS:-6471}"
EXPECTED_VAL_IMAGES="${EXPECTED_VAL_IMAGES:-548}"
EXPECTED_VAL_LABELS="${EXPECTED_VAL_LABELS:-548}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/server_prepare_ovdas_tile_yolo_dataset_${TIMESTAMP}.log"

validate_positive_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ${name} must be a positive integer, got '${value}'." >&2
    exit 2
  fi
}

validate_zero_or_one() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "[ERROR] ${name} must be 0 or 1, got '${value}'." >&2
    exit 2
  fi
}

print_command() {
  local -a command=("$@")
  printf '%q ' "${command[@]}"
  echo
}

prepare_command_array() {
  local -n command_ref="$1"
  command_ref=(
    "${PYTHON_BIN}" tools/prepare_auto_yolo_dataset.py
    --train-image-dir "${TRAIN_IMAGE_DIR}"
    --train-label-dir "${TRAIN_LABEL_DIR}"
    --val-image-dir "${VAL_IMAGE_DIR}"
    --val-label-dir "${VAL_LABEL_DIR}"
    --out-root "${OUT_ROOT}"
    --config-path "${CONFIG_PATH}"
    --classes-config "${CLASSES_CONFIG}"
    --link-mode copy
    --image-exts "${IMAGE_EXTS}"
    --expected-train-images "${EXPECTED_TRAIN_IMAGES}"
    --expected-train-labels "${EXPECTED_TRAIN_LABELS}"
    --expected-val-images "${EXPECTED_VAL_IMAGES}"
    --expected-val-labels "${EXPECTED_VAL_LABELS}"
    --strict-existing-labels
    --forbid-directory-symlinks
    --marker-name "${MARKER_NAME}"
    --marker-dataset-id "${MARKER_DATASET_ID}"
  )
  if [[ "${REBUILD}" == "1" ]]; then
    command_ref+=(--rebuild-output-root)
  fi
}

print_environment() {
  echo "[INFO] Command line: bash ${SCRIPT_PATH} $*"
  echo "[INFO] Script path: ${SCRIPT_PATH}"
  echo "[INFO] Project root: ${PROJECT_ROOT}"
  echo "[INFO] Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "[INFO] Git status --short:"
  git status --short 2>/dev/null || true
  echo "[INFO] Start time: ${START_TIME}"
  echo "[INFO] Python version: $("${PYTHON_BIN}" --version 2>&1 || echo unavailable)"
  echo "[INFO] OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
}

print_config() {
  echo "[INFO] TRAIN_IMAGE_DIR: ${TRAIN_IMAGE_DIR}"
  echo "[INFO] TRAIN_LABEL_DIR: ${TRAIN_LABEL_DIR}"
  echo "[INFO] VAL_IMAGE_DIR: ${VAL_IMAGE_DIR}"
  echo "[INFO] VAL_LABEL_DIR: ${VAL_LABEL_DIR}"
  echo "[INFO] OUT_ROOT: ${OUT_ROOT}"
  echo "[INFO] CONFIG_PATH: ${CONFIG_PATH}"
  echo "[INFO] CLASSES_CONFIG: ${CLASSES_CONFIG}"
  echo "[INFO] IMAGE_EXTS: ${IMAGE_EXTS}"
  echo "[INFO] MARKER_NAME: ${MARKER_NAME}"
  echo "[INFO] MARKER_DATASET_ID: ${MARKER_DATASET_ID}"
  echo "[INFO] EXPECTED_TRAIN_IMAGES: ${EXPECTED_TRAIN_IMAGES}"
  echo "[INFO] EXPECTED_TRAIN_LABELS: ${EXPECTED_TRAIN_LABELS}"
  echo "[INFO] EXPECTED_VAL_IMAGES: ${EXPECTED_VAL_IMAGES}"
  echo "[INFO] EXPECTED_VAL_LABELS: ${EXPECTED_VAL_LABELS}"
  echo "[INFO] REBUILD: ${REBUILD}"
  echo "[INFO] LOG_FILE: ${LOG_FILE}"
}

main_body() {
  START_TIME="$(date -Is)"
  validate_positive_integer "OMP_NUM_THREADS" "${OMP_NUM_THREADS}"
  validate_positive_integer "EXPECTED_TRAIN_IMAGES" "${EXPECTED_TRAIN_IMAGES}"
  validate_positive_integer "EXPECTED_TRAIN_LABELS" "${EXPECTED_TRAIN_LABELS}"
  validate_positive_integer "EXPECTED_VAL_IMAGES" "${EXPECTED_VAL_IMAGES}"
  validate_positive_integer "EXPECTED_VAL_LABELS" "${EXPECTED_VAL_LABELS}"
  validate_zero_or_one "REBUILD" "${REBUILD}"

  {
    print_environment "$@"
    print_config
    local -a command
    prepare_command_array command
    echo "[INFO] Prepare command:"
    print_command "${command[@]}"

    if [[ "${MODE}" == "dry-run" ]]; then
      echo "[DRY-RUN] No dataset files were copied."
      echo "[INFO] End time: $(date -Is)"
      echo "[INFO] Final status: completed"
      return 0
    fi

    "${command[@]}"
    echo "[INFO] End time: $(date -Is)"
    echo "[INFO] Final status: completed"
  }
}

main() {
  set +e
  ( set -euo pipefail; main_body "$@" ) 2>&1 | tee "${LOG_FILE}"
  local exit_code=${PIPESTATUS[0]}
  set -e
  if [[ "${exit_code}" -ne 0 ]]; then
    {
      echo "[INFO] End time: $(date -Is)"
      echo "[ERROR] Exit code: ${exit_code}"
      echo "[INFO] Final status: failed"
    } | tee -a "${LOG_FILE}"
  fi
  return "${exit_code}"
}

main "${ORIGINAL_ARGS[@]}"
