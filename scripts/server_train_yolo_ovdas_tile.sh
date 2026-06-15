#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/server_train_yolo_ovdas_tile.sh [--dry-run|--preflight-only|--resume]

Runs the locked OVDAS-Tile downstream YOLOv8s training job.

Safe modes:
  --dry-run        Print the final training command without checking files or training.
  --preflight-only Run checks, then exit before launching YOLO.
  --resume         Resume from runs/yolov8s_ovdas_tile_visdrone/weights/last.pt.
  --help          Show this help.

Environment overrides:
  PROJECT_ROOT PYTHON_BIN YOLO_BIN CUDA_VISIBLE_DEVICES OMP_NUM_THREADS LOG_DIR
USAGE
}

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
cd "${PROJECT_ROOT}"

MODE="run"
RESUME="${RESUME:-0}"
ORIGINAL_ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      ;;
    --preflight-only)
      MODE="preflight-only"
      ;;
    --resume)
      RESUME="1"
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
YOLO_BIN="${YOLO_BIN:-yolo}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_NUM_THREADS
LOG_DIR="${LOG_DIR:-logs}"

MODEL="${MODEL:-yolov8s.pt}"
DATA_CONFIG="${DATA_CONFIG:-configs/yolo_visdrone_ovdas_tile.yaml}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-1024}"
BATCH="${BATCH:-16}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
RUN_PROJECT="${RUN_PROJECT:-runs}"
RUN_NAME="${RUN_NAME:-yolov8s_ovdas_tile_visdrone}"
RUN_DIR="${RUN_PROJECT}/${RUN_NAME}"
RESUME_MODEL_ENV="${RESUME_MODEL:-}"
RESUME_MODEL="${RUN_DIR}/weights/last.pt"
DATASET_ROOT="data/processed/visdrone_ovdas_tile_yolo"
CLASSES_CONFIG="configs/classes_visdrone.yaml"
MARKER_NAME=".ovdas_tile_yolo_dataset"
MARKER_DATASET_ID="ovdas_tile_yolo"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/server_train_yolo_ovdas_tile_${TIMESTAMP}.log"

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

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing ${label}: ${path}" >&2
    exit 1
  fi
}

validate_locked_parameters() {
  [[ "${DATA_CONFIG}" == "configs/yolo_visdrone_ovdas_tile.yaml" ]] || { echo "[ERROR] Locked DATA_CONFIG changed." >&2; exit 2; }
  [[ "${EPOCHS}" == "100" ]] || { echo "[ERROR] Locked EPOCHS changed." >&2; exit 2; }
  [[ "${IMGSZ}" == "1024" ]] || { echo "[ERROR] Locked IMGSZ changed." >&2; exit 2; }
  [[ "${BATCH}" == "16" ]] || { echo "[ERROR] Locked BATCH changed." >&2; exit 2; }
  [[ "${SEED}" == "0" ]] || { echo "[ERROR] Locked SEED changed." >&2; exit 2; }
  [[ "${DEVICE}" == "0" ]] || { echo "[ERROR] Locked DEVICE changed." >&2; exit 2; }
  [[ "${WORKERS}" == "8" ]] || { echo "[ERROR] Locked WORKERS changed." >&2; exit 2; }
  [[ "${RUN_PROJECT}" == "runs" ]] || { echo "[ERROR] Locked RUN_PROJECT changed." >&2; exit 2; }
  [[ "${RUN_NAME}" == "yolov8s_ovdas_tile_visdrone" ]] || { echo "[ERROR] Locked RUN_NAME changed." >&2; exit 2; }
  if [[ -n "${RESUME_MODEL_ENV}" && "${RESUME_MODEL_ENV}" != "${RESUME_MODEL}" ]]; then
    echo "[ERROR] Locked RESUME_MODEL changed." >&2
    exit 2
  fi
  if [[ "${RESUME}" == "0" ]]; then
    [[ "${MODEL}" == "yolov8s.pt" ]] || { echo "[ERROR] Locked MODEL changed." >&2; exit 2; }
  fi
}

print_command() {
  local -a command=("$@")
  printf '%q ' "${command[@]}"
  echo
}

train_command_array() {
  local -n command_ref="$1"
  if [[ "${RESUME}" == "1" ]]; then
    command_ref=(
      "${YOLO_BIN}" detect train
      model="${RESUME_MODEL}"
      resume=True
      data="${DATA_CONFIG}"
      epochs="${EPOCHS}"
      imgsz="${IMGSZ}"
      batch="${BATCH}"
      seed="${SEED}"
      device="${DEVICE}"
      workers="${WORKERS}"
      project="${RUN_PROJECT}"
      name="${RUN_NAME}"
    )
    return
  fi

  command_ref=(
    "${YOLO_BIN}" detect train
    model="${MODEL}"
    data="${DATA_CONFIG}"
    epochs="${EPOCHS}"
    imgsz="${IMGSZ}"
    batch="${BATCH}"
    seed="${SEED}"
    device="${DEVICE}"
    workers="${WORKERS}"
    project="${RUN_PROJECT}"
    name="${RUN_NAME}"
  )
}

print_environment() {
  echo "[INFO] Command line: bash ${SCRIPT_PATH} $*"
  echo "[INFO] Script path: ${SCRIPT_PATH}"
  echo "[INFO] Project root: ${PROJECT_ROOT}"
  echo "[INFO] Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "[INFO] Git status --short:"
  git status --short 2>/dev/null || true
  echo "[INFO] Start time: ${START_TIME}"
  echo "[INFO] Hostname: $(hostname 2>/dev/null || echo unknown)"
  echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "[INFO] OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
  echo "[INFO] Python version: $("${PYTHON_BIN}" --version 2>&1 || echo unavailable)"
  "${PYTHON_BIN}" -c 'import importlib.util; spec=importlib.util.find_spec("torch"); print("[INFO] PyTorch:", "not installed" if spec is None else __import__("torch").__version__); print("[INFO] torch.cuda.is_available():", "unknown" if spec is None else __import__("torch").cuda.is_available()); print("[INFO] CUDA runtime:", "unknown" if spec is None else __import__("torch").version.cuda); print("[INFO] GPU name:", "unknown" if spec is None or not __import__("torch").cuda.is_available() else __import__("torch").cuda.get_device_name(0))' || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] nvidia-smi:"
    nvidia-smi || true
  else
    echo "[WARN] nvidia-smi not found."
  fi
}

print_config() {
  echo "[INFO] MODEL: ${MODEL}"
  echo "[INFO] DATA_CONFIG: ${DATA_CONFIG}"
  echo "[INFO] EPOCHS: ${EPOCHS}"
  echo "[INFO] IMGSZ: ${IMGSZ}"
  echo "[INFO] BATCH: ${BATCH}"
  echo "[INFO] SEED: ${SEED}"
  echo "[INFO] DEVICE: ${DEVICE}"
  echo "[INFO] WORKERS: ${WORKERS}"
  echo "[INFO] RUN_PROJECT: ${RUN_PROJECT}"
  echo "[INFO] RUN_NAME: ${RUN_NAME}"
  echo "[INFO] RUN_DIR: ${RUN_DIR}"
  echo "[INFO] RESUME: ${RESUME}"
  echo "[INFO] RESUME_MODEL: ${RESUME_MODEL}"
  echo "[INFO] DATASET_ROOT: ${DATASET_ROOT}"
  echo "[INFO] MARKER_NAME: ${MARKER_NAME}"
  echo "[INFO] MARKER_DATASET_ID: ${MARKER_DATASET_ID}"
  echo "[INFO] YOLO_BIN: ${YOLO_BIN}"
  echo "[INFO] LOG_FILE: ${LOG_FILE}"
}

validate_yolo_data_config() {
  "${PYTHON_BIN}" - "${DATA_CONFIG}" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
expected = {
    "path": "data/processed/visdrone_ovdas_tile_yolo",
    "train": "images/train",
    "val": "images/val",
    "names": {
        0: "pedestrian",
        1: "people",
        2: "bicycle",
        3: "car",
        4: "van",
        5: "truck",
        6: "bus",
        7: "motor",
    },
}
data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if data != expected:
    raise SystemExit(f"[ERROR] Locked YOLO data config changed: {config_path}")
print(f"[INFO] Locked YOLO data config OK: {config_path}")
PY
}

check_dataset_dirs_and_marker() {
  "${PYTHON_BIN}" - "${DATASET_ROOT}" "${MARKER_NAME}" "${MARKER_DATASET_ID}" <<'PY'
import sys
from pathlib import Path

dataset_root = Path(sys.argv[1])
marker_name = sys.argv[2]
expected_dataset_id = sys.argv[3]
required_dirs = [
    dataset_root,
    dataset_root / "images" / "train",
    dataset_root / "images" / "val",
    dataset_root / "labels" / "train",
    dataset_root / "labels" / "val",
]
for path in required_dirs:
    if path.is_symlink():
        raise SystemExit(f"[ERROR] Dataset directory must not be a symlink: {path}")
    if not path.is_dir():
        raise SystemExit(f"[ERROR] Missing dataset directory: {path}")
    print(f"[INFO] Dataset directory OK: {path}")

marker_path = dataset_root / marker_name
if not marker_path.is_file():
    raise SystemExit(f"[ERROR] Missing OVDAS-Tile dataset marker: {marker_path}")
values = {}
for line in marker_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    if "=" not in line:
        raise SystemExit(f"[ERROR] Invalid OVDAS-Tile dataset marker line: {line}")
    key, value = line.split("=", 1)
    values[key] = value
if values.get("dataset_id") != expected_dataset_id:
    raise SystemExit(
        f"[ERROR] OVDAS-Tile dataset marker dataset_id mismatch: "
        f"expected {expected_dataset_id}, got {values.get('dataset_id')}"
    )
if values.get("status") != "complete":
    raise SystemExit(
        f"[ERROR] OVDAS-Tile dataset marker must have status=complete, got status={values.get('status')}"
    )
print(f"[INFO] OVDAS-Tile dataset marker OK: {marker_path}")
PY
}

validate_dataset_contents() {
  echo "[INFO] Validating copied OVDAS-Tile YOLO dataset."
  "${PYTHON_BIN}" tools/prepare_auto_yolo_dataset.py \
    --train-image-dir "${DATASET_ROOT}/images/train" \
    --train-label-dir "${DATASET_ROOT}/labels/train" \
    --val-image-dir "${DATASET_ROOT}/images/val" \
    --val-label-dir "${DATASET_ROOT}/labels/val" \
    --out-root "${DATASET_ROOT}" \
    --config-path "${DATA_CONFIG}" \
    --classes-config "${CLASSES_CONFIG}" \
    --link-mode copy \
    --expected-train-images 6471 \
    --expected-train-labels 6471 \
    --expected-val-images 548 \
    --expected-val-labels 548 \
    --validate-only \
    --marker-name "${MARKER_NAME}" \
    --marker-dataset-id "${MARKER_DATASET_ID}"
}

validate_resume_state() {
  if [[ -L "${RUN_DIR}" ]]; then
    echo "[ERROR] Resume run directory must not be a symlink: ${RUN_DIR}" >&2
    exit 1
  fi
  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[ERROR] Missing resume run directory: ${RUN_DIR}" >&2
    exit 1
  fi
  if [[ -L "${RESUME_MODEL}" ]]; then
    echo "[ERROR] Resume checkpoint must not be a symlink: ${RESUME_MODEL}" >&2
    exit 1
  fi
  require_file "resume checkpoint" "${RESUME_MODEL}"
  require_file "resume args" "${RUN_DIR}/args.yaml"
  "${PYTHON_BIN}" - "${RUN_DIR}/args.yaml" <<'PY'
import sys
from pathlib import Path

import yaml

args_path = Path(sys.argv[1])
data = yaml.safe_load(args_path.read_text(encoding="utf-8"))
expected = {
    "data": "configs/yolo_visdrone_ovdas_tile.yaml",
    "epochs": 100,
    "imgsz": 1024,
    "batch": 16,
    "seed": 0,
    "project": "runs",
    "name": "yolov8s_ovdas_tile_visdrone",
}
for key, expected_value in expected.items():
    actual_value = data.get(key) if isinstance(data, dict) else None
    if actual_value != expected_value:
        raise SystemExit(
            f"[ERROR] Resume args mismatch for {key}: expected {expected_value!r}, got {actual_value!r}"
        )
print(f"[INFO] Resume args OK: {args_path}")
PY
}

preflight_checks() {
  validate_positive_integer "OMP_NUM_THREADS" "${OMP_NUM_THREADS}"
  validate_positive_integer "EPOCHS" "${EPOCHS}"
  validate_positive_integer "IMGSZ" "${IMGSZ}"
  validate_positive_integer "BATCH" "${BATCH}"
  validate_positive_integer "WORKERS" "${WORKERS}"
  validate_zero_or_one "RESUME" "${RESUME}"
  validate_locked_parameters

  require_file "YOLO data config" "${DATA_CONFIG}"
  validate_yolo_data_config
  check_dataset_dirs_and_marker
  if [[ "${RESUME}" == "1" ]]; then
    validate_resume_state
  else
    require_file "YOLO model" "${MODEL}"
    if [[ -e "${RUN_DIR}" ]]; then
      echo "[ERROR] Run directory already exists: ${RUN_DIR}. Use --resume only to continue an interrupted OVDAS-Tile run." >&2
      exit 1
    fi
  fi
  validate_dataset_contents
}

run_training() {
  local -a command
  train_command_array command
  echo "[INFO] Setting Ultralytics datasets_dir to project root."
  "${YOLO_BIN}" settings datasets_dir="$(pwd)"
  echo "[INFO] Final YOLO training command:"
  print_command "${command[@]}"
  "${command[@]}"
}

main_body() {
  START_TIME="$(date -Is)"
  validate_positive_integer "OMP_NUM_THREADS" "${OMP_NUM_THREADS}"
  validate_zero_or_one "RESUME" "${RESUME}"
  validate_locked_parameters

  {
    print_environment "$@"
    print_config
    local -a command
    train_command_array command
    echo "[INFO] Final YOLO training command:"
    print_command "${command[@]}"

    if [[ "${MODE}" == "dry-run" ]]; then
      if [[ "${RESUME}" == "1" ]]; then
        validate_resume_state
      fi
      echo "[DRY-RUN] No data checks or YOLO training were executed."
      echo "[INFO] End time: $(date -Is)"
      echo "[INFO] Final status: completed"
      return 0
    fi

    preflight_checks
    if [[ "${MODE}" == "preflight-only" ]]; then
      echo "[INFO] Preflight completed. No YOLO training was executed."
      echo "[INFO] End time: $(date -Is)"
      echo "[INFO] Final status: completed"
      return 0
    fi

    SECONDS=0
    run_training
    echo "[INFO] End time: $(date -Is)"
    echo "[INFO] Total elapsed seconds: ${SECONDS}"
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
