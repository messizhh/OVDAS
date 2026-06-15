#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/server_run_auto_label_baselines_val.sh [--dry-run] [--preflight-only] [--skip-existing]

Runs full-val DINO-only and DINO+SAM automatic-label baselines, then evaluates
both against manual VisDrone val labels. This script does not run OVDAS-Tile.

Safe modes:
  --dry-run        Print resolved parameters and commands without checking files or running models.
  --preflight-only Run environment/path/data checks, then exit before model inference.
  --skip-existing  Keep existing per-image stage outputs when supported by the tools.
  --help          Show this help.

Required server inputs can be overridden by environment variables:
  PROJECT_ROOT PYTHON_BIN CUDA_VISIBLE_DEVICES OMP_NUM_THREADS IMAGE_DIR GT_LABEL_DIR
  CLASSES_CONFIG GROUNDING_DINO_CONFIG GROUNDING_DINO_CHECKPOINT SAM_CHECKPOINT
  OUTPUT_ROOT TABLE_DIR LOG_DIR
USAGE
}

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
cd "${PROJECT_ROOT}"

MODE="run"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      ;;
    --preflight-only)
      MODE="preflight-only"
      ;;
    --skip-existing)
      SKIP_EXISTING="1"
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_NUM_THREADS

IMAGE_DIR="${IMAGE_DIR:-data/processed/visdrone/images/val}"
GT_LABEL_DIR="${GT_LABEL_DIR:-data/processed/visdrone/labels/val}"
EXPECTED_IMAGE_COUNT="${EXPECTED_IMAGE_COUNT:-548}"
EXPECTED_GT_LABEL_COUNT="${EXPECTED_GT_LABEL_COUNT:-548}"
ALLOW_IMAGE_COUNT_MISMATCH="${ALLOW_IMAGE_COUNT_MISMATCH:-0}"
CLASSES_CONFIG="${CLASSES_CONFIG:-configs/classes_visdrone.yaml}"
GROUNDING_DINO_CONFIG="${GROUNDING_DINO_CONFIG:-external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GROUNDING_DINO_CHECKPOINT="${GROUNDING_DINO_CHECKPOINT:-checkpoints/groundingdino_swint_ogc.pth}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_h_4b8939.pth}"
SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_h}"
DEVICE="${DEVICE:-cuda}"
LOG_DIR="${LOG_DIR:-logs}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/auto_label_baselines/val}"
GROUNDING_JSON_DIR="${GROUNDING_JSON_DIR:-${OUTPUT_ROOT}/grounding_json}"
SAM_JSON_DIR="${SAM_JSON_DIR:-${OUTPUT_ROOT}/sam_json}"
DINO_LABEL_DIR="${DINO_LABEL_DIR:-${OUTPUT_ROOT}/dino_only_labels}"
DINO_SAM_LABEL_DIR="${DINO_SAM_LABEL_DIR:-${OUTPUT_ROOT}/dino_sam_labels}"
VIS_DIR="${VIS_DIR:-results/visualizations/auto_label_baselines/val}"
TABLE_DIR="${TABLE_DIR:-results/tables}"

DINO_LABEL_STATS_CSV="${DINO_LABEL_STATS_CSV:-${TABLE_DIR}/dino_only_val_label_stats.csv}"
DINO_SAM_LABEL_STATS_CSV="${DINO_SAM_LABEL_STATS_CSV:-${TABLE_DIR}/dino_sam_val_label_stats.csv}"
DINO_SUMMARY_CSV="${DINO_SUMMARY_CSV:-${TABLE_DIR}/dino_only_val_summary.csv}"
DINO_CLASS_CSV="${DINO_CLASS_CSV:-${TABLE_DIR}/dino_only_val_by_class.csv}"
DINO_SIZE_CSV="${DINO_SIZE_CSV:-${TABLE_DIR}/dino_only_val_by_size.csv}"
DINO_SAM_SUMMARY_CSV="${DINO_SAM_SUMMARY_CSV:-${TABLE_DIR}/dino_sam_val_summary.csv}"
DINO_SAM_CLASS_CSV="${DINO_SAM_CLASS_CSV:-${TABLE_DIR}/dino_sam_val_by_class.csv}"
DINO_SAM_SIZE_CSV="${DINO_SAM_SIZE_CSV:-${TABLE_DIR}/dino_sam_val_by_size.csv}"
EFFICIENCY_CSV="${EFFICIENCY_CSV:-${TABLE_DIR}/auto_label_baselines_val_efficiency.csv}"

PROMPT="${PROMPT:-pedestrian. people. bicycle. car. van. truck. bus. motor.}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.35}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
MIN_REFINE_AREA_PX="${MIN_REFINE_AREA_PX:-0}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.35}"
MIN_BOX_AREA="${MIN_BOX_AREA:-4}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.50}"

OFFLINE_MODE="${OFFLINE_MODE:-1}"
CHECK_HF_CACHE="${CHECK_HF_CACHE:-1}"
if [[ "${OFFLINE_MODE}" == "1" ]]; then
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}" "${TABLE_DIR}"
LOG_FILE="${LOG_DIR}/server_run_auto_label_baselines_val_${TIMESTAMP}.log"

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

count_images() {
  local image_dir="$1"
  if [[ ! -d "${image_dir}" ]]; then
    echo 0
    return
  fi
  "${PYTHON_BIN}" - "${image_dir}" <<'PY'
import sys
from pathlib import Path

image_dir = Path(sys.argv[1])
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
image_count = sum(
    1
    for path in image_dir.iterdir()
    if path.is_file() and path.suffix.lower() in image_exts
)
print(image_count)
PY
}

count_files() {
  local dir="$1"
  local pattern="$2"
  if [[ ! -d "${dir}" ]]; then
    echo 0
    return
  fi
  find "${dir}" -maxdepth 1 -type f -name "${pattern}" | wc -l
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Missing ${label}: ${path}" >&2
    exit 1
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
  echo "[INFO] Disk free for project root:"
  df -h "${PROJECT_ROOT}" || true
}

print_config() {
  echo "[INFO] IMAGE_DIR: ${IMAGE_DIR}"
  echo "[INFO] GT_LABEL_DIR: ${GT_LABEL_DIR}"
  echo "[INFO] EXPECTED_IMAGE_COUNT: ${EXPECTED_IMAGE_COUNT}"
  echo "[INFO] EXPECTED_GT_LABEL_COUNT: ${EXPECTED_GT_LABEL_COUNT}"
  echo "[INFO] ALLOW_IMAGE_COUNT_MISMATCH: ${ALLOW_IMAGE_COUNT_MISMATCH}"
  echo "[INFO] CLASSES_CONFIG: ${CLASSES_CONFIG}"
  echo "[INFO] GROUNDING_DINO_CONFIG: ${GROUNDING_DINO_CONFIG}"
  echo "[INFO] GROUNDING_DINO_CHECKPOINT: ${GROUNDING_DINO_CHECKPOINT}"
  echo "[INFO] SAM_CHECKPOINT: ${SAM_CHECKPOINT}"
  echo "[INFO] SAM_MODEL_TYPE: ${SAM_MODEL_TYPE}"
  echo "[INFO] DEVICE: ${DEVICE}"
  echo "[INFO] SKIP_EXISTING: ${SKIP_EXISTING}"
  echo "[INFO] OUTPUT_ROOT: ${OUTPUT_ROOT}"
  echo "[INFO] GROUNDING_JSON_DIR: ${GROUNDING_JSON_DIR}"
  echo "[INFO] SAM_JSON_DIR: ${SAM_JSON_DIR}"
  echo "[INFO] DINO_LABEL_DIR: ${DINO_LABEL_DIR}"
  echo "[INFO] DINO_SAM_LABEL_DIR: ${DINO_SAM_LABEL_DIR}"
  echo "[INFO] VIS_DIR: ${VIS_DIR}"
  echo "[INFO] TABLE_DIR: ${TABLE_DIR}"
  echo "[INFO] DINO_LABEL_STATS_CSV: ${DINO_LABEL_STATS_CSV}"
  echo "[INFO] DINO_SAM_LABEL_STATS_CSV: ${DINO_SAM_LABEL_STATS_CSV}"
  echo "[INFO] DINO_SUMMARY_CSV: ${DINO_SUMMARY_CSV}"
  echo "[INFO] DINO_CLASS_CSV: ${DINO_CLASS_CSV}"
  echo "[INFO] DINO_SIZE_CSV: ${DINO_SIZE_CSV}"
  echo "[INFO] DINO_SAM_SUMMARY_CSV: ${DINO_SAM_SUMMARY_CSV}"
  echo "[INFO] DINO_SAM_CLASS_CSV: ${DINO_SAM_CLASS_CSV}"
  echo "[INFO] DINO_SAM_SIZE_CSV: ${DINO_SAM_SIZE_CSV}"
  echo "[INFO] EFFICIENCY_CSV: ${EFFICIENCY_CSV}"
  echo "[INFO] LOG_FILE: ${LOG_FILE}"
  echo "[INFO] OFFLINE_MODE: ${OFFLINE_MODE}"
  echo "[INFO] CHECK_HF_CACHE: ${CHECK_HF_CACHE}"
  echo "[INFO] Locked parameters:"
  echo "[INFO]   prompt=${PROMPT}"
  echo "[INFO]   box_threshold=${BOX_THRESHOLD}"
  echo "[INFO]   text_threshold=${TEXT_THRESHOLD}"
  echo "[INFO]   min_refine_area_px=${MIN_REFINE_AREA_PX}"
  echo "[INFO]   score_threshold=${SCORE_THRESHOLD}"
  echo "[INFO]   min_box_area=${MIN_BOX_AREA}"
  echo "[INFO]   evaluation_iou_threshold=${IOU_THRESHOLD}"
}

validate_locked_parameters() {
  [[ "${PROMPT}" == "pedestrian. people. bicycle. car. van. truck. bus. motor." ]] || { echo "[ERROR] Locked PROMPT changed." >&2; exit 2; }
  [[ "${BOX_THRESHOLD}" == "0.35" ]] || { echo "[ERROR] Locked BOX_THRESHOLD changed." >&2; exit 2; }
  [[ "${TEXT_THRESHOLD}" == "0.25" ]] || { echo "[ERROR] Locked TEXT_THRESHOLD changed." >&2; exit 2; }
  [[ "${MIN_REFINE_AREA_PX}" == "0" ]] || { echo "[ERROR] Locked MIN_REFINE_AREA_PX changed." >&2; exit 2; }
  [[ "${SCORE_THRESHOLD}" == "0.35" ]] || { echo "[ERROR] Locked SCORE_THRESHOLD changed." >&2; exit 2; }
  [[ "${MIN_BOX_AREA}" == "4" ]] || { echo "[ERROR] Locked MIN_BOX_AREA changed." >&2; exit 2; }
  [[ "${IOU_THRESHOLD}" == "0.50" ]] || { echo "[ERROR] Locked IOU_THRESHOLD changed." >&2; exit 2; }
}

validate_output_paths() {
  for path_name in OUTPUT_ROOT GROUNDING_JSON_DIR SAM_JSON_DIR DINO_LABEL_DIR DINO_SAM_LABEL_DIR VIS_DIR DINO_LABEL_STATS_CSV DINO_SAM_LABEL_STATS_CSV DINO_SUMMARY_CSV DINO_CLASS_CSV DINO_SIZE_CSV DINO_SAM_SUMMARY_CSV DINO_SAM_CLASS_CSV DINO_SAM_SIZE_CSV EFFICIENCY_CSV; do
    local path_value="${!path_name}"
    case "${path_value}" in
      train|train/*|*/train|*/train/*)
        echo "[ERROR] ${path_name} must not point to train outputs: ${path_value}" >&2
        exit 1
        ;;
      outputs/auto_label_baselines/train|outputs/auto_label_baselines/train/*|results/visualizations/auto_label_baselines/train|results/visualizations/auto_label_baselines/train/*)
        echo "[ERROR] ${path_name} must not point to train outputs: ${path_value}" >&2
        exit 1
        ;;
      outputs/ovdas_tile_full|outputs/ovdas_tile_full/*|results/visualizations/ovdas_tile_full|results/visualizations/ovdas_tile_full/*)
        echo "[ERROR] ${path_name} must not overlap OVDAS-Tile outputs: ${path_value}" >&2
        exit 1
        ;;
      *ovdas_tile*)
        echo "[ERROR] ${path_name} must not overlap OVDAS-Tile outputs: ${path_value}" >&2
        exit 1
        ;;
      *val_debug*)
        echo "[ERROR] ${path_name} must not overlap historical val_debug outputs: ${path_value}" >&2
        exit 1
        ;;
    esac
  done
}

preflight_checks() {
  validate_positive_integer "OMP_NUM_THREADS" "${OMP_NUM_THREADS}"
  validate_positive_integer "EXPECTED_IMAGE_COUNT" "${EXPECTED_IMAGE_COUNT}"
  validate_positive_integer "EXPECTED_GT_LABEL_COUNT" "${EXPECTED_GT_LABEL_COUNT}"
  validate_zero_or_one "ALLOW_IMAGE_COUNT_MISMATCH" "${ALLOW_IMAGE_COUNT_MISMATCH}"
  validate_zero_or_one "SKIP_EXISTING" "${SKIP_EXISTING}"
  validate_locked_parameters
  validate_output_paths

  require_dir "image directory" "${IMAGE_DIR}"
  local image_count
  image_count="$(count_images "${IMAGE_DIR}" | tr -d ' ')"
  echo "[INFO] Val image count: ${image_count}"
  if [[ "${image_count}" != "${EXPECTED_IMAGE_COUNT}" && "${ALLOW_IMAGE_COUNT_MISMATCH}" != "1" ]]; then
    echo "[ERROR] Expected ${EXPECTED_IMAGE_COUNT} val images, found ${image_count}. Set ALLOW_IMAGE_COUNT_MISMATCH=1 only for intentional debugging." >&2
    exit 1
  fi

  require_dir "GT label directory" "${GT_LABEL_DIR}"
  local gt_label_count
  gt_label_count="$(count_files "${GT_LABEL_DIR}" '*.txt' | tr -d ' ')"
  echo "[INFO] Val GT label count: ${gt_label_count}"
  if [[ "${gt_label_count}" != "${EXPECTED_GT_LABEL_COUNT}" ]]; then
    echo "[ERROR] Expected ${EXPECTED_GT_LABEL_COUNT} val GT label files, found ${gt_label_count}." >&2
    exit 1
  fi

  require_file "classes config" "${CLASSES_CONFIG}"
  require_file "Grounding DINO config" "${GROUNDING_DINO_CONFIG}"
  require_file "Grounding DINO checkpoint" "${GROUNDING_DINO_CHECKPOINT}"
  require_file "SAM checkpoint" "${SAM_CHECKPOINT}"
  require_file "Grounding DINO batch entry" "tools/run_grounding_dino_batch.py"
  require_file "SAM refine entry" "tools/run_sam_refine_batch.py"
  require_file "auto-label generation entry" "tools/generate_yolo_labels_from_auto.py"
  require_file "auto-label evaluation entry" "tools/evaluate_auto_labels.py"

  if [[ "${OFFLINE_MODE}" == "1" && "${CHECK_HF_CACHE}" == "1" ]]; then
    echo "[INFO] Checking local Hugging Face cache for bert-base-uncased without downloads."
    "${PYTHON_BIN}" -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True); print("[INFO] bert-base-uncased tokenizer cache available.")'
  fi

  "${PYTHON_BIN}" -c 'import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "[ERROR] CUDA is not available to PyTorch. Full val baselines must run on the GPU server." >&2
    exit 1
  }
}

append_skip_existing_arg() {
  local -n target_ref="$1"
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    target_ref+=(--skip-existing)
  fi
}

print_command() {
  local -a command=("$@")
  printf '%q ' "${command[@]}"
  echo
}

grounding_command_array() {
  local -n command_ref="$1"
  command_ref=(
    "${PYTHON_BIN}" tools/run_grounding_dino_batch.py
    --image-dir "${IMAGE_DIR}"
    --output-dir "${GROUNDING_JSON_DIR}"
    --prompt "${PROMPT}"
    --box-threshold "${BOX_THRESHOLD}"
    --text-threshold "${TEXT_THRESHOLD}"
    --device "${DEVICE}"
    --config-file "${GROUNDING_DINO_CONFIG}"
    --checkpoint "${GROUNDING_DINO_CHECKPOINT}"
  )
  append_skip_existing_arg "$1"
}

sam_command_array() {
  local -n command_ref="$1"
  command_ref=(
    "${PYTHON_BIN}" tools/run_sam_refine_batch.py
    --image-dir "${IMAGE_DIR}"
    --dino-json-dir "${GROUNDING_JSON_DIR}"
    --output-json-dir "${SAM_JSON_DIR}"
    --vis-output-dir "${VIS_DIR}"
    --sam-checkpoint "${SAM_CHECKPOINT}"
    --model-type "${SAM_MODEL_TYPE}"
    --device "${DEVICE}"
    --min-refine-area-px "${MIN_REFINE_AREA_PX}"
  )
  append_skip_existing_arg "$1"
}

dino_label_command_array() {
  local -n command_ref="$1"
  command_ref=(
    "${PYTHON_BIN}" tools/generate_yolo_labels_from_auto.py
    --json-dir "${GROUNDING_JSON_DIR}"
    --image-dir "${IMAGE_DIR}"
    --out-label-dir "${DINO_LABEL_DIR}"
    --classes-config "${CLASSES_CONFIG}"
    --bbox-key bbox_xyxy
    --fallback-bbox-key bbox_xyxy
    --score-threshold "${SCORE_THRESHOLD}"
    --min-box-area "${MIN_BOX_AREA}"
    --stats-csv "${DINO_LABEL_STATS_CSV}"
    --ensure-all-images
  )
  append_skip_existing_arg "$1"
}

sam_label_command_array() {
  local -n command_ref="$1"
  command_ref=(
    "${PYTHON_BIN}" tools/generate_yolo_labels_from_auto.py
    --json-dir "${SAM_JSON_DIR}"
    --image-dir "${IMAGE_DIR}"
    --out-label-dir "${DINO_SAM_LABEL_DIR}"
    --classes-config "${CLASSES_CONFIG}"
    --bbox-key refined_bbox_xyxy
    --fallback-bbox-key bbox_xyxy
    --score-threshold "${SCORE_THRESHOLD}"
    --min-box-area "${MIN_BOX_AREA}"
    --stats-csv "${DINO_SAM_LABEL_STATS_CSV}"
    --ensure-all-images
  )
  append_skip_existing_arg "$1"
}

dino_eval_command_array() {
  local -n command_ref="$1"
  command_ref=(
    "${PYTHON_BIN}" tools/evaluate_auto_labels.py
    --gt-label-dir "${GT_LABEL_DIR}"
    --pred-label-dir "${DINO_LABEL_DIR}"
    --image-dir "${IMAGE_DIR}"
    --metadata-json-dir "${GROUNDING_JSON_DIR}"
    --classes-config "${CLASSES_CONFIG}"
    --out-summary-csv "${DINO_SUMMARY_CSV}"
    --out-class-csv "${DINO_CLASS_CSV}"
    --out-size-csv "${DINO_SIZE_CSV}"
    --iou-threshold "${IOU_THRESHOLD}"
    --method-name DINO-only
  )
}

sam_eval_command_array() {
  local -n command_ref="$1"
  command_ref=(
    "${PYTHON_BIN}" tools/evaluate_auto_labels.py
    --gt-label-dir "${GT_LABEL_DIR}"
    --pred-label-dir "${DINO_SAM_LABEL_DIR}"
    --image-dir "${IMAGE_DIR}"
    --metadata-json-dir "${SAM_JSON_DIR}"
    --classes-config "${CLASSES_CONFIG}"
    --out-summary-csv "${DINO_SAM_SUMMARY_CSV}"
    --out-class-csv "${DINO_SAM_CLASS_CSV}"
    --out-size-csv "${DINO_SAM_SIZE_CSV}"
    --iou-threshold "${IOU_THRESHOLD}"
    --method-name DINO+SAM
  )
}

print_commands() {
  local -a command
  echo "[DRY-RUN] Full-image Grounding DINO command:"
  grounding_command_array command
  print_command "${command[@]}"
  echo "[DRY-RUN] DINO-only YOLO label command:"
  dino_label_command_array command
  print_command "${command[@]}"
  echo "[DRY-RUN] DINO-only evaluation command:"
  dino_eval_command_array command
  print_command "${command[@]}"
  echo "[DRY-RUN] Full SAM refine command:"
  sam_command_array command
  print_command "${command[@]}"
  echo "[DRY-RUN] DINO+SAM YOLO label command:"
  sam_label_command_array command
  print_command "${command[@]}"
  echo "[DRY-RUN] DINO+SAM evaluation command:"
  sam_eval_command_array command
  print_command "${command[@]}"
  echo "[DRY-RUN] Efficiency CSV output: ${EFFICIENCY_CSV}"
}

run_grounding() {
  mkdir -p "${GROUNDING_JSON_DIR}"
  echo "[INFO] Stage 1/6: full-image Grounding DINO on val"
  local -a command
  grounding_command_array command
  "${command[@]}"
  summarize_json_dir "${GROUNDING_JSON_DIR}" "grounding"
}

generate_dino_labels() {
  ensure_json_count "${GROUNDING_JSON_DIR}" '*_grounding_dino.json' "Grounding JSON"
  mkdir -p "${DINO_LABEL_DIR}" "${TABLE_DIR}"
  echo "[INFO] Stage 2/6: DINO-only YOLO label generation on val"
  local -a command
  dino_label_command_array command
  "${command[@]}"
  summarize_labels "${DINO_LABEL_DIR}" "${DINO_LABEL_STATS_CSV}" "dino_only"
}

evaluate_dino() {
  ensure_label_count "${DINO_LABEL_DIR}" "DINO-only label"
  echo "[INFO] Stage 3/6: DINO-only auto-label quality evaluation on val"
  local -a command
  dino_eval_command_array command
  "${command[@]}"
}

run_sam() {
  ensure_json_count "${GROUNDING_JSON_DIR}" '*_grounding_dino.json' "Grounding JSON"
  mkdir -p "${SAM_JSON_DIR}" "${VIS_DIR}"
  echo "[INFO] Stage 4/6: full SAM refine on val"
  local -a command
  sam_command_array command
  "${command[@]}"
  summarize_json_dir "${SAM_JSON_DIR}" "sam"
}

generate_sam_labels() {
  ensure_json_count "${SAM_JSON_DIR}" '*_sam_refine.json' "SAM JSON"
  mkdir -p "${DINO_SAM_LABEL_DIR}" "${TABLE_DIR}"
  echo "[INFO] Stage 5/6: DINO+SAM YOLO label generation on val"
  local -a command
  sam_label_command_array command
  "${command[@]}"
  summarize_labels "${DINO_SAM_LABEL_DIR}" "${DINO_SAM_LABEL_STATS_CSV}" "dino_sam"
}

evaluate_sam() {
  ensure_label_count "${DINO_SAM_LABEL_DIR}" "DINO+SAM label"
  echo "[INFO] Stage 6/6: DINO+SAM auto-label quality evaluation on val"
  local -a command
  sam_eval_command_array command
  "${command[@]}"
}

ensure_json_count() {
  local json_dir="$1"
  local pattern="$2"
  local label="$3"
  local image_count
  image_count="$(count_images "${IMAGE_DIR}" | tr -d ' ')"
  local json_count
  json_count="$(count_files "${json_dir}" "${pattern}" | tr -d ' ')"
  if [[ "${json_count}" != "${image_count}" ]]; then
    echo "[ERROR] ${label} count ${json_count} does not match image count ${image_count}." >&2
    exit 1
  fi
}

ensure_label_count() {
  local label_dir="$1"
  local label="$2"
  local image_count
  image_count="$(count_images "${IMAGE_DIR}" | tr -d ' ')"
  local label_count
  label_count="$(count_files "${label_dir}" '*.txt' | tr -d ' ')"
  if [[ "${label_count}" != "${image_count}" ]]; then
    echo "[ERROR] ${label} count ${label_count} does not match image count ${image_count}." >&2
    exit 1
  fi
}

summarize_json_dir() {
  local json_dir="$1"
  local stage="$2"
  "${PYTHON_BIN}" - "${json_dir}" "${stage}" <<'PY'
import json
import sys
from pathlib import Path

json_dir = Path(sys.argv[1])
stage = sys.argv[2]
json_files = sorted(json_dir.glob("*.json"))
invalid_json = 0
detections = 0
invalid_boxes = 0
out_of_bounds = 0
sam_refined = 0
sam_fallback = 0
inference_time_sum = 0.0
inference_time_images = 0

for path in json_files:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        invalid_json += 1
        continue
    for key in ("total_inference_time_sec", "inference_time_sec"):
        value = data.get(key)
        if isinstance(value, (float, int)):
            inference_time_sum += float(value)
            inference_time_images += 1
            break
    width = float(data.get("image_width") or 0)
    height = float(data.get("image_height") or 0)
    for det in data.get("detections", []):
        detections += 1
        bbox = det.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            invalid_boxes += 1
        else:
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox]
                if x2 <= x1 or y2 <= y1:
                    invalid_boxes += 1
                if width > 0 and height > 0 and (x1 < 0 or y1 < 0 or x2 > width or y2 > height):
                    out_of_bounds += 1
            except Exception:
                invalid_boxes += 1
        status = str(det.get("refine_status", ""))
        if status == "refined":
            sam_refined += 1
        elif "fallback" in status or "error" in status:
            sam_fallback += 1

print(f"[INFO] {stage}_json_files: {len(json_files)}")
print(f"[INFO] {stage}_invalid_json_files: {invalid_json}")
print(f"[INFO] {stage}_detections: {detections}")
print(f"[INFO] {stage}_invalid_boxes: {invalid_boxes}")
print(f"[INFO] {stage}_out_of_bounds_boxes: {out_of_bounds}")
if inference_time_images:
    print(f"[INFO] {stage}_average_inference_time_sec: {inference_time_sum / inference_time_images:.6f}")
if stage == "sam":
    print(f"[INFO] sam_refined: {sam_refined}")
    print(f"[INFO] sam_fallback_or_failed: {sam_fallback}")

if invalid_json or invalid_boxes or out_of_bounds:
    raise SystemExit(1)
PY
}

summarize_labels() {
  local label_dir="$1"
  local stats_csv="$2"
  local method="$3"
  "${PYTHON_BIN}" - "${label_dir}" "${CLASSES_CONFIG}" "${method}" <<'PY'
import sys
from pathlib import Path
import yaml

label_dir = Path(sys.argv[1])
classes_config = Path(sys.argv[2])
method = sys.argv[3]
data = yaml.safe_load(classes_config.read_text(encoding="utf-8"))
class_ids = {
    int(item.get("id", index)) if isinstance(item, dict) else index
    for index, item in enumerate(data.get("classes", []))
}

label_files = sorted(label_dir.glob("*.txt"))
empty_files = 0
total_boxes = 0
invalid_lines = 0
out_of_range_coords = 0
out_of_range_class = 0

for path in label_files:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        empty_files += 1
        continue
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            invalid_lines += 1
            continue
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            invalid_lines += 1
            continue
        if class_id not in class_ids:
            out_of_range_class += 1
        if any(value < 0.0 or value > 1.0 for value in coords):
            out_of_range_coords += 1
        total_boxes += 1

print(f"[INFO] {method}_label_files: {len(label_files)}")
print(f"[INFO] {method}_empty_label_files: {empty_files}")
print(f"[INFO] {method}_prediction_boxes: {total_boxes}")
print(f"[INFO] {method}_invalid_label_lines: {invalid_lines}")
print(f"[INFO] {method}_out_of_range_class_ids: {out_of_range_class}")
print(f"[INFO] {method}_out_of_bounds_yolo_coords: {out_of_range_coords}")

if invalid_lines or out_of_range_class or out_of_range_coords:
    raise SystemExit(1)
PY
  if [[ -f "${stats_csv}" ]]; then
    echo "[INFO] Label stats CSV: ${stats_csv}"
    grep -E 'skipped_unknown_class|skipped_invalid_bbox|kept_labels|empty_label_files' "${stats_csv}" || true
  else
    echo "[ERROR] Missing label stats CSV: ${stats_csv}" >&2
    exit 1
  fi
}

write_efficiency_csv() {
  "${PYTHON_BIN}" - \
    "${IMAGE_DIR}" \
    "${GT_LABEL_DIR}" \
    "${GROUNDING_JSON_DIR}" \
    "${SAM_JSON_DIR}" \
    "${DINO_LABEL_DIR}" \
    "${DINO_SAM_LABEL_DIR}" \
    "${EFFICIENCY_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

image_dir = Path(sys.argv[1])
gt_label_dir = Path(sys.argv[2])
grounding_json_dir = Path(sys.argv[3])
sam_json_dir = Path(sys.argv[4])
dino_label_dir = Path(sys.argv[5])
sam_label_dir = Path(sys.argv[6])
output_csv = Path(sys.argv[7])

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
image_count = sum(1 for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in image_exts)
gt_label_count = sum(1 for path in gt_label_dir.iterdir() if path.is_file() and path.suffix == ".txt")

def json_stats(json_dir: Path, kind: str) -> dict[str, object]:
    files = sorted(json_dir.glob("*.json"))
    detections = 0
    refined = 0
    fallback = 0
    time_sum = 0.0
    time_images = 0
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("total_inference_time_sec", "inference_time_sec"):
            value = data.get(key)
            if isinstance(value, (float, int)):
                time_sum += float(value)
                time_images += 1
                break
        for detection in data.get("detections", []):
            if not isinstance(detection, dict):
                continue
            detections += 1
            status = str(detection.get("refine_status", ""))
            if status == "refined":
                refined += 1
            elif "fallback" in status or "error" in status:
                fallback += 1
    return {
        f"{kind}_json_files": len(files),
        f"{kind}_detections": detections,
        f"{kind}_time_images": time_images,
        f"{kind}_average_time_sec": f"{time_sum / time_images:.6f}" if time_images else "",
        f"{kind}_sam_refined": refined,
        f"{kind}_sam_fallback": fallback,
    }

def label_count(label_dir: Path) -> tuple[int, int]:
    files = sorted(label_dir.glob("*.txt"))
    boxes = 0
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            boxes += len(text.splitlines())
    return len(files), boxes

dino_label_files, dino_boxes = label_count(dino_label_dir)
sam_label_files, sam_boxes = label_count(sam_label_dir)
row = {
    "image_count": image_count,
    "gt_label_files": gt_label_count,
    **json_stats(grounding_json_dir, "dino"),
    "dino_label_files": dino_label_files,
    "dino_prediction_boxes": dino_boxes,
    **json_stats(sam_json_dir, "dino_sam"),
    "dino_sam_label_files": sam_label_files,
    "dino_sam_prediction_boxes": sam_boxes,
}
output_csv.parent.mkdir(parents=True, exist_ok=True)
with output_csv.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(row.keys()))
    writer.writeheader()
    writer.writerow(row)
print(f"[INFO] Efficiency CSV: {output_csv.as_posix()}")
PY
}

check_failure_logs() {
  local failures=0
  local dino_eval_failure_log="${DINO_SUMMARY_CSV%.*}_failures.txt"
  local sam_eval_failure_log="${DINO_SAM_SUMMARY_CSV%.*}_failures.txt"
  for file in \
    "${GROUNDING_JSON_DIR}/grounding_dino_batch_failures.txt" \
    "${SAM_JSON_DIR}/sam_refine_batch_failures.txt" \
    "${DINO_LABEL_DIR}/auto_label_generation_failures.txt" \
    "${DINO_SAM_LABEL_DIR}/auto_label_generation_failures.txt" \
    "${dino_eval_failure_log}" \
    "${sam_eval_failure_log}"; do
    if [[ -s "${file}" ]]; then
      echo "[ERROR] Failure log is not empty: ${file}" >&2
      cat "${file}" >&2
      failures=1
    fi
  done
  if [[ "${failures}" == "1" ]]; then
    exit 1
  fi
}

main() {
  START_TIME="$(date -Is)"
  validate_positive_integer "OMP_NUM_THREADS" "${OMP_NUM_THREADS}"
  validate_zero_or_one "SKIP_EXISTING" "${SKIP_EXISTING}"

  if [[ "${MODE}" == "dry-run" ]]; then
    validate_locked_parameters
    validate_output_paths
    print_config
    print_commands
    echo "[DRY-RUN] No preflight checks or model inference were executed."
    return 0
  fi

  {
    print_environment "$@"
    print_config
    preflight_checks
    if [[ "${MODE}" == "preflight-only" ]]; then
      echo "[INFO] Preflight completed. No model inference was executed."
      echo "[INFO] End time: $(date -Is)"
      return 0
    fi

    SECONDS=0
    run_grounding
    generate_dino_labels
    evaluate_dino
    run_sam
    generate_sam_labels
    evaluate_sam
    write_efficiency_csv
    check_failure_logs
    echo "[INFO] Final grounding JSON count: $(count_files "${GROUNDING_JSON_DIR}" '*_grounding_dino.json' | tr -d ' ')"
    echo "[INFO] Final SAM JSON count: $(count_files "${SAM_JSON_DIR}" '*_sam_refine.json' | tr -d ' ')"
    echo "[INFO] Final DINO-only label count: $(count_files "${DINO_LABEL_DIR}" '*.txt' | tr -d ' ')"
    echo "[INFO] Final DINO+SAM label count: $(count_files "${DINO_SAM_LABEL_DIR}" '*.txt' | tr -d ' ')"
    echo "[INFO] DINO-only summary CSV: ${DINO_SUMMARY_CSV}"
    echo "[INFO] DINO+SAM summary CSV: ${DINO_SAM_SUMMARY_CSV}"
    echo "[INFO] Efficiency CSV: ${EFFICIENCY_CSV}"
    echo "[INFO] End time: $(date -Is)"
    echo "[INFO] Total elapsed seconds: ${SECONDS}"
    echo "[INFO] Final status: completed"
  } 2>&1 | tee "${LOG_FILE}"
}

main "$@"
