#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/server_run_ovdas_tile_full.sh [--dry-run|--preflight-only]

Runs the locked OVDAS-Tile E configuration on VisDrone train.

Safe modes:
  --dry-run        Print resolved parameters and commands without checking files or running models.
  --preflight-only Run environment/path/data checks, then exit before model inference.
  --help          Show this help.

Required server inputs can be overridden by environment variables:
  PROJECT_ROOT PYTHON_BIN CUDA_VISIBLE_DEVICES OMP_NUM_THREADS IMAGE_DIR
  CLASSES_CONFIG GROUNDING_DINO_CONFIG GROUNDING_DINO_CHECKPOINT SAM_CHECKPOINT
  OUTPUT_ROOT TABLE_DIR LOG_DIR
USAGE
}

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
cd "${PROJECT_ROOT}"

MODE="run"
if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --dry-run)
      MODE="dry-run"
      ;;
    --preflight-only)
      MODE="preflight-only"
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
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_NUM_THREADS

IMAGE_DIR="${IMAGE_DIR:-data/processed/visdrone/images/train}"
EXPECTED_IMAGE_COUNT="${EXPECTED_IMAGE_COUNT:-6471}"
ALLOW_IMAGE_COUNT_MISMATCH="${ALLOW_IMAGE_COUNT_MISMATCH:-0}"
CLASSES_CONFIG="${CLASSES_CONFIG:-configs/classes_visdrone.yaml}"
GROUNDING_DINO_CONFIG="${GROUNDING_DINO_CONFIG:-external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GROUNDING_DINO_CHECKPOINT="${GROUNDING_DINO_CHECKPOINT:-checkpoints/groundingdino_swint_ogc.pth}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_h_4b8939.pth}"
SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_h}"
DEVICE="${DEVICE:-cuda}"
LOG_DIR="${LOG_DIR:-logs}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ovdas_tile_full/train}"
GROUNDING_JSON_DIR="${GROUNDING_JSON_DIR:-${OUTPUT_ROOT}/grounding_json}"
SAM_JSON_DIR="${SAM_JSON_DIR:-${OUTPUT_ROOT}/sam_json}"
LABEL_DIR="${LABEL_DIR:-${OUTPUT_ROOT}/labels}"
VIS_DIR="${VIS_DIR:-results/visualizations/ovdas_tile_full/train}"
TABLE_DIR="${TABLE_DIR:-results/tables}"
LABEL_STATS_CSV="${LABEL_STATS_CSV:-${TABLE_DIR}/ovdas_tile_full_train_label_stats.csv}"

PROMPT="${PROMPT:-pedestrian. people. bicycle. car. van. truck. bus. motor.}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.25}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
TILE_SIZE="${TILE_SIZE:-640}"
OVERLAP_RATIO="${OVERLAP_RATIO:-0.20}"
MERGE_IOU="${MERGE_IOU:-0.50}"
MIN_REFINE_AREA_PX="${MIN_REFINE_AREA_PX:-1024}"
SMALL_AREA_RATIO="${SMALL_AREA_RATIO:-0.001}"
MEDIUM_AREA_RATIO="${MEDIUM_AREA_RATIO:-0.01}"
SMALL_SCORE_THRESHOLD="${SMALL_SCORE_THRESHOLD:-0.20}"
MEDIUM_SCORE_THRESHOLD="${MEDIUM_SCORE_THRESHOLD:-0.25}"
LARGE_SCORE_THRESHOLD="${LARGE_SCORE_THRESHOLD:-0.35}"
MIN_BOX_AREA="${MIN_BOX_AREA:-4}"

OFFLINE_MODE="${OFFLINE_MODE:-1}"
CHECK_HF_CACHE="${CHECK_HF_CACHE:-1}"
if [[ "${OFFLINE_MODE}" == "1" ]]; then
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}" "${TABLE_DIR}"
LOG_FILE="${LOG_DIR}/server_run_ovdas_tile_full_${TIMESTAMP}.log"

validate_positive_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ${name} must be a positive integer, got '${value}'." >&2
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
  echo "[INFO] EXPECTED_IMAGE_COUNT: ${EXPECTED_IMAGE_COUNT}"
  echo "[INFO] ALLOW_IMAGE_COUNT_MISMATCH: ${ALLOW_IMAGE_COUNT_MISMATCH}"
  echo "[INFO] CLASSES_CONFIG: ${CLASSES_CONFIG}"
  echo "[INFO] GROUNDING_DINO_CONFIG: ${GROUNDING_DINO_CONFIG}"
  echo "[INFO] GROUNDING_DINO_CHECKPOINT: ${GROUNDING_DINO_CHECKPOINT}"
  echo "[INFO] SAM_CHECKPOINT: ${SAM_CHECKPOINT}"
  echo "[INFO] SAM_MODEL_TYPE: ${SAM_MODEL_TYPE}"
  echo "[INFO] DEVICE: ${DEVICE}"
  echo "[INFO] OUTPUT_ROOT: ${OUTPUT_ROOT}"
  echo "[INFO] GROUNDING_JSON_DIR: ${GROUNDING_JSON_DIR}"
  echo "[INFO] SAM_JSON_DIR: ${SAM_JSON_DIR}"
  echo "[INFO] LABEL_DIR: ${LABEL_DIR}"
  echo "[INFO] VIS_DIR: ${VIS_DIR}"
  echo "[INFO] TABLE_DIR: ${TABLE_DIR}"
  echo "[INFO] LABEL_STATS_CSV: ${LABEL_STATS_CSV}"
  echo "[INFO] LOG_FILE: ${LOG_FILE}"
  echo "[INFO] OFFLINE_MODE: ${OFFLINE_MODE}"
  echo "[INFO] CHECK_HF_CACHE: ${CHECK_HF_CACHE}"
  echo "[INFO] Locked parameters:"
  echo "[INFO]   prompt=${PROMPT}"
  echo "[INFO]   box_threshold=${BOX_THRESHOLD}"
  echo "[INFO]   text_threshold=${TEXT_THRESHOLD}"
  echo "[INFO]   tile_size=${TILE_SIZE}"
  echo "[INFO]   overlap_ratio=${OVERLAP_RATIO}"
  echo "[INFO]   include_full_image=true"
  echo "[INFO]   merge_iou=${MERGE_IOU}"
  echo "[INFO]   min_refine_area_px=${MIN_REFINE_AREA_PX}"
  echo "[INFO]   small_area_ratio=${SMALL_AREA_RATIO}"
  echo "[INFO]   medium_area_ratio=${MEDIUM_AREA_RATIO}"
  echo "[INFO]   small_score_threshold=${SMALL_SCORE_THRESHOLD}"
  echo "[INFO]   medium_score_threshold=${MEDIUM_SCORE_THRESHOLD}"
  echo "[INFO]   large_score_threshold=${LARGE_SCORE_THRESHOLD}"
  echo "[INFO]   min_box_area=${MIN_BOX_AREA}"
}

validate_locked_parameters() {
  [[ "${PROMPT}" == "pedestrian. people. bicycle. car. van. truck. bus. motor." ]] || { echo "[ERROR] Locked PROMPT changed." >&2; exit 2; }
  [[ "${BOX_THRESHOLD}" == "0.25" ]] || { echo "[ERROR] Locked BOX_THRESHOLD changed." >&2; exit 2; }
  [[ "${TEXT_THRESHOLD}" == "0.25" ]] || { echo "[ERROR] Locked TEXT_THRESHOLD changed." >&2; exit 2; }
  [[ "${TILE_SIZE}" == "640" ]] || { echo "[ERROR] Locked TILE_SIZE changed." >&2; exit 2; }
  [[ "${OVERLAP_RATIO}" == "0.20" ]] || { echo "[ERROR] Locked OVERLAP_RATIO changed." >&2; exit 2; }
  [[ "${MERGE_IOU}" == "0.50" ]] || { echo "[ERROR] Locked MERGE_IOU changed." >&2; exit 2; }
  [[ "${MIN_REFINE_AREA_PX}" == "1024" ]] || { echo "[ERROR] Locked MIN_REFINE_AREA_PX changed." >&2; exit 2; }
  [[ "${SMALL_AREA_RATIO}" == "0.001" ]] || { echo "[ERROR] Locked SMALL_AREA_RATIO changed." >&2; exit 2; }
  [[ "${MEDIUM_AREA_RATIO}" == "0.01" ]] || { echo "[ERROR] Locked MEDIUM_AREA_RATIO changed." >&2; exit 2; }
  [[ "${SMALL_SCORE_THRESHOLD}" == "0.20" ]] || { echo "[ERROR] Locked SMALL_SCORE_THRESHOLD changed." >&2; exit 2; }
  [[ "${MEDIUM_SCORE_THRESHOLD}" == "0.25" ]] || { echo "[ERROR] Locked MEDIUM_SCORE_THRESHOLD changed." >&2; exit 2; }
  [[ "${LARGE_SCORE_THRESHOLD}" == "0.35" ]] || { echo "[ERROR] Locked LARGE_SCORE_THRESHOLD changed." >&2; exit 2; }
  [[ "${MIN_BOX_AREA}" == "4" ]] || { echo "[ERROR] Locked MIN_BOX_AREA changed." >&2; exit 2; }
}

preflight_checks() {
  validate_positive_integer "OMP_NUM_THREADS" "${OMP_NUM_THREADS}"
  validate_positive_integer "EXPECTED_IMAGE_COUNT" "${EXPECTED_IMAGE_COUNT}"
  validate_locked_parameters

  case "${OUTPUT_ROOT}" in
    outputs/tile_ablation_subset|outputs/tile_ablation_subset/*)
      echo "[ERROR] OUTPUT_ROOT must not overlap 200-image ablation outputs: ${OUTPUT_ROOT}" >&2
      exit 1
      ;;
  esac
  for path_name in GROUNDING_JSON_DIR SAM_JSON_DIR LABEL_DIR VIS_DIR; do
    local path_value="${!path_name}"
    case "${path_value}" in
      outputs/tile_ablation_subset|outputs/tile_ablation_subset/*|results/visualizations/tile_ablation_subset|results/visualizations/tile_ablation_subset/*)
        echo "[ERROR] ${path_name} must not overlap 200-image ablation outputs: ${path_value}" >&2
        exit 1
        ;;
    esac
  done

  require_dir "image directory" "${IMAGE_DIR}"
  local image_count
  image_count="$(count_images "${IMAGE_DIR}" | tr -d ' ')"
  echo "[INFO] Train image count: ${image_count}"
  if [[ "${image_count}" != "${EXPECTED_IMAGE_COUNT}" && "${ALLOW_IMAGE_COUNT_MISMATCH}" != "1" ]]; then
    echo "[ERROR] Expected ${EXPECTED_IMAGE_COUNT} train images, found ${image_count}. Set ALLOW_IMAGE_COUNT_MISMATCH=1 only for intentional debugging." >&2
    exit 1
  fi

  require_file "classes config" "${CLASSES_CONFIG}"
  require_file "Grounding DINO config" "${GROUNDING_DINO_CONFIG}"
  require_file "Grounding DINO checkpoint" "${GROUNDING_DINO_CHECKPOINT}"
  require_file "SAM checkpoint" "${SAM_CHECKPOINT}"
  require_file "tiled Grounding DINO entry" "tools/run_grounding_dino_tiled_batch.py"
  require_file "SAM refine entry" "tools/run_sam_refine_batch.py"
  require_file "auto-label generation entry" "tools/generate_yolo_labels_from_auto.py"

  if [[ "${OFFLINE_MODE}" == "1" && "${CHECK_HF_CACHE}" == "1" ]]; then
    echo "[INFO] Checking local Hugging Face cache for bert-base-uncased without downloads."
    "${PYTHON_BIN}" -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True); print("[INFO] bert-base-uncased tokenizer cache available.")'
  fi

  "${PYTHON_BIN}" -c 'import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)' || {
    echo "[ERROR] CUDA is not available to PyTorch. Full OVDAS-Tile train inference must run on the GPU server." >&2
    exit 1
  }
}

grounding_command() {
  printf '%q ' "${PYTHON_BIN}" tools/run_grounding_dino_tiled_batch.py \
    --image-dir "${IMAGE_DIR}" \
    --output-dir "${GROUNDING_JSON_DIR}" \
    --prompt "${PROMPT}" \
    --box-threshold "${BOX_THRESHOLD}" \
    --text-threshold "${TEXT_THRESHOLD}" \
    --tile-size "${TILE_SIZE}" \
    --overlap-ratio "${OVERLAP_RATIO}" \
    --include-full-image \
    --merge-iou "${MERGE_IOU}" \
    --device "${DEVICE}" \
    --config-file "${GROUNDING_DINO_CONFIG}" \
    --checkpoint "${GROUNDING_DINO_CHECKPOINT}" \
    --classes-config "${CLASSES_CONFIG}" \
    --skip-existing
}

sam_command() {
  printf '%q ' "${PYTHON_BIN}" tools/run_sam_refine_batch.py \
    --image-dir "${IMAGE_DIR}" \
    --dino-json-dir "${GROUNDING_JSON_DIR}" \
    --output-json-dir "${SAM_JSON_DIR}" \
    --vis-output-dir "${VIS_DIR}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --model-type "${SAM_MODEL_TYPE}" \
    --device "${DEVICE}" \
    --min-refine-area-px "${MIN_REFINE_AREA_PX}" \
    --skip-existing
}

label_command() {
  printf '%q ' "${PYTHON_BIN}" tools/generate_yolo_labels_from_auto.py \
    --json-dir "${SAM_JSON_DIR}" \
    --image-dir "${IMAGE_DIR}" \
    --out-label-dir "${LABEL_DIR}" \
    --classes-config "${CLASSES_CONFIG}" \
    --bbox-key refined_bbox_xyxy \
    --fallback-bbox-key bbox_xyxy \
    --score-threshold 0.0 \
    --min-box-area "${MIN_BOX_AREA}" \
    --enable-size-aware-filter \
    --small-area-ratio "${SMALL_AREA_RATIO}" \
    --medium-area-ratio "${MEDIUM_AREA_RATIO}" \
    --small-score-threshold "${SMALL_SCORE_THRESHOLD}" \
    --medium-score-threshold "${MEDIUM_SCORE_THRESHOLD}" \
    --large-score-threshold "${LARGE_SCORE_THRESHOLD}" \
    --stats-csv "${LABEL_STATS_CSV}" \
    --ensure-all-images \
    --skip-existing
}

print_commands() {
  echo "[DRY-RUN] Grounding DINO tiled command:"
  grounding_command
  echo
  echo "[DRY-RUN] Selective SAM command:"
  sam_command
  echo
  echo "[DRY-RUN] Size-aware YOLO label command:"
  label_command
  echo
}

run_grounding() {
  mkdir -p "${GROUNDING_JSON_DIR}"
  echo "[INFO] Stage 1/3: tiled Grounding DINO"
  "${PYTHON_BIN}" tools/run_grounding_dino_tiled_batch.py \
    --image-dir "${IMAGE_DIR}" \
    --output-dir "${GROUNDING_JSON_DIR}" \
    --prompt "${PROMPT}" \
    --box-threshold "${BOX_THRESHOLD}" \
    --text-threshold "${TEXT_THRESHOLD}" \
    --tile-size "${TILE_SIZE}" \
    --overlap-ratio "${OVERLAP_RATIO}" \
    --include-full-image \
    --merge-iou "${MERGE_IOU}" \
    --device "${DEVICE}" \
    --config-file "${GROUNDING_DINO_CONFIG}" \
    --checkpoint "${GROUNDING_DINO_CHECKPOINT}" \
    --classes-config "${CLASSES_CONFIG}" \
    --skip-existing
  summarize_json_dir "${GROUNDING_JSON_DIR}" "grounding"
}

run_sam() {
  local image_count
  image_count="$(count_images "${IMAGE_DIR}" | tr -d ' ')"
  local grounding_json_count
  grounding_json_count="$(count_files "${GROUNDING_JSON_DIR}" '*_grounding_dino.json' | tr -d ' ')"
  if [[ "${grounding_json_count}" != "${image_count}" ]]; then
    echo "[ERROR] Grounding JSON count ${grounding_json_count} does not match image count ${image_count}." >&2
    exit 1
  fi

  mkdir -p "${SAM_JSON_DIR}" "${VIS_DIR}"
  echo "[INFO] Stage 2/3: selective SAM refine"
  "${PYTHON_BIN}" tools/run_sam_refine_batch.py \
    --image-dir "${IMAGE_DIR}" \
    --dino-json-dir "${GROUNDING_JSON_DIR}" \
    --output-json-dir "${SAM_JSON_DIR}" \
    --vis-output-dir "${VIS_DIR}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --model-type "${SAM_MODEL_TYPE}" \
    --device "${DEVICE}" \
    --min-refine-area-px "${MIN_REFINE_AREA_PX}" \
    --skip-existing
  summarize_json_dir "${SAM_JSON_DIR}" "sam"
}

run_labels() {
  local image_count
  image_count="$(count_images "${IMAGE_DIR}" | tr -d ' ')"
  local sam_json_count
  sam_json_count="$(count_files "${SAM_JSON_DIR}" '*_sam_refine.json' | tr -d ' ')"
  if [[ "${sam_json_count}" != "${image_count}" ]]; then
    echo "[ERROR] SAM JSON count ${sam_json_count} does not match image count ${image_count}." >&2
    exit 1
  fi

  mkdir -p "${LABEL_DIR}" "${TABLE_DIR}"
  echo "[INFO] Stage 3/3: size-aware YOLO label generation"
  "${PYTHON_BIN}" tools/generate_yolo_labels_from_auto.py \
    --json-dir "${SAM_JSON_DIR}" \
    --image-dir "${IMAGE_DIR}" \
    --out-label-dir "${LABEL_DIR}" \
    --classes-config "${CLASSES_CONFIG}" \
    --bbox-key refined_bbox_xyxy \
    --fallback-bbox-key bbox_xyxy \
    --score-threshold 0.0 \
    --min-box-area "${MIN_BOX_AREA}" \
    --enable-size-aware-filter \
    --small-area-ratio "${SMALL_AREA_RATIO}" \
    --medium-area-ratio "${MEDIUM_AREA_RATIO}" \
    --small-score-threshold "${SMALL_SCORE_THRESHOLD}" \
    --medium-score-threshold "${MEDIUM_SCORE_THRESHOLD}" \
    --large-score-threshold "${LARGE_SCORE_THRESHOLD}" \
    --stats-csv "${LABEL_STATS_CSV}" \
    --ensure-all-images \
    --skip-existing
  summarize_labels
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
unknown_class = 0
invalid_boxes = 0
out_of_bounds = 0
sam_refined = 0
sam_skipped_small = 0
sam_fallback = 0

for path in json_files:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        invalid_json += 1
        continue
    width = float(data.get("image_width") or 0)
    height = float(data.get("image_height") or 0)
    for det in data.get("detections", []):
        detections += 1
        if det.get("class_id") is None and stage == "grounding":
            unknown_class += 1
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
        elif status == "skipped_small":
            sam_skipped_small += 1
        elif "fallback" in status or "error" in status:
            sam_fallback += 1

print(f"[INFO] {stage}_json_files: {len(json_files)}")
print(f"[INFO] {stage}_invalid_json_files: {invalid_json}")
print(f"[INFO] {stage}_detections: {detections}")
print(f"[INFO] {stage}_unknown_class: {unknown_class}")
print(f"[INFO] {stage}_invalid_boxes: {invalid_boxes}")
print(f"[INFO] {stage}_out_of_bounds_boxes: {out_of_bounds}")
if stage == "sam":
    print(f"[INFO] sam_refined: {sam_refined}")
    print(f"[INFO] sam_skipped_small: {sam_skipped_small}")
    print(f"[INFO] sam_fallback_or_failed: {sam_fallback}")

if invalid_json or invalid_boxes or out_of_bounds:
    raise SystemExit(1)
if stage == "grounding" and unknown_class:
    raise SystemExit(1)
PY
}

summarize_labels() {
  "${PYTHON_BIN}" - "${LABEL_DIR}" "${CLASSES_CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

label_dir = Path(sys.argv[1])
classes_config = Path(sys.argv[2])
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

print(f"[INFO] label_files: {len(label_files)}")
print(f"[INFO] empty_label_files: {empty_files}")
print(f"[INFO] total_label_boxes: {total_boxes}")
print(f"[INFO] invalid_label_lines: {invalid_lines}")
print(f"[INFO] out_of_range_class_ids: {out_of_range_class}")
print(f"[INFO] out_of_range_yolo_coords: {out_of_range_coords}")

if invalid_lines or out_of_range_class or out_of_range_coords:
    raise SystemExit(1)
PY
  if [[ -f "${LABEL_STATS_CSV}" ]]; then
    echo "[INFO] Label stats CSV: ${LABEL_STATS_CSV}"
    grep -E 'skipped_unknown_class|skipped_invalid_bbox|skipped_size_aware_low_score|kept_labels|empty_label_files' "${LABEL_STATS_CSV}" || true
  else
    echo "[ERROR] Missing label stats CSV: ${LABEL_STATS_CSV}" >&2
    exit 1
  fi
}

check_failure_logs() {
  local failures=0
  for file in \
    "${GROUNDING_JSON_DIR}/grounding_dino_tiled_batch_failures.txt" \
    "${SAM_JSON_DIR}/sam_refine_batch_failures.txt" \
    "${LABEL_DIR}/auto_label_generation_failures.txt"; do
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

  if [[ "${MODE}" == "dry-run" ]]; then
    validate_locked_parameters
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
    run_sam
    run_labels
    check_failure_logs
    echo "[INFO] Final grounding JSON count: $(count_files "${GROUNDING_JSON_DIR}" '*_grounding_dino.json' | tr -d ' ')"
    echo "[INFO] Final SAM JSON count: $(count_files "${SAM_JSON_DIR}" '*_sam_refine.json' | tr -d ' ')"
    echo "[INFO] Final YOLO label count: $(count_files "${LABEL_DIR}" '*.txt' | tr -d ' ')"
    echo "[INFO] End time: $(date -Is)"
    echo "[INFO] Total elapsed seconds: ${SECONDS}"
    echo "[INFO] Final status: completed"
  } 2>&1 | tee "${LOG_FILE}"
}

main "$@"
