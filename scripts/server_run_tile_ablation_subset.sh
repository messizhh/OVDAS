#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
IMAGE_DIR="${IMAGE_DIR:-data/processed/visdrone/images/train}"
SUBSET_LIST="${SUBSET_LIST:-outputs/experiment_subsets/visdrone_train_seed42_200.txt}"
CLASSES_CONFIG="${CLASSES_CONFIG:-configs/classes_visdrone.yaml}"
PROMPT="${PROMPT:-pedestrian. people. bicycle. car. van. truck. bus. motor.}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
DEVICE="${DEVICE:-cuda}"
DINO_CONFIG_FILE="${DINO_CONFIG_FILE:-external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-checkpoints/groundingdino_swint_ogc.pth}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_h_4b8939.pth}"
SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_h}"
TILE_SIZE="${TILE_SIZE:-640}"
OVERLAP_RATIO="${OVERLAP_RATIO:-0.20}"
MERGE_IOU="${MERGE_IOU:-0.50}"
MIN_REFINE_AREA_PX="${MIN_REFINE_AREA_PX:-1024}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tile_ablation_subset}"
VIS_ROOT="${VIS_ROOT:-results/visualizations/tile_ablation_subset}"
TABLE_DIR="${TABLE_DIR:-results/tables}"

mkdir -p logs "${TABLE_DIR}"
LOG_FILE="logs/server_run_tile_ablation_subset_$(date +%Y%m%d_%H%M%S).log"

print_environment() {
  echo "[INFO] Start time: $(date -Is)"
  echo "[INFO] Project root: ${PROJECT_ROOT}"
  echo "[INFO] Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "[INFO] Python: $("${PYTHON_BIN}" --version)"
  "${PYTHON_BIN}" -c 'import importlib.util; spec=importlib.util.find_spec("torch"); print("[INFO] PyTorch:", "not installed" if spec is None else __import__("torch").__version__); print("[INFO] CUDA available:", "unknown" if spec is None else __import__("torch").cuda.is_available()); print("[INFO] CUDA version:", "unknown" if spec is None else __import__("torch").version.cuda)'
  echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "[INFO] Image dir: ${IMAGE_DIR}"
  echo "[INFO] Subset list: ${SUBSET_LIST}"
  echo "[INFO] Output root: ${OUTPUT_ROOT}"
  echo "[INFO] Visualization root: ${VIS_ROOT}"
  echo "[INFO] Prompt: ${PROMPT}"
  echo "[INFO] Text threshold: ${TEXT_THRESHOLD}"
  echo "[INFO] Tile size: ${TILE_SIZE}"
  echo "[INFO] Overlap ratio: ${OVERLAP_RATIO}"
  echo "[INFO] Merge IoU: ${MERGE_IOU}"
  echo "[INFO] Min refine area px: ${MIN_REFINE_AREA_PX}"
  echo "[INFO] DINO config: ${DINO_CONFIG_FILE}"
  echo "[INFO] DINO checkpoint: ${DINO_CHECKPOINT}"
  echo "[INFO] SAM checkpoint: ${SAM_CHECKPOINT}"
  echo "[INFO] Log file: ${LOG_FILE}"
}

run_full_dino() {
  local method="$1"
  local box_threshold="$2"
  local output_dir="${OUTPUT_ROOT}/${method}/grounding_json"
  echo "[INFO] Running full-image DINO: method=${method}, box_threshold=${box_threshold}"
  "${PYTHON_BIN}" tools/run_grounding_dino_batch.py \
    --image-dir "${IMAGE_DIR}" \
    --image-list "${SUBSET_LIST}" \
    --output-dir "${output_dir}" \
    --prompt "${PROMPT}" \
    --box-threshold "${box_threshold}" \
    --text-threshold "${TEXT_THRESHOLD}" \
    --device "${DEVICE}" \
    --config-file "${DINO_CONFIG_FILE}" \
    --checkpoint "${DINO_CHECKPOINT}" \
    --skip-existing
}

run_tiled_dino() {
  local method="$1"
  local box_threshold="$2"
  local output_dir="${OUTPUT_ROOT}/${method}/grounding_json"
  echo "[INFO] Running tiled DINO: method=${method}, box_threshold=${box_threshold}"
  "${PYTHON_BIN}" tools/run_grounding_dino_tiled_batch.py \
    --image-dir "${IMAGE_DIR}" \
    --image-list "${SUBSET_LIST}" \
    --output-dir "${output_dir}" \
    --prompt "${PROMPT}" \
    --box-threshold "${box_threshold}" \
    --text-threshold "${TEXT_THRESHOLD}" \
    --tile-size "${TILE_SIZE}" \
    --overlap-ratio "${OVERLAP_RATIO}" \
    --include-full-image \
    --merge-iou "${MERGE_IOU}" \
    --device "${DEVICE}" \
    --config-file "${DINO_CONFIG_FILE}" \
    --checkpoint "${DINO_CHECKPOINT}" \
    --classes-config "${CLASSES_CONFIG}" \
    --skip-existing
}

run_sam_refine() {
  local method="$1"
  local dino_json_dir="$2"
  local min_area="$3"
  echo "[INFO] Running SAM refine: method=${method}, min_refine_area_px=${min_area}"
  "${PYTHON_BIN}" tools/run_sam_refine_batch.py \
    --image-dir "${IMAGE_DIR}" \
    --image-list "${SUBSET_LIST}" \
    --dino-json-dir "${dino_json_dir}" \
    --output-json-dir "${OUTPUT_ROOT}/${method}/sam_json" \
    --vis-output-dir "${VIS_ROOT}/${method}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --model-type "${SAM_MODEL_TYPE}" \
    --device "${DEVICE}" \
    --min-refine-area-px "${min_area}" \
    --skip-existing
}

generate_labels() {
  local method="$1"
  local json_dir="$2"
  local bbox_key="$3"
  local fallback_bbox_key="$4"
  local score_threshold="$5"
  shift 5
  echo "[INFO] Generating labels: method=${method}, score_threshold=${score_threshold}, bbox_key=${bbox_key}"
  "${PYTHON_BIN}" tools/generate_yolo_labels_from_auto.py \
    --json-dir "${json_dir}" \
    --image-dir "${IMAGE_DIR}" \
    --image-list "${SUBSET_LIST}" \
    --out-label-dir "${OUTPUT_ROOT}/${method}/labels" \
    --classes-config "${CLASSES_CONFIG}" \
    --bbox-key "${bbox_key}" \
    --fallback-bbox-key "${fallback_bbox_key}" \
    --score-threshold "${score_threshold}" \
    --min-box-area 4 \
    --stats-csv "${TABLE_DIR}/tile_ablation_subset_${method}_label_stats.csv" \
    --ensure-all-images \
    "$@"
}

{
  print_environment

  run_full_dino "dino_only_035" "0.35"
  generate_labels "dino_only_035" \
    "${OUTPUT_ROOT}/dino_only_035/grounding_json" \
    "bbox_xyxy" \
    "bbox_xyxy" \
    "0.35"

  run_sam_refine "dino_sam_035" "${OUTPUT_ROOT}/dino_only_035/grounding_json" "0"
  generate_labels "dino_sam_035" \
    "${OUTPUT_ROOT}/dino_sam_035/sam_json" \
    "refined_bbox_xyxy" \
    "bbox_xyxy" \
    "0.35"

  run_tiled_dino "dino_tile_035" "0.35"
  generate_labels "dino_tile_035" \
    "${OUTPUT_ROOT}/dino_tile_035/grounding_json" \
    "bbox_xyxy" \
    "bbox_xyxy" \
    "0.35"

  run_tiled_dino "dino_tile_025" "0.25"
  generate_labels "dino_tile_025" \
    "${OUTPUT_ROOT}/dino_tile_025/grounding_json" \
    "bbox_xyxy" \
    "bbox_xyxy" \
    "0.25"

  run_sam_refine "ovdas_tile" "${OUTPUT_ROOT}/dino_tile_025/grounding_json" "${MIN_REFINE_AREA_PX}"
  generate_labels "ovdas_tile" \
    "${OUTPUT_ROOT}/ovdas_tile/sam_json" \
    "refined_bbox_xyxy" \
    "bbox_xyxy" \
    "0.0" \
    --enable-size-aware-filter \
    --small-area-ratio 0.001 \
    --medium-area-ratio 0.01 \
    --small-score-threshold 0.20 \
    --medium-score-threshold 0.25 \
    --large-score-threshold 0.35

  echo "[INFO] End time: $(date -Is)"
} 2>&1 | tee "${LOG_FILE}"
