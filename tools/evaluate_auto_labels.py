"""Evaluate automatic YOLO labels against manual YOLO labels."""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.image_lists import read_image_list, resolve_image_entries


DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SIZE_BUCKETS = ("small", "medium", "large")
SMALL_AREA = 32 * 32
MEDIUM_AREA = 96 * 96


@dataclass
class Box:
    """One pixel-space detection or ground-truth box."""

    class_id: int
    xyxy: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        """Return box area in pixels."""
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class MetricStats:
    """Detection matching statistics."""

    gt_count: int = 0
    pred_count: int = 0
    matched_count: int = 0
    iou_sum: float = 0.0

    @property
    def false_positive_count(self) -> int:
        """Return unmatched prediction count."""
        return max(0, self.pred_count - self.matched_count)

    @property
    def false_negative_count(self) -> int:
        """Return unmatched ground-truth count."""
        return max(0, self.gt_count - self.matched_count)

    @property
    def precision(self) -> float:
        """Return precision with zero-safe division."""
        if self.pred_count == 0:
            return 0.0
        return self.matched_count / self.pred_count

    @property
    def recall(self) -> float:
        """Return recall with zero-safe division."""
        if self.gt_count == 0:
            return 0.0
        return self.matched_count / self.gt_count

    @property
    def mean_iou(self) -> float:
        """Return mean IoU over matched pairs."""
        if self.matched_count == 0:
            return 0.0
        return self.iou_sum / self.matched_count

    @property
    def f1(self) -> float:
        """Return F1 score with zero-safe division."""
        denominator = self.precision + self.recall
        if denominator == 0.0:
            return 0.0
        return 2.0 * self.precision * self.recall / denominator


@dataclass
class EvalSummary:
    """Evaluation-level missing/failure counters."""

    total_label_files: int = 0
    evaluated_images: int = 0
    failed_images: int = 0
    missing_images: int = 0
    missing_gt_label_files: int = 0
    missing_pred_label_files: int = 0
    invalid_gt_label_lines: int = 0
    invalid_pred_label_lines: int = 0
    inference_time_sum: float = 0.0
    inference_time_images: int = 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate automatic YOLO labels against manual YOLO labels."
    )
    parser.add_argument("--gt-label-dir", required=True, help="Manual YOLO label directory.")
    parser.add_argument("--pred-label-dir", required=True, help="Automatic YOLO label directory.")
    parser.add_argument("--image-dir", required=True, help="Image directory for size lookup.")
    parser.add_argument(
        "--image-list",
        default=None,
        help="Optional txt/json image list. When set, evaluate exactly these images.",
    )
    parser.add_argument(
        "--metadata-json-dir",
        default=None,
        help="Optional auto JSON directory used to compute average inference time.",
    )
    parser.add_argument(
        "--method-name",
        default=None,
        help="Optional method name column for ablation CSV aggregation.",
    )
    parser.add_argument(
        "--append-csv",
        action="store_true",
        help="Append rows to existing CSV files instead of overwriting them.",
    )
    parser.add_argument(
        "--classes-config",
        default="configs/classes_visdrone.yaml",
        help="Class config YAML with id/name entries.",
    )
    parser.add_argument("--out-summary-csv", required=True, help="Output overall summary CSV.")
    parser.add_argument("--out-class-csv", required=True, help="Output per-class metrics CSV.")
    parser.add_argument("--out-size-csv", required=True, help="Output size-bucket metrics CSV.")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for greedy one-to-one matching.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of prediction label files to evaluate after sorting.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line values."""
    if args.iou_threshold < 0.0 or args.iou_threshold > 1.0:
        raise ValueError("--iou-threshold must be between 0 and 1.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")


def load_class_names(classes_config: Path) -> dict[int, str]:
    """Load class id to class name mapping."""
    with classes_config.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {classes_config}")

    classes = data.get("classes", [])
    if not isinstance(classes, list):
        raise ValueError(f"'classes' must be a list in {classes_config}")

    names: dict[int, str] = {}
    for index, item in enumerate(classes):
        if isinstance(item, str):
            names[index] = item
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            class_id = item.get("id", index)
            if not isinstance(class_id, int):
                raise ValueError(f"Invalid class id in {classes_config}: {class_id}")
            names[class_id] = item["name"]
        else:
            raise ValueError(f"Invalid class entry in {classes_config}: {item}")

    if not names:
        raise ValueError(f"No classes found in {classes_config}")
    return names


def collect_pred_label_files(pred_label_dir: Path, limit: int | None) -> list[Path]:
    """Collect prediction txt files in deterministic order."""
    if not pred_label_dir.is_dir():
        raise FileNotFoundError(f"Prediction label directory does not exist: {pred_label_dir}")
    label_files = sorted(
        path
        for path in pred_label_dir.iterdir()
        if path.is_file()
        and path.suffix == ".txt"
        and not path.name.endswith("_failures.txt")
    )
    if limit is not None:
        return label_files[:limit]
    return label_files


def collect_pred_label_files_for_images(
    pred_label_dir: Path,
    image_paths: list[Path],
    limit: int | None,
) -> list[Path]:
    """Build prediction label paths for a fixed image subset."""
    label_files = [pred_label_dir / f"{image_path.stem}.txt" for image_path in image_paths]
    if limit is not None:
        return label_files[:limit]
    return label_files


def resolve_image_path(image_dir: Path, stem: str) -> Path | None:
    """Find one image path by label stem."""
    for ext in DEFAULT_IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def read_image_size(image_path: Path) -> tuple[int, int]:
    """Read image width and height, with a header-only fallback."""
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.size
    except ModuleNotFoundError:
        return read_image_size_from_header(image_path)


def read_image_size_from_header(image_path: Path) -> tuple[int, int]:
    """Read common image dimensions using only file headers."""
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return read_png_size(image_path)
    if suffix == ".bmp":
        return read_bmp_size(image_path)
    if suffix in {".jpg", ".jpeg"}:
        return read_jpeg_size(image_path)
    raise ValueError(f"Unsupported image extension without Pillow: {image_path.suffix}")


def read_png_size(image_path: Path) -> tuple[int, int]:
    """Read PNG width and height from IHDR."""
    with image_path.open("rb") as file:
        header = file.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG header")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def read_bmp_size(image_path: Path) -> tuple[int, int]:
    """Read BMP width and height from the DIB header."""
    with image_path.open("rb") as file:
        header = file.read(26)
    if len(header) < 26 or header[:2] != b"BM":
        raise ValueError("Invalid BMP header")
    width, height = struct.unpack("<ii", header[18:26])
    return width, abs(height)


def read_jpeg_size(image_path: Path) -> tuple[int, int]:
    """Read JPEG width and height from a start-of-frame marker."""
    with image_path.open("rb") as file:
        if file.read(2) != b"\xff\xd8":
            raise ValueError("Invalid JPEG header")

        while True:
            marker_start = file.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue

            marker = file.read(1)
            while marker == b"\xff":
                marker = file.read(1)
            if not marker:
                break

            marker_value = marker[0]
            if marker_value in {0xD8, 0xD9}:
                continue

            length_bytes = file.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", length_bytes)[0]
            if segment_length < 2:
                raise ValueError("Invalid JPEG segment length")

            if marker_value in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                data = file.read(5)
                if len(data) != 5:
                    break
                height, width = struct.unpack(">HH", data[1:5])
                return width, height

            file.seek(segment_length - 2, 1)

    raise ValueError("Could not find JPEG size marker")


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Convert normalized YOLO xywh to clipped pixel xyxy."""
    x1 = (x_center - width / 2.0) * image_width
    y1 = (y_center - height / 2.0) * image_height
    x2 = (x_center + width / 2.0) * image_width
    y2 = (y_center + height / 2.0) * image_height

    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def parse_yolo_label_file(
    label_path: Path,
    image_width: int,
    image_height: int,
    valid_class_ids: set[int],
) -> tuple[list[Box], int, bool]:
    """Parse one YOLO label file into pixel-space boxes."""
    if not label_path.is_file():
        return [], 0, False

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return [], 0, True

    boxes: list[Box] = []
    invalid_lines = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            invalid_lines += 1
            print(f"[WARN] Invalid label line ignored: {label_path}:{line_number}")
            continue
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            invalid_lines += 1
            print(f"[WARN] Invalid label values ignored: {label_path}:{line_number}")
            continue

        if class_id not in valid_class_ids or width <= 0.0 or height <= 0.0:
            invalid_lines += 1
            print(f"[WARN] Out-of-range label ignored: {label_path}:{line_number}")
            continue

        xyxy = yolo_to_xyxy(x_center, y_center, width, height, image_width, image_height)
        if xyxy is None:
            invalid_lines += 1
            print(f"[WARN] Degenerate label ignored: {label_path}:{line_number}")
            continue

        boxes.append(Box(class_id=class_id, xyxy=xyxy))

    return boxes, invalid_lines, True


def bbox_iou(a: Box, b: Box) -> float:
    """Compute IoU for two pixel-space boxes."""
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = a.area + b.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def greedy_match(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedily match same-class boxes by descending IoU."""
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt_box in enumerate(gt_boxes):
        for pred_index, pred_box in enumerate(pred_boxes):
            iou = bbox_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))

    candidates.sort(reverse=True, key=lambda item: item[0])
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))
    return matches


def metadata_json_path_for_stem(metadata_json_dir: Path, stem: str) -> Path | None:
    """Resolve a metadata JSON path for one image stem."""
    candidates = [
        metadata_json_dir / f"{stem}_sam_refine.json",
        metadata_json_dir / f"{stem}_grounding_dino.json",
        metadata_json_dir / f"{stem}_grounding_dino_tiled.json",
        metadata_json_dir / f"{stem}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def extract_inference_time(data: dict[str, Any]) -> float | None:
    """Extract an inference-time value from supported JSON metadata fields."""
    for key in ("total_inference_time_sec", "inference_time_sec"):
        value = data.get(key)
        if isinstance(value, (float, int)):
            return float(value)

    tiled = data.get("tiled_inference")
    if isinstance(tiled, dict) and isinstance(tiled.get("inference_time_sec"), (float, int)):
        return float(tiled["inference_time_sec"])

    sam_refine = data.get("sam_refine")
    if isinstance(sam_refine, dict) and isinstance(sam_refine.get("sam_inference_time_sec"), (float, int)):
        return float(sam_refine["sam_inference_time_sec"])
    return None


def read_inference_time(metadata_json_dir: Path, stem: str) -> float | None:
    """Read inference time for one image when metadata is available."""
    json_path = metadata_json_path_for_stem(metadata_json_dir, stem)
    if json_path is None:
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return extract_inference_time(data)


def size_bucket(box: Box) -> str:
    """Return COCO-style size bucket for one box."""
    area = box.area
    if area < SMALL_AREA:
        return "small"
    if area < MEDIUM_AREA:
        return "medium"
    return "large"


def add_match_stats(
    stats: MetricStats,
    gt_count: int,
    pred_count: int,
    matches: list[tuple[int, int, float]],
) -> None:
    """Add one class/image matching result to aggregate stats."""
    stats.gt_count += gt_count
    stats.pred_count += pred_count
    stats.matched_count += len(matches)
    stats.iou_sum += sum(iou for _, _, iou in matches)


def evaluate_boxes(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    class_names: dict[int, str],
    iou_threshold: float,
    overall_stats: MetricStats,
    class_stats: dict[int, MetricStats],
    size_stats: dict[str, MetricStats],
) -> None:
    """Evaluate one image and update aggregate stats."""
    class_ids = set(class_names) | {box.class_id for box in gt_boxes} | {box.class_id for box in pred_boxes}
    for class_id in sorted(class_ids):
        gt_for_class = [box for box in gt_boxes if box.class_id == class_id]
        pred_for_class = [box for box in pred_boxes if box.class_id == class_id]
        if not gt_for_class and not pred_for_class:
            continue

        matches = greedy_match(gt_for_class, pred_for_class, iou_threshold)
        add_match_stats(overall_stats, len(gt_for_class), len(pred_for_class), matches)
        add_match_stats(class_stats.setdefault(class_id, MetricStats()), len(gt_for_class), len(pred_for_class), matches)

        matched_gt = {gt_index for gt_index, _, _ in matches}
        matched_pred = {pred_index for _, pred_index, _ in matches}
        for gt_index, pred_index, iou in matches:
            bucket = size_bucket(gt_for_class[gt_index])
            size_stats[bucket].gt_count += 1
            size_stats[bucket].pred_count += 1
            size_stats[bucket].matched_count += 1
            size_stats[bucket].iou_sum += iou

        for gt_index, gt_box in enumerate(gt_for_class):
            if gt_index in matched_gt:
                continue
            size_stats[size_bucket(gt_box)].gt_count += 1

        for pred_index, pred_box in enumerate(pred_for_class):
            if pred_index in matched_pred:
                continue
            size_stats[size_bucket(pred_box)].pred_count += 1


def metric_row(stats: MetricStats) -> dict[str, Any]:
    """Build a CSV row fragment for metric stats."""
    return {
        "gt_count": stats.gt_count,
        "pred_count": stats.pred_count,
        "matched_count": stats.matched_count,
        "false_positive_count": stats.false_positive_count,
        "false_negative_count": stats.false_negative_count,
        "precision": f"{stats.precision:.6f}",
        "recall": f"{stats.recall:.6f}",
        "f1": f"{stats.f1:.6f}",
        "mean_iou": f"{stats.mean_iou:.6f}",
    }


def write_rows(output_path: Path, fieldnames: list[str], rows: list[dict[str, Any]], append_csv: bool) -> None:
    """Write or append CSV rows with a stable header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append_csv or not output_path.is_file() or output_path.stat().st_size == 0
    mode = "a" if append_csv else "w"
    with output_path.open(mode, encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(
    output_path: Path,
    overall_stats: MetricStats,
    eval_summary: EvalSummary,
    iou_threshold: float,
    method_name: str | None = None,
    append_csv: bool = False,
) -> None:
    """Write one-row overall summary CSV."""
    average_predictions = 0.0
    if eval_summary.evaluated_images > 0:
        average_predictions = overall_stats.pred_count / eval_summary.evaluated_images
    average_inference_time: str | float = ""
    if eval_summary.inference_time_images > 0:
        average_inference_time = f"{eval_summary.inference_time_sum / eval_summary.inference_time_images:.6f}"

    row = {
        **metric_row(overall_stats),
        "total_label_files": eval_summary.total_label_files,
        "evaluated_images": eval_summary.evaluated_images,
        "failed_images": eval_summary.failed_images,
        "missing_images": eval_summary.missing_images,
        "missing_gt_label_files": eval_summary.missing_gt_label_files,
        "missing_pred_label_files": eval_summary.missing_pred_label_files,
        "invalid_gt_label_lines": eval_summary.invalid_gt_label_lines,
        "invalid_pred_label_lines": eval_summary.invalid_pred_label_lines,
        "iou_threshold": f"{iou_threshold:.6f}",
        "average_predictions_per_image": f"{average_predictions:.6f}",
        "average_inference_time_sec": average_inference_time,
        "inference_time_images": eval_summary.inference_time_images,
    }
    if method_name is not None:
        row = {"method": method_name, **row}
    write_rows(output_path, list(row.keys()), [row], append_csv)


def write_class_csv(
    output_path: Path,
    class_stats: dict[int, MetricStats],
    class_names: dict[int, str],
    method_name: str | None = None,
    append_csv: bool = False,
) -> None:
    """Write per-class metrics CSV."""
    fieldnames = [
        "class_id",
        "class_name",
        "gt_count",
        "pred_count",
        "matched_count",
        "false_positive_count",
        "false_negative_count",
        "precision",
        "recall",
        "f1",
        "mean_iou",
    ]
    rows: list[dict[str, Any]] = []
    for class_id in sorted(class_names):
        stats = class_stats.get(class_id, MetricStats())
        row = {
            "class_id": class_id,
            "class_name": class_names[class_id],
            **metric_row(stats),
        }
        if method_name is not None:
            row = {"method": method_name, **row}
        rows.append(row)
    if method_name is not None:
        fieldnames = ["method", *fieldnames]
    write_rows(output_path, fieldnames, rows, append_csv)


def write_size_csv(
    output_path: Path,
    size_stats: dict[str, MetricStats],
    method_name: str | None = None,
    append_csv: bool = False,
) -> None:
    """Write COCO-style size-bucket metrics CSV."""
    fieldnames = [
        "size",
        "gt_count",
        "pred_count",
        "matched_count",
        "false_positive_count",
        "false_negative_count",
        "precision",
        "recall",
        "f1",
        "mean_iou",
    ]
    rows: list[dict[str, Any]] = []
    for bucket in SIZE_BUCKETS:
        row = {"size": bucket, **metric_row(size_stats[bucket])}
        if method_name is not None:
            row = {"method": method_name, **row}
        rows.append(row)
    if method_name is not None:
        fieldnames = ["method", *fieldnames]
    write_rows(output_path, fieldnames, rows, append_csv)


def print_summary(overall_stats: MetricStats, eval_summary: EvalSummary) -> None:
    """Print concise evaluation summary."""
    average_predictions = 0.0
    if eval_summary.evaluated_images > 0:
        average_predictions = overall_stats.pred_count / eval_summary.evaluated_images
    print("Auto-label quality evaluation summary:")
    print(f"- gt_count: {overall_stats.gt_count}")
    print(f"- pred_count: {overall_stats.pred_count}")
    print(f"- matched_count: {overall_stats.matched_count}")
    print(f"- false_positive_count: {overall_stats.false_positive_count}")
    print(f"- false_negative_count: {overall_stats.false_negative_count}")
    print(f"- precision: {overall_stats.precision:.6f}")
    print(f"- recall: {overall_stats.recall:.6f}")
    print(f"- f1: {overall_stats.f1:.6f}")
    print(f"- mean_iou: {overall_stats.mean_iou:.6f}")
    print(f"- average_predictions_per_image: {average_predictions:.6f}")
    if eval_summary.inference_time_images > 0:
        average_time = eval_summary.inference_time_sum / eval_summary.inference_time_images
        print(f"- average_inference_time_sec: {average_time:.6f}")
    else:
        print("- average_inference_time_sec: unavailable")
    print(f"- evaluated_images: {eval_summary.evaluated_images}")
    print(f"- failed_images: {eval_summary.failed_images}")
    print(f"- missing_images: {eval_summary.missing_images}")
    print(f"- missing_gt_label_files: {eval_summary.missing_gt_label_files}")
    print(f"- missing_pred_label_files: {eval_summary.missing_pred_label_files}")
    print(f"- invalid_gt_label_lines: {eval_summary.invalid_gt_label_lines}")
    print(f"- invalid_pred_label_lines: {eval_summary.invalid_pred_label_lines}")


def write_failure_log(output_summary_csv: Path, failure_records: list[str]) -> None:
    """Write image-level failure records when present."""
    if not failure_records:
        return
    failure_log = output_summary_csv.with_name(output_summary_csv.stem + "_failures.txt")
    failure_log.write_text("\n".join(failure_records) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote failure log: {failure_log.as_posix()}")


def run(args: argparse.Namespace) -> tuple[MetricStats, EvalSummary]:
    """Run auto-label quality evaluation."""
    validate_args(args)

    gt_label_dir = Path(args.gt_label_dir)
    pred_label_dir = Path(args.pred_label_dir)
    image_dir = Path(args.image_dir)
    metadata_json_dir = Path(args.metadata_json_dir) if args.metadata_json_dir else None
    if not gt_label_dir.is_dir():
        raise FileNotFoundError(f"GT label directory does not exist: {gt_label_dir}")
    if not pred_label_dir.is_dir():
        raise FileNotFoundError(f"Prediction label directory does not exist: {pred_label_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if metadata_json_dir is not None and not metadata_json_dir.is_dir():
        raise FileNotFoundError(f"Metadata JSON directory does not exist: {metadata_json_dir}")

    class_names = load_class_names(Path(args.classes_config))
    valid_class_ids = set(class_names)
    image_path_by_stem: dict[str, Path] = {}
    if args.image_list:
        image_paths = resolve_image_entries(image_dir, read_image_list(Path(args.image_list)), DEFAULT_IMAGE_EXTS)
        if args.limit is not None:
            image_paths = image_paths[: args.limit]
        image_path_by_stem = {image_path.stem: image_path for image_path in image_paths}
        pred_label_files = collect_pred_label_files_for_images(pred_label_dir, image_paths, None)
    else:
        pred_label_files = collect_pred_label_files(pred_label_dir, args.limit)

    overall_stats = MetricStats()
    class_stats = {class_id: MetricStats() for class_id in class_names}
    size_stats = {bucket: MetricStats() for bucket in SIZE_BUCKETS}
    eval_summary = EvalSummary(total_label_files=len(pred_label_files))
    failure_records: list[str] = []

    for pred_label_path in pred_label_files:
        stem = pred_label_path.stem
        try:
            image_path = image_path_by_stem.get(stem)
            if image_path is None:
                image_path = resolve_image_path(image_dir, stem)
            if image_path is None:
                eval_summary.missing_images += 1
                raise FileNotFoundError(f"Image not found for label stem: {stem}")

            image_width, image_height = read_image_size(image_path)
            if image_width <= 0 or image_height <= 0:
                raise ValueError(f"Invalid image size for {image_path}: {image_width}x{image_height}")

            gt_label_path = gt_label_dir / f"{stem}.txt"
            gt_boxes, invalid_gt, gt_exists = parse_yolo_label_file(
                gt_label_path,
                image_width,
                image_height,
                valid_class_ids,
            )
            pred_boxes, invalid_pred, pred_exists = parse_yolo_label_file(
                pred_label_path,
                image_width,
                image_height,
                valid_class_ids,
            )

            if not gt_exists:
                eval_summary.missing_gt_label_files += 1
            if not pred_exists:
                eval_summary.missing_pred_label_files += 1
            eval_summary.invalid_gt_label_lines += invalid_gt
            eval_summary.invalid_pred_label_lines += invalid_pred

            evaluate_boxes(
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                class_names=class_names,
                iou_threshold=args.iou_threshold,
                overall_stats=overall_stats,
                class_stats=class_stats,
                size_stats=size_stats,
            )
            eval_summary.evaluated_images += 1
            if metadata_json_dir is not None:
                inference_time = read_inference_time(metadata_json_dir, stem)
                if inference_time is not None:
                    eval_summary.inference_time_sum += inference_time
                    eval_summary.inference_time_images += 1
        except Exception as exc:
            eval_summary.failed_images += 1
            message = f"{pred_label_path.as_posix()}: {exc}"
            failure_records.append(message)
            print(f"[WARN] Failed image skipped: {message}")
            continue

    write_summary_csv(
        Path(args.out_summary_csv),
        overall_stats,
        eval_summary,
        args.iou_threshold,
        method_name=args.method_name,
        append_csv=args.append_csv,
    )
    write_class_csv(
        Path(args.out_class_csv),
        class_stats,
        class_names,
        method_name=args.method_name,
        append_csv=args.append_csv,
    )
    write_size_csv(
        Path(args.out_size_csv),
        size_stats,
        method_name=args.method_name,
        append_csv=args.append_csv,
    )
    write_failure_log(Path(args.out_summary_csv), failure_records)

    return overall_stats, eval_summary


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    try:
        overall_stats, eval_summary = run(args)
        print_summary(overall_stats, eval_summary)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
