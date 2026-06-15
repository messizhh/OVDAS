"""Generate YOLO labels from Grounding DINO or SAM-refined JSON results."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.open_vocab.phrase_normalization import clean_phrase, load_class_mapping, resolve_class_id
from src.utils.image_lists import read_image_list, resolve_image_entries


DEFAULT_IMAGE_EXTS = "jpg,jpeg,png,bmp,tif,tiff"
AUTO_JSON_SUFFIXES = ("_grounding_dino", "_grounding_dino_tiled", "_sam_refine")


@dataclass
class LabelRecord:
    """One generated YOLO label."""

    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class Summary:
    """Accumulate automatic label generation statistics."""

    total_json_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_detections: int = 0
    kept_labels: int = 0
    mapped_alias_labels: int = 0
    skipped_low_score: int = 0
    skipped_unknown_class: int = 0
    skipped_invalid_bbox: int = 0
    skipped_small_box: int = 0
    skipped_size_aware_low_score: int = 0
    empty_label_files: int = 0
    ensured_empty_label_files: int = 0
    copied_images: int = 0
    output_label_dir: str = ""
    class_counts: dict[int, int] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert automatic detection JSON files to YOLO txt labels."
    )
    parser.add_argument("--json-dir", required=True, help="Directory containing auto JSON files.")
    parser.add_argument("--image-dir", required=True, help="Directory containing source images.")
    parser.add_argument(
        "--image-list",
        default=None,
        help="Optional txt/json image list. Used to keep subset experiments fixed.",
    )
    parser.add_argument("--out-label-dir", required=True, help="Output directory for YOLO txt labels.")
    parser.add_argument(
        "--classes-config",
        default="configs/classes_visdrone.yaml",
        help="Class config YAML used for phrase to class-id mapping.",
    )
    parser.add_argument(
        "--bbox-key",
        default="refined_bbox_xyxy",
        help="Detection bbox field to use first, for example refined_bbox_xyxy.",
    )
    parser.add_argument(
        "--fallback-bbox-key",
        default="bbox_xyxy",
        help="Fallback detection bbox field when --bbox-key is missing or invalid.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.35,
        help="Minimum detection score required to keep a label.",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=4.0,
        help="Minimum clipped bbox area in pixels.",
    )
    parser.add_argument(
        "--enable-size-aware-filter",
        action="store_true",
        help="Use bbox area ratio to choose score thresholds for small/medium/large objects.",
    )
    parser.add_argument(
        "--small-area-ratio",
        type=float,
        default=0.001,
        help="Area ratio below this value is treated as small.",
    )
    parser.add_argument(
        "--medium-area-ratio",
        type=float,
        default=0.01,
        help="Area ratio below this value is treated as medium; larger boxes are large.",
    )
    parser.add_argument(
        "--small-score-threshold",
        type=float,
        default=0.20,
        help="Score threshold for small boxes when size-aware filtering is enabled.",
    )
    parser.add_argument(
        "--medium-score-threshold",
        type=float,
        default=0.25,
        help="Score threshold for medium boxes when size-aware filtering is enabled.",
    )
    parser.add_argument(
        "--large-score-threshold",
        type=float,
        default=0.35,
        help="Score threshold for large boxes when size-aware filtering is enabled.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a JSON when its output txt label already exists.",
    )
    parser.add_argument(
        "--ensure-all-images",
        action="store_true",
        help="Create an empty txt label for each image that did not receive an output label.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy source images to --out-image-dir.",
    )
    parser.add_argument(
        "--out-image-dir",
        default=None,
        help="Output image directory. Required when --copy-images is set.",
    )
    parser.add_argument(
        "--stats-csv",
        required=True,
        help="Output CSV path for summary and per-class label counts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of JSON files to process after sorting. Omit for all JSON files.",
    )
    parser.add_argument(
        "--image-exts",
        default=DEFAULT_IMAGE_EXTS,
        help="Comma-separated image extensions, for example jpg,jpeg,png.",
    )
    return parser.parse_args()


def parse_image_exts(raw_exts: str) -> list[str]:
    """Parse image extensions in deterministic order."""
    exts: list[str] = []
    seen: set[str] = set()
    for item in raw_exts.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in seen:
            exts.append(ext)
            seen.add(ext)
    if not exts:
        raise ValueError("--image-exts must contain at least one extension.")
    return exts


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line values that do not depend on data files."""
    threshold_values = {
        "--score-threshold": args.score_threshold,
        "--small-score-threshold": args.small_score_threshold,
        "--medium-score-threshold": args.medium_score_threshold,
        "--large-score-threshold": args.large_score_threshold,
    }
    for name, value in threshold_values.items():
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if args.min_box_area < 0.0:
        raise ValueError("--min-box-area must be greater than or equal to 0.")
    if args.small_area_ratio < 0.0:
        raise ValueError("--small-area-ratio must be greater than or equal to 0.")
    if args.medium_area_ratio <= args.small_area_ratio:
        raise ValueError("--medium-area-ratio must be greater than --small-area-ratio.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    if args.copy_images and not args.out_image_dir:
        raise ValueError("--out-image-dir is required when --copy-images is set.")


def collect_json_files(json_dir: Path, limit: int | None) -> list[Path]:
    """Collect sorted JSON files from one directory."""
    if not json_dir.is_dir():
        raise FileNotFoundError(f"JSON directory does not exist: {json_dir}")
    json_paths = sorted(path for path in json_dir.iterdir() if path.is_file() and path.suffix == ".json")
    if limit is not None:
        return json_paths[:limit]
    return json_paths


def json_path_for_image(json_dir: Path, image_stem: str) -> Path | None:
    """Resolve an auto JSON path for one image stem."""
    candidates = [
        json_dir / f"{image_stem}_grounding_dino.json",
        json_dir / f"{image_stem}_grounding_dino_tiled.json",
        json_dir / f"{image_stem}_sam_refine.json",
        json_dir / f"{image_stem}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def collect_json_files_for_images(json_dir: Path, image_paths: list[Path], limit: int | None) -> list[Path]:
    """Collect existing JSON files for a fixed image subset."""
    if not json_dir.is_dir():
        raise FileNotFoundError(f"JSON directory does not exist: {json_dir}")
    json_paths: list[Path] = []
    for image_path in image_paths:
        json_path = json_path_for_image(json_dir, image_path.stem)
        if json_path is not None:
            json_paths.append(json_path)
    if limit is not None:
        return json_paths[:limit]
    return json_paths


def collect_images(image_dir: Path, image_exts: list[str]) -> list[Path]:
    """Collect sorted image files from one directory."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    image_ext_set = set(image_exts)
    return sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in image_ext_set
    )


def load_auto_json(json_path: Path) -> dict[str, Any]:
    """Load one automatic annotation JSON file."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {json_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {json_path}")
    detections = data.get("detections", [])
    if not isinstance(detections, list):
        raise ValueError(f"'detections' must be a list in {json_path}")
    return data


def strip_auto_suffix(stem: str) -> str:
    """Remove known auto-output suffixes from a JSON stem."""
    for suffix in AUTO_JSON_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def resolve_image_path(
    auto_data: dict[str, Any],
    json_path: Path,
    image_dir: Path,
    image_exts: list[str],
) -> Path:
    """Resolve the source image path for one JSON file."""
    raw_image_path = auto_data.get("image_path")
    if isinstance(raw_image_path, str) and raw_image_path.strip():
        candidate = Path(raw_image_path)
        if candidate.is_file():
            return candidate
        image_candidate = image_dir / candidate.name
        if image_candidate.is_file():
            return image_candidate

    stem = strip_auto_suffix(json_path.stem)
    for ext in image_exts:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find image for JSON: {json_path}")


def read_image_size(image_path: Path) -> tuple[int, int]:
    """Read image width and height without requiring Pillow at import time."""
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
    """Read PNG width and height from the IHDR chunk."""
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


def coerce_score(value: Any) -> float | None:
    """Convert a score-like value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_bbox_xyxy(value: Any, image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    """Convert a bbox-like value to a clipped xyxy box."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None

    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def select_bbox(
    detection: dict[str, Any],
    bbox_key: str,
    fallback_bbox_key: str,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Select and validate the primary or fallback bbox from one detection."""
    bbox = coerce_bbox_xyxy(detection.get(bbox_key), image_width, image_height)
    if bbox is not None:
        return bbox
    if fallback_bbox_key and fallback_bbox_key != bbox_key:
        return coerce_bbox_xyxy(detection.get(fallback_bbox_key), image_width, image_height)
    return None


def bbox_area(xyxy: tuple[float, float, float, float]) -> float:
    """Return bbox area in pixels."""
    x1, y1, x2, y2 = xyxy
    return (x2 - x1) * (y2 - y1)


def size_aware_threshold(
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    args: argparse.Namespace,
) -> tuple[str, float]:
    """Return size bucket and score threshold for one bbox."""
    image_area = max(1.0, float(image_width * image_height))
    area_ratio = bbox_area(xyxy) / image_area
    if area_ratio < args.small_area_ratio:
        return "small", args.small_score_threshold
    if area_ratio < args.medium_area_ratio:
        return "medium", args.medium_score_threshold
    return "large", args.large_score_threshold


def bbox_to_yolo(
    class_id: int,
    class_name: str,
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> LabelRecord:
    """Convert one clipped xyxy box to a normalized YOLO label record."""
    x1, y1, x2, y2 = xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    return LabelRecord(
        class_id=class_id,
        class_name=class_name,
        x_center=(x1 + box_width / 2.0) / image_width,
        y_center=(y1 + box_height / 2.0) / image_height,
        width=box_width / image_width,
        height=box_height / image_height,
    )


def format_yolo_line(label: LabelRecord) -> str:
    """Format one YOLO label line."""
    return (
        f"{label.class_id} "
        f"{label.x_center:.6f} "
        f"{label.y_center:.6f} "
        f"{label.width:.6f} "
        f"{label.height:.6f}"
    )


def convert_detections(
    detections: list[Any],
    image_width: int,
    image_height: int,
    phrase_to_id: dict[str, int],
    id_to_name: dict[int, str],
    args: argparse.Namespace,
    summary: Summary,
) -> list[LabelRecord]:
    """Convert one image's detections to YOLO label records."""
    labels: list[LabelRecord] = []
    for item in detections:
        summary.total_detections += 1
        if not isinstance(item, dict):
            summary.skipped_invalid_bbox += 1
            continue

        score = coerce_score(item.get("score"))
        if (
            not args.enable_size_aware_filter
            and score is not None
            and score < args.score_threshold
        ):
            summary.skipped_low_score += 1
            continue

        phrase = clean_phrase(item.get("phrase", ""))
        class_id, used_alias = resolve_class_id(phrase, phrase_to_id)
        if class_id is None:
            summary.skipped_unknown_class += 1
            continue
        if used_alias:
            summary.mapped_alias_labels += 1

        bbox = select_bbox(
            detection=item,
            bbox_key=args.bbox_key,
            fallback_bbox_key=args.fallback_bbox_key,
            image_width=image_width,
            image_height=image_height,
        )
        if bbox is None:
            summary.skipped_invalid_bbox += 1
            continue
        if bbox_area(bbox) < args.min_box_area:
            summary.skipped_small_box += 1
            continue
        if args.enable_size_aware_filter:
            _, threshold = size_aware_threshold(bbox, image_width, image_height, args)
            if score is not None and score < threshold:
                summary.skipped_low_score += 1
                summary.skipped_size_aware_low_score += 1
                continue

        class_name = id_to_name.get(class_id, phrase)
        labels.append(bbox_to_yolo(class_id, class_name, bbox, image_width, image_height))

    return labels


def write_label_file(labels: list[LabelRecord], output_label_path: Path) -> None:
    """Write a YOLO txt file, including empty files for empty labels."""
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    label_text = "\n".join(format_yolo_line(label) for label in labels)
    if label_text:
        label_text += "\n"
    output_label_path.write_text(label_text, encoding="utf-8")


def ensure_empty_labels_for_all_images(
    image_dir: Path,
    out_label_dir: Path,
    image_exts: list[str],
    summary: Summary,
    image_paths: list[Path] | None = None,
) -> None:
    """Create empty YOLO txt files for images that have no generated label file."""
    paths = image_paths if image_paths is not None else collect_images(image_dir, image_exts)
    for image_path in paths:
        output_label_path = out_label_dir / f"{image_path.stem}.txt"
        if output_label_path.exists():
            continue
        write_label_file([], output_label_path)
        summary.empty_label_files += 1
        summary.ensured_empty_label_files += 1


def write_stats_csv(
    summary: Summary,
    id_to_name: dict[int, str],
    stats_csv: Path,
    args: argparse.Namespace,
) -> None:
    """Write summary and per-class label counts to CSV."""
    stats_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        ("summary", "", "", "total_json_files", summary.total_json_files),
        ("summary", "", "", "processed_files", summary.processed_files),
        ("summary", "", "", "failed_files", summary.failed_files),
        ("summary", "", "", "total_detections", summary.total_detections),
        ("summary", "", "", "kept_labels", summary.kept_labels),
        ("summary", "", "", "mapped_alias_labels", summary.mapped_alias_labels),
        ("summary", "", "", "skipped_low_score", summary.skipped_low_score),
        ("summary", "", "", "skipped_unknown_class", summary.skipped_unknown_class),
        ("summary", "", "", "skipped_invalid_bbox", summary.skipped_invalid_bbox),
        ("summary", "", "", "skipped_small_box", summary.skipped_small_box),
        ("summary", "", "", "skipped_size_aware_low_score", summary.skipped_size_aware_low_score),
        ("summary", "", "", "empty_label_files", summary.empty_label_files),
        ("summary", "", "", "ensured_empty_label_files", summary.ensured_empty_label_files),
        ("summary", "", "", "copied_images", summary.copied_images),
        ("config", "", "", "enable_size_aware_filter", args.enable_size_aware_filter),
        ("config", "", "", "score_threshold", args.score_threshold),
        ("config", "", "", "small_area_ratio", args.small_area_ratio),
        ("config", "", "", "medium_area_ratio", args.medium_area_ratio),
        ("config", "", "", "small_score_threshold", args.small_score_threshold),
        ("config", "", "", "medium_score_threshold", args.medium_score_threshold),
        ("config", "", "", "large_score_threshold", args.large_score_threshold),
    ]

    with stats_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["scope", "class_id", "class_name", "metric", "value"])
        writer.writerows(summary_rows)
        for class_id in sorted(id_to_name):
            writer.writerow(
                [
                    "class",
                    class_id,
                    id_to_name[class_id],
                    "kept_labels",
                    summary.class_counts.get(class_id, 0),
                ]
            )


def write_metadata_json(summary: Summary, args: argparse.Namespace, metadata_json: Path) -> None:
    """Write label-generation metadata, including size-aware filter parameters."""
    metadata = {
        "json_dir": args.json_dir,
        "image_dir": args.image_dir,
        "image_list": args.image_list,
        "out_label_dir": args.out_label_dir,
        "bbox_key": args.bbox_key,
        "fallback_bbox_key": args.fallback_bbox_key,
        "score_threshold": args.score_threshold,
        "min_box_area": args.min_box_area,
        "enable_size_aware_filter": args.enable_size_aware_filter,
        "size_aware_filter": {
            "small_area_ratio": args.small_area_ratio,
            "medium_area_ratio": args.medium_area_ratio,
            "small_score_threshold": args.small_score_threshold,
            "medium_score_threshold": args.medium_score_threshold,
            "large_score_threshold": args.large_score_threshold,
        },
        "summary": {
            "total_json_files": summary.total_json_files,
            "processed_files": summary.processed_files,
            "failed_files": summary.failed_files,
            "total_detections": summary.total_detections,
            "kept_labels": summary.kept_labels,
            "skipped_low_score": summary.skipped_low_score,
            "skipped_size_aware_low_score": summary.skipped_size_aware_low_score,
            "empty_label_files": summary.empty_label_files,
            "ensured_empty_label_files": summary.ensured_empty_label_files,
        },
    }
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(summary: Summary) -> None:
    """Print the required batch summary."""
    print("Auto YOLO label generation summary:")
    print(f"- total_json_files: {summary.total_json_files}")
    print(f"- processed_files: {summary.processed_files}")
    print(f"- failed_files: {summary.failed_files}")
    print(f"- total_detections: {summary.total_detections}")
    print(f"- kept_labels: {summary.kept_labels}")
    print(f"- mapped_alias_labels: {summary.mapped_alias_labels}")
    print(f"- skipped_low_score: {summary.skipped_low_score}")
    print(f"- skipped_unknown_class: {summary.skipped_unknown_class}")
    print(f"- skipped_invalid_bbox: {summary.skipped_invalid_bbox}")
    print(f"- skipped_small_box: {summary.skipped_small_box}")
    print(f"- skipped_size_aware_low_score: {summary.skipped_size_aware_low_score}")
    print(f"- empty_label_files: {summary.empty_label_files}")
    print(f"- ensured_empty_label_files: {summary.ensured_empty_label_files}")
    print(f"- output_label_dir: {summary.output_label_dir}")


def print_filter_config(args: argparse.Namespace) -> None:
    """Print score-filter parameters used for this run."""
    print("Auto label filter config:")
    print(f"- score_threshold: {args.score_threshold}")
    print(f"- enable_size_aware_filter: {args.enable_size_aware_filter}")
    print(f"- small_area_ratio: {args.small_area_ratio}")
    print(f"- medium_area_ratio: {args.medium_area_ratio}")
    print(f"- small_score_threshold: {args.small_score_threshold}")
    print(f"- medium_score_threshold: {args.medium_score_threshold}")
    print(f"- large_score_threshold: {args.large_score_threshold}")


def process_json_file(
    json_path: Path,
    image_dir: Path,
    out_label_dir: Path,
    out_image_dir: Path | None,
    image_exts: list[str],
    phrase_to_id: dict[str, int],
    id_to_name: dict[int, str],
    args: argparse.Namespace,
    summary: Summary,
) -> None:
    """Process one JSON file into one YOLO label txt file."""
    auto_data = load_auto_json(json_path)
    image_path = resolve_image_path(auto_data, json_path, image_dir, image_exts)
    image_width, image_height = read_image_size(image_path)
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size for {image_path}: {image_width}x{image_height}")

    detections = auto_data.get("detections", [])
    labels = convert_detections(
        detections=detections,
        image_width=image_width,
        image_height=image_height,
        phrase_to_id=phrase_to_id,
        id_to_name=id_to_name,
        args=args,
        summary=summary,
    )

    output_label_path = out_label_dir / f"{image_path.stem}.txt"
    write_label_file(labels, output_label_path)
    if not labels:
        summary.empty_label_files += 1

    if args.copy_images and out_image_dir is not None:
        out_image_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, out_image_dir / image_path.name)
        summary.copied_images += 1

    summary.kept_labels += len(labels)
    for label in labels:
        summary.class_counts[label.class_id] = summary.class_counts.get(label.class_id, 0) + 1


def run(args: argparse.Namespace) -> Summary:
    """Run automatic label conversion."""
    validate_args(args)

    json_dir = Path(args.json_dir)
    image_dir = Path(args.image_dir)
    out_label_dir = Path(args.out_label_dir)
    out_image_dir = Path(args.out_image_dir) if args.out_image_dir else None
    stats_csv = Path(args.stats_csv)
    image_exts = parse_image_exts(args.image_exts)
    image_paths: list[Path] | None = None

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    phrase_to_id, id_to_name = load_class_mapping(Path(args.classes_config))
    if args.image_list:
        image_paths = resolve_image_entries(image_dir, read_image_list(Path(args.image_list)), image_exts)
        if args.limit is not None:
            image_paths = image_paths[: args.limit]
        json_paths = collect_json_files_for_images(json_dir, image_paths, None)
    else:
        json_paths = collect_json_files(json_dir, args.limit)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    summary = Summary(
        total_json_files=len(json_paths),
        output_label_dir=out_label_dir.as_posix(),
    )
    failure_records: list[str] = []

    for json_path in json_paths:
        try:
            auto_data = load_auto_json(json_path)
            image_path = resolve_image_path(auto_data, json_path, image_dir, image_exts)
            output_label_path = out_label_dir / f"{image_path.stem}.txt"
            if args.skip_existing and output_label_path.is_file():
                continue
            process_json_file(
                json_path=json_path,
                image_dir=image_dir,
                out_label_dir=out_label_dir,
                out_image_dir=out_image_dir,
                image_exts=image_exts,
                phrase_to_id=phrase_to_id,
                id_to_name=id_to_name,
                args=args,
                summary=summary,
            )
            summary.processed_files += 1
        except Exception as exc:
            summary.failed_files += 1
            failure_records.append(f"{json_path.as_posix()}: {exc}")
            print(f"[WARN] Failed JSON skipped: {json_path} ({exc})")

    if failure_records:
        failure_log = out_label_dir / "auto_label_generation_failures.txt"
        failure_log.write_text("\n".join(failure_records) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote failure log: {failure_log.as_posix()}")

    if args.ensure_all_images:
        ensure_empty_labels_for_all_images(
            image_dir=image_dir,
            out_label_dir=out_label_dir,
            image_exts=image_exts,
            summary=summary,
            image_paths=image_paths,
        )

    write_stats_csv(summary, id_to_name, stats_csv, args)
    write_metadata_json(summary, args, stats_csv.with_suffix(".metadata.json"))
    return summary


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    try:
        summary = run(args)
        print_filter_config(args)
        print_summary(summary)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
