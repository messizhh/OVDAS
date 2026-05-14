"""Convert VisDrone DET annotations to YOLO label format."""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VISDRONE_TO_YOLO = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    9: 6,   # bus
    10: 7,  # motor
}
EXPECTED_CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "bus",
    "motor",
]


@dataclass
class Summary:
    """Accumulate conversion statistics."""

    total_images: int = 0
    converted_images: int = 0
    copied_images: int = 0
    total_boxes: int = 0
    kept_boxes: int = 0
    ignored_boxes: int = 0
    failed_images: int = 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert VisDrone DET annotations to YOLO format."
    )
    parser.add_argument(
        "--src-root",
        required=True,
        help="VisDrone split root containing images/ and annotations/.",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Output root for YOLO-format data.",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Output split name, for example train or val.",
    )
    parser.add_argument(
        "--classes-config",
        default="configs/classes_visdrone.yaml",
        help="Class config used to validate the expected 8 target classes.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into images/<split>/ when set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to process.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a mapping."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def validate_classes_config(path: Path) -> None:
    """Validate that the class config contains the expected target classes."""
    data = load_yaml(path)
    classes = data.get("classes", [])
    if not isinstance(classes, list):
        raise ValueError(f"'classes' must be a list in {path}")

    names: list[str] = []
    for item in classes:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])

    missing = [name for name in EXPECTED_CLASS_NAMES if name not in names]
    if missing:
        raise ValueError(
            f"Missing classes in {path}: " + ", ".join(missing)
        )


def list_images(image_dir: Path, limit: int | None) -> list[Path]:
    """List supported image files in deterministic order."""
    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be greater than or equal to 0")
        return image_paths[:limit]
    return image_paths


def read_image_size(image_path: Path) -> tuple[int, int]:
    """Read image width and height without loading full image content."""
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
    raise ValueError(f"Unsupported image extension: {image_path.suffix}")


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


def clip_bbox(
    left: float,
    top: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Clip a bbox to image bounds and return xyxy coordinates."""
    x1 = max(0.0, left)
    y1 = max(0.0, top)
    x2 = min(float(image_width), left + width)
    y2 = min(float(image_height), top + height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def to_yolo_line(
    yolo_class_id: int,
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> str:
    """Convert clipped xyxy coordinates to one YOLO label line."""
    x1, y1, x2, y2 = xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    x_center = x1 + box_width / 2.0
    y_center = y1 + box_height / 2.0
    values = [
        yolo_class_id,
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    ]
    return (
        f"{values[0]} "
        f"{values[1]:.6f} "
        f"{values[2]:.6f} "
        f"{values[3]:.6f} "
        f"{values[4]:.6f}"
    )


def convert_annotation(
    annotation_path: Path,
    image_width: int,
    image_height: int,
) -> tuple[list[str], int, int, int]:
    """Convert one VisDrone annotation file to YOLO lines."""
    if not annotation_path.exists():
        return [], 0, 0, 0

    yolo_lines: list[str] = []
    total_boxes = 0
    kept_boxes = 0
    ignored_boxes = 0

    text = annotation_path.read_text(encoding="utf-8").strip()
    if not text:
        return yolo_lines, total_boxes, kept_boxes, ignored_boxes

    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            ignored_boxes += 1
            continue

        total_boxes += 1
        try:
            left = float(parts[0])
            top = float(parts[1])
            width = float(parts[2])
            height = float(parts[3])
            category = int(float(parts[5]))
        except ValueError:
            ignored_boxes += 1
            print(
                f"[WARN] Invalid annotation line ignored: "
                f"{annotation_path}:{line_number}"
            )
            continue

        yolo_class_id = VISDRONE_TO_YOLO.get(category)
        if yolo_class_id is None or width <= 0 or height <= 0:
            ignored_boxes += 1
            continue

        clipped = clip_bbox(left, top, width, height, image_width, image_height)
        if clipped is None:
            ignored_boxes += 1
            continue

        yolo_lines.append(
            to_yolo_line(yolo_class_id, clipped, image_width, image_height)
        )
        kept_boxes += 1

    return yolo_lines, total_boxes, kept_boxes, ignored_boxes


def convert_image(
    image_path: Path,
    annotation_dir: Path,
    output_image_dir: Path,
    output_label_dir: Path,
    copy_images: bool,
) -> tuple[bool, int, int, int, bool]:
    """Convert labels for one image and optionally copy the image."""
    try:
        image_width, image_height = read_image_size(image_path)
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid image size: {image_width}x{image_height}")

        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        yolo_lines, total_boxes, kept_boxes, ignored_boxes = convert_annotation(
            annotation_path, image_width, image_height
        )

        label_path = output_label_dir / f"{image_path.stem}.txt"
        label_text = "\n".join(yolo_lines)
        if label_text:
            label_text += "\n"
        label_path.write_text(label_text, encoding="utf-8")

        copied = False
        if copy_images:
            shutil.copy2(image_path, output_image_dir / image_path.name)
            copied = True

        return True, total_boxes, kept_boxes, ignored_boxes, copied
    except (OSError, ValueError) as exc:
        print(f"[WARN] Failed image skipped: {image_path} ({exc})")
        return False, 0, 0, 0, False


def print_summary(summary: Summary) -> None:
    """Print conversion summary."""
    print("VisDrone to YOLO conversion summary:")
    print(f"- total_images: {summary.total_images}")
    print(f"- converted_images: {summary.converted_images}")
    print(f"- copied_images: {summary.copied_images}")
    print(f"- total_boxes: {summary.total_boxes}")
    print(f"- kept_boxes: {summary.kept_boxes}")
    print(f"- ignored_boxes: {summary.ignored_boxes}")
    print(f"- failed_images: {summary.failed_images}")


def run(
    src_root: Path,
    out_root: Path,
    split: str,
    classes_config: Path,
    copy_images: bool,
    limit: int | None,
) -> int:
    """Run VisDrone DET to YOLO conversion."""
    image_dir = src_root / "images"
    annotation_dir = src_root / "annotations"
    if not image_dir.is_dir():
        print(f"[ERROR] Missing image directory: {image_dir}")
        return 1
    if not annotation_dir.is_dir():
        print(f"[ERROR] Missing annotation directory: {annotation_dir}")
        return 1

    try:
        validate_classes_config(classes_config)
        image_paths = list_images(image_dir, limit)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    output_image_dir = out_root / "images" / split
    output_label_dir = out_root / "labels" / split
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    summary = Summary(total_images=len(image_paths))
    if not image_paths:
        print(f"[WARN] No supported images found in: {image_dir}")

    for image_path in image_paths:
        success, total_boxes, kept_boxes, ignored_boxes, copied = convert_image(
            image_path=image_path,
            annotation_dir=annotation_dir,
            output_image_dir=output_image_dir,
            output_label_dir=output_label_dir,
            copy_images=copy_images,
        )
        if success:
            summary.converted_images += 1
            summary.total_boxes += total_boxes
            summary.kept_boxes += kept_boxes
            summary.ignored_boxes += ignored_boxes
            if copied:
                summary.copied_images += 1
        else:
            summary.failed_images += 1

    print_summary(summary)
    return 0


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    return run(
        src_root=Path(args.src_root),
        out_root=Path(args.out_root),
        split=args.split,
        classes_config=Path(args.classes_config),
        copy_images=args.copy_images,
        limit=args.limit,
    )


if __name__ == "__main__":
    sys.exit(main())
