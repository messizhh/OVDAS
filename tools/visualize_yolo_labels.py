"""Visualize YOLO labels for converted VisDrone samples."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_IMAGE_EXTS = ".jpg,.jpeg,.png,.bmp"


@dataclass
class LabelBox:
    """One YOLO label converted to normalized xywh values."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class Summary:
    """Accumulate visualization statistics."""

    total_images: int = 0
    visualized_images: int = 0
    missing_label_files: int = 0
    invalid_label_lines: int = 0
    failed_images: int = 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize YOLO labels for one dataset split."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="YOLO-format data root containing images/<split>/ and labels/<split>/.",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Dataset split to visualize, for example train or val.",
    )
    parser.add_argument(
        "--classes-config",
        default="configs/classes_visdrone.yaml",
        help="Class config YAML with class id and name entries.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where visualization images will be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to visualize.",
    )
    parser.add_argument(
        "--image-exts",
        default=DEFAULT_IMAGE_EXTS,
        help="Comma-separated image extensions to scan.",
    )
    return parser.parse_args()


def parse_image_exts(value: str) -> set[str]:
    """Parse a comma-separated extension list."""
    extensions: set[str] = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        extensions.add(item if item.startswith(".") else f".{item}")
    return extensions


def load_class_names(path: Path) -> list[str]:
    """Load class names from classes_visdrone.yaml."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")

    classes = data.get("classes", [])
    if not isinstance(classes, list):
        raise ValueError(f"'classes' must be a list in {path}")

    names_by_id: dict[int, str] = {}
    ordered_names: list[str] = []
    for index, item in enumerate(classes):
        if isinstance(item, str):
            names_by_id[index] = item
            ordered_names.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            class_id = item.get("id", index)
            if not isinstance(class_id, int):
                raise ValueError(f"Invalid class id in {path}: {class_id}")
            names_by_id[class_id] = item["name"]

    if names_by_id:
        max_id = max(names_by_id)
        return [names_by_id.get(class_id, f"class_{class_id}") for class_id in range(max_id + 1)]
    return ordered_names


def list_images(image_dir: Path, image_exts: set[str], limit: int | None) -> list[Path]:
    """List images in deterministic order."""
    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_exts
        ],
        key=lambda path: path.name.lower(),
    )
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be greater than or equal to 0")
        return image_paths[:limit]
    return image_paths


def parse_label_file(
    label_path: Path,
    class_count: int,
) -> tuple[list[LabelBox], int, bool]:
    """Parse one YOLO label file and return boxes, invalid line count, and existence."""
    if not label_path.exists():
        return [], 0, False

    boxes: list[LabelBox] = []
    invalid_lines = 0
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes, invalid_lines, True

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

        if (
            class_id < 0
            or class_id >= class_count
            or width <= 0
            or height <= 0
        ):
            invalid_lines += 1
            print(f"[WARN] Out-of-range label ignored: {label_path}:{line_number}")
            continue

        boxes.append(
            LabelBox(
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )

    return boxes, invalid_lines, True


def box_to_pixels(
    box: LabelBox,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Convert normalized YOLO xywh to clipped pixel xyxy."""
    x1 = (box.x_center - box.width / 2.0) * image_width
    y1 = (box.y_center - box.height / 2.0) * image_height
    x2 = (box.x_center + box.width / 2.0) * image_width
    y2 = (box.y_center + box.height / 2.0) * image_height

    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def draw_boxes(
    image_path: Path,
    boxes: list[LabelBox],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Draw boxes and class names on one image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for visualization. Install pillow.") from exc

    with Image.open(image_path) as image:
        canvas = image.convert("RGB")

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    image_width, image_height = canvas.size

    for box in boxes:
        xyxy = box_to_pixels(box, image_width, image_height)
        if xyxy is None:
            continue

        class_name = class_names[box.class_id]
        x1, y1, x2, y2 = xyxy
        draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=2)
        text_position = (x1, max(0, y1 - 12))
        draw.text(text_position, class_name, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def print_summary(summary: Summary, output_dir: Path) -> None:
    """Print visualization summary."""
    print("YOLO label visualization summary:")
    print(f"- total_images: {summary.total_images}")
    print(f"- visualized_images: {summary.visualized_images}")
    print(f"- missing_label_files: {summary.missing_label_files}")
    print(f"- invalid_label_lines: {summary.invalid_label_lines}")
    print(f"- failed_images: {summary.failed_images}")
    print(f"- output_dir: {output_dir.as_posix()}")


def run(
    data_root: Path,
    split: str,
    classes_config: Path,
    output_dir: Path,
    limit: int | None,
    image_exts: set[str],
) -> int:
    """Run YOLO label visualization for one split."""
    image_dir = data_root / "images" / split
    label_dir = data_root / "labels" / split
    if not image_dir.is_dir():
        print(f"[ERROR] Missing image directory: {image_dir}")
        return 1
    if not label_dir.is_dir():
        print(f"[ERROR] Missing label directory: {label_dir}")
        return 1

    try:
        class_names = load_class_names(classes_config)
        image_paths = list_images(image_dir, image_exts, limit)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not class_names:
        print(f"[ERROR] No class names found in: {classes_config}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = Summary(total_images=len(image_paths))
    if not image_paths:
        print(f"[WARN] No supported images found in: {image_dir}")

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        try:
            boxes, invalid_lines, label_exists = parse_label_file(
                label_path, len(class_names)
            )
            if not label_exists:
                summary.missing_label_files += 1
            summary.invalid_label_lines += invalid_lines

            output_path = output_dir / f"{image_path.stem}.jpg"
            draw_boxes(image_path, boxes, class_names, output_path)
            summary.visualized_images += 1
        except (OSError, RuntimeError, ValueError) as exc:
            summary.failed_images += 1
            print(f"[WARN] Failed image skipped: {image_path} ({exc})")

    print_summary(summary, output_dir)
    return 0


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    return run(
        data_root=Path(args.data_root),
        split=args.split,
        classes_config=Path(args.classes_config),
        output_dir=Path(args.out_dir),
        limit=args.limit,
        image_exts=parse_image_exts(args.image_exts),
    )


if __name__ == "__main__":
    sys.exit(main())
