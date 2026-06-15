"""Create a deterministic VisDrone train subset for OVDAS-Tile ablations."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.open_vocab.phrase_normalization import load_class_mapping
from tools.evaluate_auto_labels import read_image_size, yolo_to_xyxy


DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SIZE_BUCKETS = ("small", "medium", "large")
SMALL_AREA = 32 * 32
MEDIUM_AREA = 96 * 96


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create a fixed VisDrone experiment subset.")
    parser.add_argument(
        "--image-dir",
        default="data/processed/visdrone/images/train",
        help="VisDrone train image directory.",
    )
    parser.add_argument(
        "--label-dir",
        default="data/processed/visdrone/labels/train",
        help="VisDrone train YOLO label directory.",
    )
    parser.add_argument(
        "--classes-config",
        default="configs/classes_visdrone.yaml",
        help="Class config YAML.",
    )
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-list",
        default="outputs/experiment_subsets/visdrone_train_seed42_200.txt",
        help="Output txt file containing image file names.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/experiment_subsets/visdrone_train_seed42_200.json",
        help="Output JSON metadata file.",
    )
    parser.add_argument(
        "--stats-csv",
        default="results/tables/visdrone_train_seed42_200_subset_stats.csv",
        help="Output CSV with class and size statistics.",
    )
    return parser.parse_args()


def collect_images(image_dir: Path) -> list[Path]:
    """Collect supported images in deterministic order."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in DEFAULT_IMAGE_EXTS
    )


def choose_subset(images: list[Path], num_images: int, seed: int) -> list[Path]:
    """Choose a deterministic random subset and return it sorted by file name."""
    if num_images <= 0:
        raise ValueError("--num-images must be positive.")
    if len(images) < num_images:
        raise ValueError(f"Need {num_images} images, but only found {len(images)}.")
    rng = random.Random(seed)
    selected = rng.sample(images, num_images)
    return sorted(selected, key=lambda path: path.name)


def size_bucket(area: float) -> str:
    """Return COCO-style size bucket."""
    if area < SMALL_AREA:
        return "small"
    if area < MEDIUM_AREA:
        return "medium"
    return "large"


def parse_label_file(
    label_path: Path,
    image_width: int,
    image_height: int,
    valid_class_ids: set[int],
) -> tuple[list[tuple[int, float]], int]:
    """Parse one YOLO label file into class id and pixel area records."""
    if not label_path.is_file():
        return [], 0
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return [], 0

    records: list[tuple[int, float]] = []
    invalid_lines = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            invalid_lines += 1
            continue
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            invalid_lines += 1
            continue
        if class_id not in valid_class_ids:
            invalid_lines += 1
            continue
        xyxy = yolo_to_xyxy(x_center, y_center, width, height, image_width, image_height)
        if xyxy is None:
            invalid_lines += 1
            continue
        x1, y1, x2, y2 = xyxy
        records.append((class_id, (x2 - x1) * (y2 - y1)))
    return records, invalid_lines


def build_stats(
    selected_images: list[Path],
    label_dir: Path,
    id_to_name: dict[int, str],
) -> dict[str, Any]:
    """Build class and size statistics for selected images."""
    valid_class_ids = set(id_to_name)
    class_counts = {class_id: 0 for class_id in id_to_name}
    size_counts = {bucket: 0 for bucket in SIZE_BUCKETS}
    total_objects = 0
    missing_label_files = 0
    invalid_label_lines = 0

    for image_path in selected_images:
        image_width, image_height = read_image_size(image_path)
        label_path = label_dir / f"{image_path.stem}.txt"
        records, invalid_lines = parse_label_file(label_path, image_width, image_height, valid_class_ids)
        if not label_path.is_file():
            missing_label_files += 1
        invalid_label_lines += invalid_lines
        for class_id, area in records:
            class_counts[class_id] += 1
            size_counts[size_bucket(area)] += 1
            total_objects += 1

    return {
        "class_counts": class_counts,
        "size_counts": size_counts,
        "total_objects": total_objects,
        "missing_label_files": missing_label_files,
        "invalid_label_lines": invalid_label_lines,
    }


def write_outputs(
    selected_images: list[Path],
    image_dir: Path,
    label_dir: Path,
    id_to_name: dict[int, str],
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Write txt/json list files and statistics CSV."""
    output_list = Path(args.output_list)
    output_json = Path(args.output_json)
    stats_csv = Path(args.stats_csv)
    output_list.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    stats_csv.parent.mkdir(parents=True, exist_ok=True)

    image_names = [path.name for path in selected_images]
    output_list.write_text("\n".join(image_names) + "\n", encoding="utf-8")

    metadata = {
        "seed": args.seed,
        "num_images": args.num_images,
        "image_dir": image_dir.as_posix(),
        "label_dir": label_dir.as_posix(),
        "output_list": output_list.as_posix(),
        "stats_csv": stats_csv.as_posix(),
        "images": image_names,
        "statistics": stats,
    }
    output_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    with stats_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["scope", "class_id", "class_name", "size", "metric", "value"])
        writer.writerow(["summary", "", "", "", "image_count", len(selected_images)])
        writer.writerow(["summary", "", "", "", "total_objects", stats["total_objects"]])
        writer.writerow(["summary", "", "", "", "missing_label_files", stats["missing_label_files"]])
        writer.writerow(["summary", "", "", "", "invalid_label_lines", stats["invalid_label_lines"]])
        for class_id in sorted(id_to_name):
            writer.writerow(
                [
                    "class",
                    class_id,
                    id_to_name[class_id],
                    "",
                    "object_count",
                    stats["class_counts"].get(class_id, 0),
                ]
            )
        for bucket in SIZE_BUCKETS:
            writer.writerow(["size", "", "", bucket, "object_count", stats["size_counts"][bucket]])


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    try:
        image_dir = Path(args.image_dir)
        label_dir = Path(args.label_dir)
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Label directory does not exist: {label_dir}")
        _, id_to_name = load_class_mapping(Path(args.classes_config))
        images = collect_images(image_dir)
        selected_images = choose_subset(images, args.num_images, args.seed)
        stats = build_stats(selected_images, label_dir, id_to_name)
        write_outputs(selected_images, image_dir, label_dir, id_to_name, stats, args)
        print("Fixed experiment subset summary:")
        print(f"- image_count: {len(selected_images)}")
        print(f"- seed: {args.seed}")
        print(f"- output_list: {args.output_list}")
        print(f"- output_json: {args.output_json}")
        print(f"- stats_csv: {args.stats_csv}")
        print(f"- total_objects: {stats['total_objects']}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
