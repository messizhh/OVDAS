"""Prepare YOLO datasets that use automatic train labels and manual val labels."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_IMAGE_EXTS = "jpg,jpeg,png,bmp,tif,tiff"
DEFAULT_MARKER_NAME = ".ovdas_auto_yolo_dataset"
DEFAULT_MARKER_DATASET_ID = "auto_yolo_dataset"
VALID_MARKER_STATUSES = {"building", "complete"}


@dataclass
class PrepareSummary:
    """Summary counters for one prepared YOLO dataset."""

    out_root: str
    config_path: str
    link_mode: str
    train_images: int
    train_labels: int
    val_images: int
    val_labels: int
    created_empty_train_labels: int
    reused_existing_links: int
    created_links: int
    copied_files: int
    train_empty_label_files: int = 0
    train_total_boxes: int = 0
    train_class_counts: dict[int, int] = field(default_factory=dict)
    val_empty_label_files: int = 0
    val_total_boxes: int = 0
    val_class_counts: dict[int, int] = field(default_factory=dict)


@dataclass
class LabelValidationStats:
    """YOLO label counters collected during strict validation."""

    label_files: int
    empty_label_files: int
    total_boxes: int
    class_counts: dict[int, int] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare a YOLO dataset from auto train labels and manual val labels."
    )
    parser.add_argument("--train-image-dir", required=True, help="Source train image directory.")
    parser.add_argument("--train-label-dir", required=True, help="Source automatic train label directory.")
    parser.add_argument("--val-image-dir", required=True, help="Source validation image directory.")
    parser.add_argument("--val-label-dir", required=True, help="Source manual validation label directory.")
    parser.add_argument("--out-root", required=True, help="Output YOLO dataset root.")
    parser.add_argument("--config-path", required=True, help="YOLO data YAML path to write.")
    parser.add_argument(
        "--classes-config",
        default="configs/classes_visdrone.yaml",
        help="Class config YAML used to write YOLO names.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Use directory symlinks or copy files into the output dataset.",
    )
    parser.add_argument(
        "--replace-existing-links",
        action="store_true",
        help="Replace existing symlinks that point to a different directory.",
    )
    parser.add_argument(
        "--image-exts",
        default=DEFAULT_IMAGE_EXTS,
        help="Comma-separated image extensions, for example jpg,jpeg,png.",
    )
    parser.add_argument("--expected-train-images", type=int, help="Required train image count.")
    parser.add_argument("--expected-train-labels", type=int, help="Required train label count.")
    parser.add_argument("--expected-val-images", type=int, help="Required validation image count.")
    parser.add_argument("--expected-val-labels", type=int, help="Required validation label count.")
    parser.add_argument(
        "--strict-existing-labels",
        action="store_true",
        help="Require exact image/label stem alignment and never create missing train labels.",
    )
    parser.add_argument(
        "--forbid-directory-symlinks",
        action="store_true",
        help="Fail if an output split directory would be a directory symlink.",
    )
    parser.add_argument(
        "--rebuild-output-root",
        action="store_true",
        help="Delete and rebuild a previously marked output root after strict source validation.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run strict source validation and print statistics without writing outputs.",
    )
    parser.add_argument(
        "--marker-name",
        default=DEFAULT_MARKER_NAME,
        help="Marker file used to identify output roots created by this tool.",
    )
    parser.add_argument(
        "--marker-dataset-id",
        default=DEFAULT_MARKER_DATASET_ID,
        help="Dataset id written to and required from the output marker.",
    )
    return parser.parse_args()


def parse_image_exts(raw_exts: str) -> set[str]:
    """Parse a comma-separated extension list into normalized suffixes."""
    exts: set[str] = set()
    for item in raw_exts.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        exts.add(ext)
    if not exts:
        raise ValueError("--image-exts must contain at least one extension.")
    return exts


def require_dir(path: Path, name: str) -> None:
    """Raise a clear error when a required directory is missing."""
    if not path.is_dir():
        raise FileNotFoundError(f"{name} does not exist or is not a directory: {path}")


def collect_images(image_dir: Path, image_exts: set[str]) -> list[Path]:
    """Collect sorted image files from one directory."""
    return sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in image_exts
    )


def collect_txt_files(label_dir: Path) -> list[Path]:
    """Collect sorted YOLO txt label files from one directory."""
    return sorted(path for path in label_dir.iterdir() if path.is_file() and path.suffix == ".txt")


def count_txt_files(label_dir: Path) -> int:
    """Count YOLO txt label files in one directory."""
    return len(collect_txt_files(label_dir))


def require_count(actual: int, expected: int | None, label: str) -> None:
    """Require an exact count when expected is configured."""
    if expected is None or actual == expected:
        return
    raise ValueError(f"Expected {expected} {label}, found {actual}.")


def ensure_train_labels(train_images: list[Path], train_label_dir: Path) -> int:
    """Ensure every train image has a YOLO txt file, creating empty files when needed."""
    train_label_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for image_path in train_images:
        label_path = train_label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            continue
        label_path.write_text("", encoding="utf-8")
        created += 1
    return created


def validate_val_labels(val_images: list[Path], val_label_dir: Path) -> None:
    """Validate that manual validation labels are aligned with validation images."""
    missing = [image_path.stem for image_path in val_images if not (val_label_dir / f"{image_path.stem}.txt").is_file()]
    if not missing:
        return
    preview = ", ".join(missing[:5])
    raise FileNotFoundError(
        f"Missing {len(missing)} validation label file(s) in {val_label_dir}. "
        f"First missing stems: {preview}"
    )


def validate_label_stems(split: str, images: list[Path], label_files: list[Path]) -> None:
    """Validate that images and labels have the same stem set."""
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in label_files}
    missing = sorted(image_stems - label_stems)
    extra = sorted(label_stems - image_stems)
    errors: list[str] = []
    if missing:
        preview = ", ".join(missing[:5])
        errors.append(f"Missing {len(missing)} {split} label file(s). First missing stems: {preview}")
    if extra:
        preview = ", ".join(extra[:5])
        errors.append(f"Extra {len(extra)} {split} label file(s). First extra stems: {preview}")
    if errors:
        raise FileNotFoundError("; ".join(errors))


def validate_yolo_label_file(path: Path, valid_class_ids: set[int]) -> LabelValidationStats:
    """Validate one YOLO label file and collect counters."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return LabelValidationStats(label_files=1, empty_label_files=1, total_boxes=0)

    total_boxes = 0
    class_counts = {class_id: 0 for class_id in sorted(valid_class_ids)}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        parts = raw_line.split()
        location = f"{path}:{line_number}"
        if len(parts) != 5:
            raise ValueError(f"{location}: expected 5 columns, got {len(parts)}.")
        if not parts[0].lstrip("-").isdigit():
            raise ValueError(f"{location}: class_id must be an integer, got '{parts[0]}'.")
        class_id = int(parts[0])
        if class_id not in valid_class_ids:
            raise ValueError(
                f"{location}: class_id {class_id} is outside valid range "
                f"{min(valid_class_ids)}..{max(valid_class_ids)}."
            )
        try:
            cx, cy, width, height = (float(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{location}: bbox values must be numeric.") from exc
        values = (cx, cy, width, height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{location}: bbox values must be finite.")
        if not 0.0 <= cx <= 1.0:
            raise ValueError(f"{location}: cx must be in [0, 1], got {cx}.")
        if not 0.0 <= cy <= 1.0:
            raise ValueError(f"{location}: cy must be in [0, 1], got {cy}.")
        if not 0.0 < width <= 1.0:
            raise ValueError(f"{location}: w must be in (0, 1], got {width}.")
        if not 0.0 < height <= 1.0:
            raise ValueError(f"{location}: h must be in (0, 1], got {height}.")
        total_boxes += 1
        class_counts[class_id] += 1

    return LabelValidationStats(
        label_files=1,
        empty_label_files=0,
        total_boxes=total_boxes,
        class_counts=class_counts,
    )


def validate_yolo_labels(label_files: list[Path], valid_class_ids: set[int]) -> LabelValidationStats:
    """Validate YOLO label files and aggregate counters."""
    aggregate = LabelValidationStats(
        label_files=len(label_files),
        empty_label_files=0,
        total_boxes=0,
        class_counts={class_id: 0 for class_id in sorted(valid_class_ids)},
    )
    for path in label_files:
        stats = validate_yolo_label_file(path, valid_class_ids)
        aggregate.empty_label_files += stats.empty_label_files
        aggregate.total_boxes += stats.total_boxes
        for class_id, count in stats.class_counts.items():
            aggregate.class_counts[class_id] = aggregate.class_counts.get(class_id, 0) + count
    return aggregate


def read_output_marker(marker_path: Path, expected_dataset_id: str) -> dict[str, str]:
    """Read and validate a dataset output marker."""
    values: dict[str, str] = {}
    for line in marker_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "=" not in line:
            raise ValueError(f"Invalid marker line in {marker_path}: {line}")
        key, value = line.split("=", 1)
        if not key or not value:
            raise ValueError(f"Invalid marker line in {marker_path}: {line}")
        values[key] = value

    dataset_id = values.get("dataset_id")
    status = values.get("status")
    if dataset_id != expected_dataset_id:
        raise ValueError(
            f"Invalid marker dataset_id in {marker_path}: expected {expected_dataset_id}, got {dataset_id}"
        )
    if status not in VALID_MARKER_STATUSES:
        raise ValueError(f"Invalid marker status in {marker_path}: {status}")
    return values


def prepare_output_root(
    out_root: Path,
    rebuild_output_root: bool,
    marker_name: str,
    marker_dataset_id: str,
) -> None:
    """Validate or rebuild an output root without silently overwriting unknown directories."""
    marker_path = out_root / marker_name
    if out_root.is_symlink():
        raise FileExistsError(f"Refusing to use symlink output root: {out_root}")
    if not out_root.exists():
        return
    if not out_root.is_dir():
        raise FileExistsError(f"Refusing to replace non-directory output root: {out_root}")

    has_contents = any(out_root.iterdir())
    if rebuild_output_root:
        if not marker_path.is_file():
            raise FileExistsError(
                f"Refusing to rebuild unmarked output root: {out_root}. "
                f"Expected marker file: {marker_path}"
            )
        read_output_marker(marker_path, marker_dataset_id)
        shutil.rmtree(out_root)
        return
    if has_contents:
        raise FileExistsError(
            f"Refusing to write into existing output root: {out_root}. "
            "Use --rebuild-output-root for a previously marked dataset root, or remove it manually."
        )


def write_output_marker(out_root: Path, marker_name: str, marker_dataset_id: str, status: str) -> None:
    """Mark an output root as created by this dataset preparation tool."""
    if status not in VALID_MARKER_STATUSES:
        raise ValueError(f"Invalid marker status: {status}")
    out_root.mkdir(parents=True, exist_ok=True)
    marker_path = out_root / marker_name
    marker_path.write_text(
        "created_by=tools/prepare_auto_yolo_dataset.py\n"
        f"dataset_id={marker_dataset_id}\n"
        f"status={status}\n",
        encoding="utf-8",
    )


def symlink_matches(link_path: Path, source_dir: Path) -> bool:
    """Return whether an existing symlink points to the requested source directory."""
    try:
        return link_path.is_symlink() and link_path.resolve() == source_dir.resolve()
    except FileNotFoundError:
        return False


def link_directory(
    source_dir: Path,
    link_path: Path,
    replace_existing_links: bool,
) -> tuple[int, int]:
    """Create or reuse one directory symlink and return created/reused counters."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if symlink_matches(link_path, source_dir):
        return 0, 1

    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and replace_existing_links:
            link_path.unlink()
        else:
            raise FileExistsError(
                f"Refusing to replace existing path: {link_path}. "
                "Remove it manually or use --link-mode copy if you need real directories."
            )

    relative_target = os.path.relpath(source_dir.resolve(), start=link_path.parent.resolve())
    link_path.symlink_to(relative_target, target_is_directory=True)
    return 1, 0


def copy_directory_contents(
    source_dir: Path,
    target_dir: Path,
    forbid_directory_symlinks: bool = False,
) -> int:
    """Copy files from source_dir into target_dir, preserving relative paths."""
    if forbid_directory_symlinks and target_dir.is_symlink():
        raise FileExistsError(f"Refusing to copy into directory symlink: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source_path in sorted(source_dir.rglob("*")):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(source_dir)
        target_path = target_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied += 1
    return copied


def place_directory(
    source_dir: Path,
    target_dir: Path,
    link_mode: str,
    replace_existing_links: bool,
    forbid_directory_symlinks: bool = False,
) -> tuple[int, int, int]:
    """Place one source directory at target_dir by symlink or copy."""
    if forbid_directory_symlinks and target_dir.is_symlink():
        raise FileExistsError(f"Refusing to use directory symlink: {target_dir}")
    if link_mode == "copy":
        return 0, 0, copy_directory_contents(source_dir, target_dir, forbid_directory_symlinks)
    created, reused = link_directory(source_dir, target_dir, replace_existing_links)
    return created, reused, 0


def load_class_names(classes_config: Path) -> dict[int, str]:
    """Load class names from the project classes YAML."""
    with classes_config.open("r", encoding="utf-8") as file:
        data: Any = yaml.safe_load(file)
    if not isinstance(data, dict) or not isinstance(data.get("classes"), list):
        raise ValueError(f"Invalid classes config: {classes_config}")

    names: dict[int, str] = {}
    for index, item in enumerate(data["classes"]):
        if isinstance(item, str):
            names[index] = item
            continue
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            raw_id = item.get("id", index)
            if not isinstance(raw_id, int):
                raise ValueError(f"Invalid class id in {classes_config}: {raw_id}")
            names[raw_id] = item["name"]
            continue
        raise ValueError(f"Invalid class entry in {classes_config}: {item}")

    if not names:
        raise ValueError(f"No class names found in {classes_config}")
    return names


def write_yolo_config(config_path: Path, out_root: Path, names: dict[int, str]) -> None:
    """Write a YOLO data config using project-root-relative paths."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "path": out_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {class_id: names[class_id] for class_id in sorted(names)},
    }
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=False, sort_keys=False)


def prepare_dataset(args: argparse.Namespace) -> PrepareSummary:
    """Prepare one auto-label YOLO dataset."""
    train_image_dir = Path(args.train_image_dir)
    train_label_dir = Path(args.train_label_dir)
    val_image_dir = Path(args.val_image_dir)
    val_label_dir = Path(args.val_label_dir)
    out_root = Path(args.out_root)
    config_path = Path(args.config_path)
    image_exts = parse_image_exts(args.image_exts)
    names = load_class_names(Path(args.classes_config))
    valid_class_ids = set(names)

    require_dir(train_image_dir, "--train-image-dir")
    require_dir(train_label_dir, "--train-label-dir")
    require_dir(val_image_dir, "--val-image-dir")
    require_dir(val_label_dir, "--val-label-dir")

    train_images = collect_images(train_image_dir, image_exts)
    val_images = collect_images(val_image_dir, image_exts)
    train_label_files = collect_txt_files(train_label_dir)
    val_label_files = collect_txt_files(val_label_dir)

    require_count(len(train_images), args.expected_train_images, "train images")
    require_count(len(train_label_files), args.expected_train_labels, "train labels")
    require_count(len(val_images), args.expected_val_images, "val images")
    require_count(len(val_label_files), args.expected_val_labels, "val labels")

    strict_validation = bool(args.strict_existing_labels or args.validate_only)
    created_empty_train_labels = 0
    train_label_stats = LabelValidationStats(0, 0, 0, {class_id: 0 for class_id in sorted(valid_class_ids)})
    val_label_stats = LabelValidationStats(0, 0, 0, {class_id: 0 for class_id in sorted(valid_class_ids)})
    if strict_validation:
        validate_label_stems("train", train_images, train_label_files)
        validate_label_stems("val", val_images, val_label_files)
        train_label_stats = validate_yolo_labels(train_label_files, valid_class_ids)
        val_label_stats = validate_yolo_labels(val_label_files, valid_class_ids)
    else:
        created_empty_train_labels = ensure_train_labels(train_images, train_label_dir)
        validate_val_labels(val_images, val_label_dir)

    if args.validate_only:
        return PrepareSummary(
            out_root=out_root.as_posix(),
            config_path=config_path.as_posix(),
            link_mode=args.link_mode,
            train_images=len(train_images),
            train_labels=len(train_label_files),
            val_images=len(val_images),
            val_labels=len(val_label_files),
            created_empty_train_labels=created_empty_train_labels,
            reused_existing_links=0,
            created_links=0,
            copied_files=0,
            train_empty_label_files=train_label_stats.empty_label_files,
            train_total_boxes=train_label_stats.total_boxes,
            train_class_counts=train_label_stats.class_counts,
            val_empty_label_files=val_label_stats.empty_label_files,
            val_total_boxes=val_label_stats.total_boxes,
            val_class_counts=val_label_stats.class_counts,
        )

    prepare_output_root(out_root, args.rebuild_output_root, args.marker_name, args.marker_dataset_id)
    use_marker = bool(args.strict_existing_labels or args.rebuild_output_root)
    if use_marker:
        write_output_marker(out_root, args.marker_name, args.marker_dataset_id, "building")

    placements = [
        (train_image_dir, out_root / "images" / "train"),
        (val_image_dir, out_root / "images" / "val"),
        (train_label_dir, out_root / "labels" / "train"),
        (val_label_dir, out_root / "labels" / "val"),
    ]
    created_links = 0
    reused_existing_links = 0
    copied_files = 0
    for source_dir, target_dir in placements:
        created, reused, copied = place_directory(
            source_dir=source_dir,
            target_dir=target_dir,
            link_mode=args.link_mode,
            replace_existing_links=args.replace_existing_links,
            forbid_directory_symlinks=args.forbid_directory_symlinks,
        )
        created_links += created
        reused_existing_links += reused
        copied_files += copied

    write_yolo_config(config_path, out_root, names)
    if use_marker:
        write_output_marker(out_root, args.marker_name, args.marker_dataset_id, "complete")

    return PrepareSummary(
        out_root=out_root.as_posix(),
        config_path=config_path.as_posix(),
        link_mode=args.link_mode,
        train_images=len(train_images),
        train_labels=count_txt_files(train_label_dir),
        val_images=len(val_images),
        val_labels=count_txt_files(val_label_dir),
        created_empty_train_labels=created_empty_train_labels,
        reused_existing_links=reused_existing_links,
        created_links=created_links,
        copied_files=copied_files,
        train_empty_label_files=train_label_stats.empty_label_files,
        train_total_boxes=train_label_stats.total_boxes,
        train_class_counts=train_label_stats.class_counts,
        val_empty_label_files=val_label_stats.empty_label_files,
        val_total_boxes=val_label_stats.total_boxes,
        val_class_counts=val_label_stats.class_counts,
    )


def print_summary(summary: PrepareSummary) -> None:
    """Print a concise preparation summary."""
    print("Auto YOLO dataset preparation summary:")
    print(f"- out_root: {summary.out_root}")
    print(f"- config_path: {summary.config_path}")
    print(f"- link_mode: {summary.link_mode}")
    print(f"- train_images: {summary.train_images}")
    print(f"- train_labels: {summary.train_labels}")
    print(f"- val_images: {summary.val_images}")
    print(f"- val_labels: {summary.val_labels}")
    print(f"- created_empty_train_labels: {summary.created_empty_train_labels}")
    print(f"- created_links: {summary.created_links}")
    print(f"- reused_existing_links: {summary.reused_existing_links}")
    print(f"- copied_files: {summary.copied_files}")
    print(f"- train_empty_label_files: {summary.train_empty_label_files}")
    print(f"- train_total_boxes: {summary.train_total_boxes}")
    for class_id, count in sorted(summary.train_class_counts.items()):
        print(f"  train class_{class_id}: {count}")
    print(f"- val_empty_label_files: {summary.val_empty_label_files}")
    print(f"- val_total_boxes: {summary.val_total_boxes}")
    for class_id, count in sorted(summary.val_class_counts.items()):
        print(f"  val class_{class_id}: {count}")


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    try:
        summary = prepare_dataset(args)
        print_summary(summary)
        return 0
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
