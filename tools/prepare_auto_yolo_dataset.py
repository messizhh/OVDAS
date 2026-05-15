"""Prepare YOLO datasets that use automatic train labels and manual val labels."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_IMAGE_EXTS = "jpg,jpeg,png,bmp,tif,tiff"


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


def count_txt_files(label_dir: Path) -> int:
    """Count YOLO txt label files in one directory."""
    return sum(1 for path in label_dir.iterdir() if path.is_file() and path.suffix == ".txt")


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


def copy_directory_contents(source_dir: Path, target_dir: Path) -> int:
    """Copy files from source_dir into target_dir, preserving relative paths."""
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
) -> tuple[int, int, int]:
    """Place one source directory at target_dir by symlink or copy."""
    if link_mode == "copy":
        return 0, 0, copy_directory_contents(source_dir, target_dir)
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

    require_dir(train_image_dir, "--train-image-dir")
    require_dir(train_label_dir, "--train-label-dir")
    require_dir(val_image_dir, "--val-image-dir")
    require_dir(val_label_dir, "--val-label-dir")

    train_images = collect_images(train_image_dir, image_exts)
    val_images = collect_images(val_image_dir, image_exts)
    created_empty_train_labels = ensure_train_labels(train_images, train_label_dir)
    validate_val_labels(val_images, val_label_dir)

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
        )
        created_links += created
        reused_existing_links += reused
        copied_files += copied

    names = load_class_names(Path(args.classes_config))
    write_yolo_config(config_path, out_root, names)

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
