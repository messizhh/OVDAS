"""Generate a lightweight manifest for local sample images and labels."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_EXTENSION = ".txt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="List sample images and matching label files."
    )
    parser.add_argument(
        "--image-dir",
        default="data/samples/images",
        help="Directory containing sample images.",
    )
    parser.add_argument(
        "--label-dir",
        default="data/samples/labels",
        help="Directory containing sample label files.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/tables/sample_manifest.csv",
        help="Output CSV path for the sample manifest.",
    )
    return parser.parse_args()


def validate_directory(path: Path, name: str) -> list[str]:
    """Return an error if a required directory is missing."""
    if not path.exists():
        return [f"{name} directory does not exist: {path}"]
    if not path.is_dir():
        return [f"{name} path is not a directory: {path}"]
    return []


def list_files_by_extension(directory: Path, extensions: set[str]) -> list[Path]:
    """List files in a directory whose suffix matches the allowed extensions."""
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ],
        key=lambda path: path.name.lower(),
    )


def build_manifest_rows(
    image_paths: list[Path],
    labels_by_stem: dict[str, Path],
) -> list[dict[str, str]]:
    """Build CSV rows for sample images and their optional matching labels."""
    rows: list[dict[str, str]] = []
    for image_path in image_paths:
        label_path = labels_by_stem.get(image_path.stem)
        rows.append(
            {
                "image_name": image_path.name,
                "image_path": image_path.as_posix(),
                "label_name": label_path.name if label_path else "",
                "label_path": label_path.as_posix() if label_path else "",
                "has_label": "true" if label_path else "false",
            }
        )
    return rows


def write_manifest_csv(output_csv: Path, rows: list[dict[str, str]]) -> None:
    """Write manifest rows to a CSV file, creating the parent directory."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_name", "image_path", "label_name", "label_path", "has_label"]
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    image_count: int,
    label_count: int,
    matched_count: int,
    missing_label_count: int,
    extra_label_count: int,
    output_csv: Path,
) -> None:
    """Print a clear terminal summary for the sample manifest."""
    print("Sample manifest summary:")
    print(f"- image_count: {image_count}")
    print(f"- label_count: {label_count}")
    print(f"- matched_count: {matched_count}")
    print(f"- missing_label_count: {missing_label_count}")
    print(f"- extra_label_count: {extra_label_count}")
    print(f"- output_csv: {output_csv.as_posix()}")


def run(image_dir: Path, label_dir: Path, output_csv: Path) -> int:
    """Generate the sample manifest and return a process exit code."""
    errors: list[str] = []
    errors.extend(validate_directory(image_dir, "Image"))
    errors.extend(validate_directory(label_dir, "Label"))
    if errors:
        print("Sample manifest generation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    image_paths = list_files_by_extension(image_dir, IMAGE_EXTENSIONS)
    label_paths = list_files_by_extension(label_dir, {LABEL_EXTENSION})
    labels_by_stem = {path.stem: path for path in label_paths}
    image_stems = {path.stem for path in image_paths}

    rows = build_manifest_rows(image_paths, labels_by_stem)
    write_manifest_csv(output_csv, rows)

    matched_count = sum(1 for row in rows if row["has_label"] == "true")
    missing_label_count = len(image_paths) - matched_count
    extra_label_count = sum(1 for path in label_paths if path.stem not in image_stems)

    if not image_paths:
        print("No sample images found. Empty manifest CSV was generated.")

    print_summary(
        image_count=len(image_paths),
        label_count=len(label_paths),
        matched_count=matched_count,
        missing_label_count=missing_label_count,
        extra_label_count=extra_label_count,
        output_csv=output_csv,
    )
    return 0


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    return run(
        image_dir=Path(args.image_dir),
        label_dir=Path(args.label_dir),
        output_csv=Path(args.output_csv),
    )


if __name__ == "__main__":
    sys.exit(main())
