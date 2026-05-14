"""Command-line entry for Grounding DINO batch inference."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.open_vocab.grounding_dino_infer import (
    GroundingDinoPredictor,
    save_result_json,
    save_visualization,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_IMAGE_EXTS = "jpg,jpeg,png,bmp,tif,tiff"


@dataclass
class BatchSummary:
    """Grounding DINO batch inference counters."""

    total_images: int = 0
    processed_images: int = 0
    skipped_images: int = 0
    failed_images: int = 0
    total_boxes: int = 0
    output_dir: str = ""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Grounding DINO inference on all images in a directory."
    )
    parser.add_argument("--image-dir", required=True, help="Directory containing input images.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for per-image JSON results and visualization images.",
    )
    parser.add_argument("--prompt", required=True, help="Text prompt for open-vocabulary detection.")
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.35,
        help="Grounding DINO box threshold.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Grounding DINO text threshold.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, for example cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--config-file",
        required=True,
        help="Grounding DINO model config path.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Grounding DINO checkpoint path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process after sorting. Omit for all images.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an image when its output JSON already exists.",
    )
    parser.add_argument(
        "--image-exts",
        default=DEFAULT_IMAGE_EXTS,
        help="Comma-separated image extensions, for example jpg,jpeg,png.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure concise console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def validate_threshold(name: str, value: float) -> None:
    """Validate a threshold value."""
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


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


def collect_images(image_dir: Path, image_exts: set[str], limit: int | None) -> list[Path]:
    """Collect sorted image files from one directory."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if limit is not None and limit <= 0:
        raise ValueError(f"--limit must be positive when provided, got {limit}")

    images = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in image_exts
    )
    if limit is not None:
        return images[:limit]
    return images


def output_paths(image_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Build output visualization and JSON paths for one input image."""
    output_image = output_dir / f"{image_path.stem}_grounding_dino.jpg"
    output_json = output_dir / f"{image_path.stem}_grounding_dino.json"
    return output_image, output_json


def write_failure_log(output_dir: Path, failure_records: list[str]) -> None:
    """Write failed image records to a text file when any image fails."""
    if not failure_records:
        return
    failure_log = output_dir / "grounding_dino_batch_failures.txt"
    failure_log.write_text("\n".join(failure_records) + "\n", encoding="utf-8")
    LOGGER.info("Wrote failure log: %s", failure_log.as_posix())


def print_summary(summary: BatchSummary) -> None:
    """Print the required batch summary."""
    print("Grounding DINO batch inference summary:")
    print(f"- total_images: {summary.total_images}")
    print(f"- processed_images: {summary.processed_images}")
    print(f"- skipped_images: {summary.skipped_images}")
    print(f"- failed_images: {summary.failed_images}")
    print(f"- total_boxes: {summary.total_boxes}")
    print(f"- output_dir: {summary.output_dir}")


def run_batch(args: argparse.Namespace) -> BatchSummary:
    """Run Grounding DINO batch inference."""
    validate_threshold("--box-threshold", args.box_threshold)
    validate_threshold("--text-threshold", args.text_threshold)

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    config_file = Path(args.config_file)
    checkpoint = Path(args.checkpoint)
    image_exts = parse_image_exts(args.image_exts)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_images(image_dir, image_exts, args.limit)
    summary = BatchSummary(total_images=len(image_paths), output_dir=output_dir.as_posix())
    failure_records: list[str] = []

    LOGGER.info("Found %d image(s) in %s", len(image_paths), image_dir.as_posix())
    LOGGER.info("Output directory: %s", output_dir.as_posix())

    pending_images: list[Path] = []
    for image_path in image_paths:
        _, output_json = output_paths(image_path, output_dir)
        if args.skip_existing and output_json.is_file():
            summary.skipped_images += 1
            LOGGER.info("Skipping existing result: %s", output_json.as_posix())
            continue
        pending_images.append(image_path)

    if not pending_images:
        write_failure_log(output_dir, failure_records)
        return summary

    LOGGER.info("Loading Grounding DINO model once for batch inference.")
    predictor = GroundingDinoPredictor(
        config_file=config_file,
        checkpoint=checkpoint,
        device=args.device,
    )

    for index, image_path in enumerate(pending_images, start=1):
        output_image, output_json = output_paths(image_path, output_dir)
        LOGGER.info(
            "[%d/%d] Processing %s",
            index,
            len(pending_images),
            image_path.as_posix(),
        )

        try:
            result = predictor.predict(
                image_path=image_path,
                prompt=args.prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            save_visualization(image_path, result.detections, output_image)
            save_result_json(result, output_json)
        except Exception as exc:
            summary.failed_images += 1
            message = f"{image_path.as_posix()}: {exc}"
            failure_records.append(message)
            LOGGER.error("Failed image, continuing: %s", message)
            continue

        num_boxes = len(result.detections)
        summary.processed_images += 1
        summary.total_boxes += num_boxes
        LOGGER.info(
            "Saved %d box(es): %s and %s",
            num_boxes,
            output_json.as_posix(),
            output_image.as_posix(),
        )

    write_failure_log(output_dir, failure_records)
    return summary


def main() -> int:
    """Run the command-line program."""
    configure_logging()
    args = parse_args()
    try:
        summary = run_batch(args)
        print_summary(summary)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
