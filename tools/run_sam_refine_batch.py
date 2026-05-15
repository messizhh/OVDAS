"""Command-line entry for SAM refinement on a directory of images."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.sam_refine import (
    SamBoxRefiner,
    SamRefineOutput,
    build_empty_refine_output,
    load_dino_json,
    save_refine_visualization,
    save_refined_json,
    validate_refine_inputs,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_IMAGE_EXTS = "jpg,jpeg,png,bmp,tif,tiff"


@dataclass
class BatchSummary:
    """SAM refinement batch counters."""

    total_images: int = 0
    processed_images: int = 0
    skipped_images: int = 0
    failed_images: int = 0
    total_detections: int = 0
    refined_detections: int = 0
    output_dir: str = ""


@dataclass
class PendingRefineItem:
    """One image whose loaded detections need SAM refinement."""

    image_path: Path
    dino_json_path: Path
    output_json: Path
    vis_output: Path
    mask_output_dir: Path | None
    dino_data: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Refine Grounding DINO JSON files with SAM bbox prompts."
    )
    parser.add_argument("--image-dir", required=True, help="Directory containing input images.")
    parser.add_argument(
        "--dino-json-dir",
        required=True,
        help="Directory containing Day 4 Grounding DINO JSON files.",
    )
    parser.add_argument(
        "--output-json-dir",
        required=True,
        help="Directory for refined SAM JSON files.",
    )
    parser.add_argument(
        "--vis-output-dir",
        required=True,
        help="Directory for SAM before/after visualization images.",
    )
    parser.add_argument("--sam-checkpoint", required=True, help="SAM checkpoint path.")
    parser.add_argument("--model-type", default="vit_h", help="SAM model type, for example vit_h.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference device, for example cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to consider after sorting. Omit for all images.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an image when its refined output JSON already exists.",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Save per-detection binary mask PNG files.",
    )
    parser.add_argument(
        "--mask-output-dir",
        default=None,
        help="Directory for mask PNG files. Required when --save-mask is set.",
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

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in image_exts
    )
    if limit is not None:
        return image_paths[:limit]
    return image_paths


def dino_json_path_for_image(image_path: Path, dino_json_dir: Path) -> Path:
    """Resolve the Grounding DINO JSON path for one image."""
    preferred = dino_json_dir / f"{image_path.stem}_grounding_dino.json"
    if preferred.is_file():
        return preferred
    return dino_json_dir / f"{image_path.stem}.json"


def output_paths(
    image_path: Path,
    output_json_dir: Path,
    vis_output_dir: Path,
) -> tuple[Path, Path]:
    """Build output JSON and visualization paths for one image."""
    output_json = output_json_dir / f"{image_path.stem}_sam_refine.json"
    vis_output = vis_output_dir / f"{image_path.stem}_sam_refine.jpg"
    return output_json, vis_output


def mask_dir_for_image(
    image_path: Path,
    mask_output_root: Path | None,
) -> Path | None:
    """Build the optional mask output directory for one image."""
    if mask_output_root is None:
        return None
    return mask_output_root / image_path.stem


def write_failure_log(output_json_dir: Path, failure_records: list[str]) -> None:
    """Write failed image records to a text file when any image fails."""
    if not failure_records:
        return
    output_json_dir.mkdir(parents=True, exist_ok=True)
    failure_log = output_json_dir / "sam_refine_batch_failures.txt"
    failure_log.write_text("\n".join(failure_records) + "\n", encoding="utf-8")
    LOGGER.info("Wrote failure log: %s", failure_log.as_posix())


def print_summary(summary: BatchSummary) -> None:
    """Print the required batch summary."""
    print("SAM refine batch summary:")
    print(f"- total_images: {summary.total_images}")
    print(f"- processed_images: {summary.processed_images}")
    print(f"- skipped_images: {summary.skipped_images}")
    print(f"- failed_images: {summary.failed_images}")
    print(f"- total_detections: {summary.total_detections}")
    print(f"- refined_detections: {summary.refined_detections}")
    print(f"- output_dir: {summary.output_dir}")


def save_successful_output(
    output: SamRefineOutput,
    image_path: Path,
    output_json: Path,
    vis_output: Path,
    summary: BatchSummary,
) -> None:
    """Save refined JSON and visualization, then update summary counters."""
    save_refined_json(output.result, output_json)
    save_refine_visualization(
        image_path=image_path,
        detections=output.result["detections"],
        masks=output.masks,
        output_image=vis_output,
    )
    summary.processed_images += 1
    summary.total_detections += output.total_detections
    summary.refined_detections += output.refined_detections


def prepare_pending_items(
    image_paths: list[Path],
    args: argparse.Namespace,
    summary: BatchSummary,
    failure_records: list[str],
) -> list[PendingRefineItem]:
    """Validate pending images and immediately handle empty detections."""
    dino_json_dir = Path(args.dino_json_dir)
    output_json_dir = Path(args.output_json_dir)
    vis_output_dir = Path(args.vis_output_dir)
    checkpoint = Path(args.sam_checkpoint)
    mask_output_root = Path(args.mask_output_dir) if args.mask_output_dir else None

    pending_items: list[PendingRefineItem] = []
    for image_path in image_paths:
        dino_json_path = dino_json_path_for_image(image_path, dino_json_dir)
        output_json, vis_output = output_paths(image_path, output_json_dir, vis_output_dir)

        if args.skip_existing and output_json.is_file():
            summary.skipped_images += 1
            LOGGER.info("Skipping existing result: %s", output_json.as_posix())
            continue

        try:
            validate_refine_inputs(image_path, dino_json_path, checkpoint)
            dino_data = load_dino_json(dino_json_path)
        except Exception as exc:
            summary.failed_images += 1
            message = f"{image_path.as_posix()}: {exc}"
            failure_records.append(message)
            LOGGER.error("Failed to prepare image, continuing: %s", message)
            continue

        detections = dino_data.get("detections", [])
        if not detections:
            try:
                output = build_empty_refine_output(
                    image_path=image_path,
                    dino_json_path=dino_json_path,
                    dino_data=dino_data,
                    model_type=args.model_type,
                    checkpoint=checkpoint,
                    device=args.device,
                    save_mask=args.save_mask,
                )
                save_successful_output(
                    output=output,
                    image_path=image_path,
                    output_json=output_json,
                    vis_output=vis_output,
                    summary=summary,
                )
            except Exception as exc:
                summary.failed_images += 1
                message = f"{image_path.as_posix()}: {exc}"
                failure_records.append(message)
                LOGGER.error("Failed empty-detection image, continuing: %s", message)
            continue

        pending_items.append(
            PendingRefineItem(
                image_path=image_path,
                dino_json_path=dino_json_path,
                output_json=output_json,
                vis_output=vis_output,
                mask_output_dir=mask_dir_for_image(image_path, mask_output_root),
                dino_data=dino_data,
            )
        )

    return pending_items


def run_batch(args: argparse.Namespace) -> BatchSummary:
    """Run SAM refinement for all images in a directory."""
    image_dir = Path(args.image_dir)
    output_json_dir = Path(args.output_json_dir)
    vis_output_dir = Path(args.vis_output_dir)
    image_exts = parse_image_exts(args.image_exts)

    if args.save_mask and not args.mask_output_dir:
        raise ValueError("--mask-output-dir is required when --save-mask is set.")

    output_json_dir.mkdir(parents=True, exist_ok=True)
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(image_dir, image_exts, args.limit)
    summary = BatchSummary(total_images=len(image_paths), output_dir=output_json_dir.as_posix())
    failure_records: list[str] = []

    LOGGER.info("Found %d image(s) in %s", len(image_paths), image_dir.as_posix())
    LOGGER.info("Refined JSON output directory: %s", output_json_dir.as_posix())
    LOGGER.info("Visualization output directory: %s", vis_output_dir.as_posix())

    pending_items = prepare_pending_items(
        image_paths=image_paths,
        args=args,
        summary=summary,
        failure_records=failure_records,
    )

    if not pending_items:
        write_failure_log(output_json_dir, failure_records)
        return summary

    LOGGER.info("Loading SAM model once for %d image(s).", len(pending_items))
    refiner = SamBoxRefiner(
        checkpoint=Path(args.sam_checkpoint),
        model_type=args.model_type,
        device=args.device,
    )

    for index, item in enumerate(pending_items, start=1):
        LOGGER.info(
            "[%d/%d] Refining %s",
            index,
            len(pending_items),
            item.image_path.as_posix(),
        )
        try:
            output = refiner.refine_loaded_json(
                image_path=item.image_path,
                dino_json_path=item.dino_json_path,
                dino_data=item.dino_data,
                save_mask=args.save_mask,
                mask_output_dir=item.mask_output_dir,
            )
            save_successful_output(
                output=output,
                image_path=item.image_path,
                output_json=item.output_json,
                vis_output=item.vis_output,
                summary=summary,
            )
        except Exception as exc:
            summary.failed_images += 1
            message = f"{item.image_path.as_posix()}: {exc}"
            failure_records.append(message)
            LOGGER.error("Failed image, continuing: %s", message)
            continue

        LOGGER.info(
            "Saved %d/%d refined detection(s): %s",
            output.refined_detections,
            output.total_detections,
            item.output_json.as_posix(),
        )

    write_failure_log(output_json_dir, failure_records)
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
