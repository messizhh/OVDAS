"""Command-line entry for OVDAS-Tile Grounding DINO batch inference."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.open_vocab.grounding_dino_infer import GroundingDinoPredictor, read_image_size
from src.open_vocab.phrase_normalization import load_class_mapping
from src.open_vocab.tiled_inference import (
    Tile,
    build_tiled_result,
    class_aware_nms,
    generate_tiles,
    map_tile_bbox_to_image,
    normalize_detection_for_merge,
    save_tiled_result_json,
)
from src.utils.image_lists import read_image_list, resolve_image_entries

LOGGER = logging.getLogger(__name__)
DEFAULT_IMAGE_EXTS = "jpg,jpeg,png,bmp,tif,tiff"


@dataclass
class TiledBatchSummary:
    """OVDAS-Tile inference counters."""

    total_images: int = 0
    processed_images: int = 0
    skipped_images: int = 0
    failed_images: int = 0
    raw_boxes: int = 0
    merged_boxes: int = 0
    output_dir: str = ""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run full-image plus tiled Grounding DINO inference for OVDAS-Tile."
    )
    parser.add_argument("--image-dir", required=True, help="Directory containing input images.")
    parser.add_argument(
        "--image-list",
        default=None,
        help="Optional txt/json image list. Entries may be image names or paths.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for per-image tiled Grounding DINO JSON and visualization files.",
    )
    parser.add_argument("--prompt", required=True, help="Text prompt for open-vocabulary detection.")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap-ratio", type=float, default=0.20)
    parser.add_argument(
        "--include-full-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run Grounding DINO on the original full image.",
    )
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config-file", required=True, help="Grounding DINO model config path.")
    parser.add_argument("--checkpoint", required=True, help="Grounding DINO checkpoint path.")
    parser.add_argument(
        "--classes-config",
        default="configs/classes_visdrone.yaml",
        help="Class config used before class-aware NMS.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process after sorting or image-list resolution.",
    )
    parser.add_argument(
        "--image-exts",
        default=DEFAULT_IMAGE_EXTS,
        help="Comma-separated image extensions, for example jpg,jpeg,png.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    """Configure concise console logging."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_image_exts(raw_exts: str) -> list[str]:
    """Parse a comma-separated extension list into normalized suffixes."""
    exts: list[str] = []
    seen: set[str] = set()
    for item in raw_exts.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext in seen:
            continue
        exts.append(ext)
        seen.add(ext)
    if not exts:
        raise ValueError("--image-exts must contain at least one extension.")
    return exts


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line values."""
    for name in ("box_threshold", "text_threshold", "merge_iou"):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive.")
    if args.overlap_ratio < 0.0 or args.overlap_ratio >= 1.0:
        raise ValueError("--overlap-ratio must be in [0, 1).")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive when provided.")


def collect_images(
    image_dir: Path,
    image_exts: list[str],
    image_list: Path | None,
    max_images: int | None,
) -> list[Path]:
    """Collect images either from a fixed list or from a directory."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if image_list is not None:
        images = resolve_image_entries(image_dir, read_image_list(image_list), image_exts)
    else:
        image_ext_set = set(image_exts)
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_ext_set
        )
    if max_images is not None:
        return images[:max_images]
    return images


def output_paths(image_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Build output paths compatible with existing downstream tools."""
    output_image = output_dir / f"{image_path.stem}_grounding_dino.jpg"
    output_json = output_dir / f"{image_path.stem}_grounding_dino.json"
    return output_image, output_json


def detection_to_dict(value: Any) -> dict[str, Any]:
    """Convert a dataclass or mapping detection to a plain dict."""
    if isinstance(value, dict):
        return dict(value)
    return asdict(value)


def normalize_full_detections(
    detections: list[Any],
    image_width: int,
    image_height: int,
    phrase_to_id: dict[str, int],
    id_to_name: dict[int, str],
) -> list[dict[str, Any]]:
    """Normalize full-image detections for merge."""
    normalized: list[dict[str, Any]] = []
    for detection in detections:
        item = normalize_detection_for_merge(
            detection=detection_to_dict(detection),
            image_width=image_width,
            image_height=image_height,
            phrase_to_id=phrase_to_id,
            id_to_name=id_to_name,
            source="full",
            tile=None,
        )
        if item is not None:
            normalized.append(item)
    return normalized


def normalize_tile_detections(
    detections: list[Any],
    tile: Tile,
    image_width: int,
    image_height: int,
    phrase_to_id: dict[str, int],
    id_to_name: dict[int, str],
) -> list[dict[str, Any]]:
    """Map tile detections to original coordinates and normalize for merge."""
    normalized: list[dict[str, Any]] = []
    for detection in detections:
        item = detection_to_dict(detection)
        mapped_bbox = map_tile_bbox_to_image(
            item.get("bbox_xyxy"),
            tile.xyxy,
            image_width=image_width,
            image_height=image_height,
        )
        if mapped_bbox is None:
            continue
        item["bbox_xyxy"] = mapped_bbox
        merged_item = normalize_detection_for_merge(
            detection=item,
            image_width=image_width,
            image_height=image_height,
            phrase_to_id=phrase_to_id,
            id_to_name=id_to_name,
            source="tile",
            tile=tile,
        )
        if merged_item is not None:
            normalized.append(merged_item)
    return normalized


def save_tiled_visualization(
    image_path: Path,
    detections: list[dict[str, Any]],
    output_image: Path,
) -> None:
    """Save a lightweight visualization of merged tiled detections."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required to save visualization images.") from exc

    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    output_image.parent.mkdir(parents=True, exist_ok=True)

    for detection in detections:
        bbox = detection.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        color = (64, 160, 255) if detection.get("source") == "tile" else (255, 64, 64)
        label = f"{detection.get('phrase', 'object')} {float(detection.get('score', 0.0)):.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1, max(0.0, y1 - 12.0)), label, fill=(255, 255, 255), font=font)
    canvas.save(output_image, quality=95)


def run_tiled_image(
    image_path: Path,
    output_json: Path,
    output_image: Path,
    predictor: GroundingDinoPredictor,
    args: argparse.Namespace,
    phrase_to_id: dict[str, int],
    id_to_name: dict[int, str],
) -> tuple[int, int]:
    """Run tiled inference for one image and return raw/merged detection counts."""
    image_width, image_height = read_image_size(image_path)
    normalized_detections: list[dict[str, Any]] = []
    full_count = 0
    tile_count = 0
    started_at = time.perf_counter()

    if args.include_full_image:
        full_result = predictor.predict(
            image_path=image_path,
            prompt=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        full_count = len(full_result.detections)
        normalized_detections.extend(
            normalize_full_detections(
                detections=full_result.detections,
                image_width=image_width,
                image_height=image_height,
                phrase_to_id=phrase_to_id,
                id_to_name=id_to_name,
            )
        )

    tiles = generate_tiles(
        image_width=image_width,
        image_height=image_height,
        tile_size=args.tile_size,
        overlap_ratio=args.overlap_ratio,
    )

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for tiled inference crops.") from exc

    with tempfile.TemporaryDirectory(prefix="ovdas_tile_") as temp_root:
        temp_dir = Path(temp_root)
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            for tile in tiles:
                tile_image = rgb_image.crop(tile.xyxy)
                tile_path = temp_dir / f"{image_path.stem}_tile_{tile.index:04d}.jpg"
                tile_image.save(tile_path, quality=95)

                tile_result = predictor.predict(
                    image_path=tile_path,
                    prompt=args.prompt,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                )
                tile_count += len(tile_result.detections)
                normalized_detections.extend(
                    normalize_tile_detections(
                        detections=tile_result.detections,
                        tile=tile,
                        image_width=image_width,
                        image_height=image_height,
                        phrase_to_id=phrase_to_id,
                        id_to_name=id_to_name,
                    )
                )

    merged_detections = class_aware_nms(normalized_detections, iou_threshold=args.merge_iou)
    inference_time = time.perf_counter() - started_at
    result = build_tiled_result(
        image_path=image_path,
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        image_width=image_width,
        image_height=image_height,
        tile_size=args.tile_size,
        overlap_ratio=args.overlap_ratio,
        include_full_image=args.include_full_image,
        merge_iou=args.merge_iou,
        detections=merged_detections,
        raw_detection_count=full_count + tile_count,
        full_detection_count=full_count,
        tile_detection_count=tile_count,
        inference_time_sec=inference_time,
    )
    save_tiled_result_json(result, output_json)
    save_tiled_visualization(image_path, merged_detections, output_image)
    return full_count + tile_count, len(merged_detections)


def write_failure_log(output_dir: Path, failure_records: list[str]) -> None:
    """Write failed image records to a text file when any image fails."""
    if not failure_records:
        return
    failure_log = output_dir / "grounding_dino_tiled_batch_failures.txt"
    failure_log.write_text("\n".join(failure_records) + "\n", encoding="utf-8")
    LOGGER.info("Wrote failure log: %s", failure_log.as_posix())


def print_summary(summary: TiledBatchSummary) -> None:
    """Print the required batch summary."""
    print("OVDAS-Tile Grounding DINO batch inference summary:")
    print(f"- total_images: {summary.total_images}")
    print(f"- processed_images: {summary.processed_images}")
    print(f"- skipped_images: {summary.skipped_images}")
    print(f"- failed_images: {summary.failed_images}")
    print(f"- raw_boxes: {summary.raw_boxes}")
    print(f"- merged_boxes: {summary.merged_boxes}")
    print(f"- output_dir: {summary.output_dir}")


def run_batch(args: argparse.Namespace) -> TiledBatchSummary:
    """Run OVDAS-Tile batch inference."""
    validate_args(args)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    image_exts = parse_image_exts(args.image_exts)
    image_list = Path(args.image_list) if args.image_list else None
    image_paths = collect_images(image_dir, image_exts, image_list, args.max_images)
    phrase_to_id, id_to_name = load_class_mapping(Path(args.classes_config))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = TiledBatchSummary(total_images=len(image_paths), output_dir=output_dir.as_posix())
    failure_records: list[str] = []

    LOGGER.info("Found %d image(s). Output directory: %s", len(image_paths), output_dir.as_posix())
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

    LOGGER.info("Loading Grounding DINO model once for tiled batch inference.")
    predictor = GroundingDinoPredictor(
        config_file=Path(args.config_file),
        checkpoint=Path(args.checkpoint),
        device=args.device,
    )

    for index, image_path in enumerate(pending_images, start=1):
        output_image, output_json = output_paths(image_path, output_dir)
        LOGGER.info("[%d/%d] Processing %s", index, len(pending_images), image_path.as_posix())
        try:
            raw_count, merged_count = run_tiled_image(
                image_path=image_path,
                output_json=output_json,
                output_image=output_image,
                predictor=predictor,
                args=args,
                phrase_to_id=phrase_to_id,
                id_to_name=id_to_name,
            )
        except Exception as exc:
            summary.failed_images += 1
            message = f"{image_path.as_posix()}: {exc}"
            failure_records.append(message)
            LOGGER.error("Failed image, continuing: %s", message)
            continue

        summary.processed_images += 1
        summary.raw_boxes += raw_count
        summary.merged_boxes += merged_count
        LOGGER.info(
            "Saved %d merged box(es) from %d raw box(es): %s",
            merged_count,
            raw_count,
            output_json.as_posix(),
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
