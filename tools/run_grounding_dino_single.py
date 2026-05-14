"""Command-line entry for Grounding DINO single-image inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.open_vocab.grounding_dino_infer import (
    run_grounding_dino_single,
    save_result_json,
    save_visualization,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Grounding DINO inference on a single image."
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--prompt", required=True, help="Text prompt for open-vocabulary detection.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for visualization image and JSON result.",
    )
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
    return parser.parse_args()


def validate_threshold(name: str, value: float) -> None:
    """Validate a threshold value."""
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def output_paths(image_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Build output image and JSON paths for one input image."""
    output_image = output_dir / f"{image_path.stem}_grounding_dino.jpg"
    output_json = output_dir / f"{image_path.stem}_grounding_dino.json"
    return output_image, output_json


def print_summary(
    image_path: Path,
    prompt: str,
    num_boxes: int,
    output_image: Path,
    output_json: Path,
    device: str,
) -> None:
    """Print a clear single-image inference summary."""
    print("Grounding DINO single-image inference summary:")
    print(f"- image_path: {image_path.as_posix()}")
    print(f"- prompt: {prompt}")
    print(f"- num_boxes: {num_boxes}")
    print(f"- output_image: {output_image.as_posix()}")
    print(f"- output_json: {output_json.as_posix()}")
    print(f"- device: {device}")


def main() -> int:
    """Run Grounding DINO single-image inference."""
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    config_file = Path(args.config_file)
    checkpoint = Path(args.checkpoint)

    try:
        validate_threshold("--box-threshold", args.box_threshold)
        validate_threshold("--text-threshold", args.text_threshold)

        result = run_grounding_dino_single(
            image_path=image_path,
            prompt=args.prompt,
            config_file=config_file,
            checkpoint=checkpoint,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
        )

        output_image, output_json = output_paths(image_path, output_dir)
        save_result_json(result, output_json)
        save_visualization(image_path, result.detections, output_image)

        print_summary(
            image_path=image_path,
            prompt=args.prompt,
            num_boxes=len(result.detections),
            output_image=output_image,
            output_json=output_json,
            device=result.device,
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
