"""Command-line entry for SAM refinement on a single image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.sam_refine import (
    run_sam_refine_single,
    save_refine_visualization,
    save_refined_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Refine one Grounding DINO JSON with SAM bbox prompts."
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--dino-json", required=True, help="Grounding DINO JSON path.")
    parser.add_argument("--output-json", required=True, help="Output refined JSON path.")
    parser.add_argument("--vis-output", required=True, help="Output visualization image path.")
    parser.add_argument("--sam-checkpoint", required=True, help="SAM checkpoint path.")
    parser.add_argument("--model-type", default="vit_h", help="SAM model type, for example vit_h.")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Inference device, for example cpu, cuda, or cuda:0.",
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
    return parser.parse_args()


def print_summary(
    image_path: Path,
    output_json: Path,
    vis_output: Path,
    total_detections: int,
    refined_detections: int,
    device: str,
) -> None:
    """Print a concise single-image summary."""
    print("SAM refine single-image summary:")
    print(f"- image_path: {image_path.as_posix()}")
    print(f"- total_detections: {total_detections}")
    print(f"- refined_detections: {refined_detections}")
    print(f"- output_json: {output_json.as_posix()}")
    print(f"- vis_output: {vis_output.as_posix()}")
    print(f"- device: {device}")


def main() -> int:
    """Run SAM refinement for one image."""
    args = parse_args()
    image_path = Path(args.image)
    dino_json_path = Path(args.dino_json)
    output_json = Path(args.output_json)
    vis_output = Path(args.vis_output)
    checkpoint = Path(args.sam_checkpoint)
    mask_output_dir = Path(args.mask_output_dir) if args.mask_output_dir else None

    try:
        output = run_sam_refine_single(
            image_path=image_path,
            dino_json_path=dino_json_path,
            checkpoint=checkpoint,
            model_type=args.model_type,
            device=args.device,
            save_mask=args.save_mask,
            mask_output_dir=mask_output_dir,
        )
        save_refined_json(output.result, output_json)
        save_refine_visualization(
            image_path=image_path,
            detections=output.result["detections"],
            masks=output.masks,
            output_image=vis_output,
        )
        print_summary(
            image_path=image_path,
            output_json=output_json,
            vis_output=vis_output,
            total_detections=output.total_detections,
            refined_detections=output.refined_detections,
            device=output.result["sam_refine"]["device"],
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
