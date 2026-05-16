"""Analyze small/medium/large YOLO detection performance on VisDrone val."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.small_object_eval import analyze_dataset, parse_image_exts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze YOLO prediction txt files by small/medium/large object size. "
            "Predictions must use YOLO normalized cx cy w h with optional confidence."
        )
    )
    parser.add_argument("--image-dir", required=True, help="Validation image directory.")
    parser.add_argument("--label-dir", required=True, help="Manual YOLO validation label directory.")
    parser.add_argument("--pred-dir", required=True, help="YOLO prediction txt directory.")
    parser.add_argument(
        "--class-names",
        "--classes-config",
        dest="class_names",
        default="configs/classes_visdrone.yaml",
        help="Class YAML, for example configs/classes_visdrone.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for summary.csv, by_size.csv, by_class.csv, and related outputs.",
    )
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="Directory for failure-case visualizations when --save-visualizations is set.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name written to CSV rows. Defaults to pred-dir parent name when possible.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for same-class greedy matching.",
    )
    parser.add_argument(
        "--size-mode",
        choices=("coco", "relative"),
        default="coco",
        help="Size split rule: COCO absolute pixel area or relative image area.",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save failure-case images, prioritizing small-object false negatives.",
    )
    parser.add_argument(
        "--max-vis",
        type=int,
        default=50,
        help="Maximum number of failure-case images to save.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of validation images for local smoke tests.",
    )
    parser.add_argument(
        "--image-exts",
        default=".jpg,.jpeg,.png,.bmp,.tif,.tiff",
        help="Comma-separated image extensions to evaluate.",
    )
    return parser.parse_args()


def infer_model_name(pred_dir: Path) -> str:
    """Infer a readable model name from a prediction label directory."""
    if pred_dir.name == "labels" and pred_dir.parent.name:
        return pred_dir.parent.name
    return pred_dir.name


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    model_name = args.model_name or infer_model_name(pred_dir)

    try:
        result = analyze_dataset(
            image_dir=Path(args.image_dir),
            label_dir=Path(args.label_dir),
            pred_dir=pred_dir,
            class_names_path=Path(args.class_names),
            output_dir=Path(args.output_dir),
            model_name=model_name,
            iou_threshold=args.iou_threshold,
            size_mode=args.size_mode,
            image_exts=parse_image_exts(args.image_exts),
            limit=args.limit,
            save_visualizations=args.save_visualizations,
            vis_dir=Path(args.vis_dir) if args.vis_dir else None,
            max_vis=args.max_vis,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    stats = result.overall_stats
    print("Small-object analysis summary:")
    print(f"- model_name: {result.model_name}")
    print(f"- gt_count: {stats.gt_count}")
    print(f"- pred_count: {stats.pred_count}")
    print(f"- matched_count: {stats.matched_count}")
    print(f"- false_negative_count: {stats.false_negative_count}")
    print(f"- false_positive_count: {stats.false_positive_count}")
    print(f"- precision: {stats.precision:.6f}")
    print(f"- recall: {stats.recall:.6f}")
    print(f"- evaluated_images: {result.counters.evaluated_images}")
    print(f"- failed_images: {result.counters.failed_images}")
    print(f"- output_dir: {Path(args.output_dir).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
