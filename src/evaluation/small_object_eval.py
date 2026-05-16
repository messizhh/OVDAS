"""Small/medium/large object analysis for YOLO detection predictions."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SIZE_GROUPS = ("small", "medium", "large")
COCO_SMALL_AREA = 32 * 32
COCO_MEDIUM_AREA = 96 * 96
RELATIVE_SMALL_AREA = 0.001
RELATIVE_MEDIUM_AREA = 0.01


@dataclass(frozen=True)
class Box:
    """One pixel-space YOLO box."""

    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float | None = None

    @property
    def area(self) -> float:
        """Return box area in pixels."""
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class MetricStats:
    """Detection matching statistics."""

    gt_count: int = 0
    pred_count: int = 0
    matched_count: int = 0
    iou_sum: float = 0.0

    @property
    def false_positive_count(self) -> int:
        """Return unmatched prediction count."""
        return max(0, self.pred_count - self.matched_count)

    @property
    def false_negative_count(self) -> int:
        """Return unmatched ground-truth count."""
        return max(0, self.gt_count - self.matched_count)

    @property
    def precision(self) -> float:
        """Return precision with zero-safe division."""
        if self.pred_count == 0:
            return 0.0
        return self.matched_count / self.pred_count

    @property
    def recall(self) -> float:
        """Return recall with zero-safe division."""
        if self.gt_count == 0:
            return 0.0
        return self.matched_count / self.gt_count

    @property
    def mean_iou(self) -> float:
        """Return mean IoU over matched pairs."""
        if self.matched_count == 0:
            return 0.0
        return self.iou_sum / self.matched_count


@dataclass
class EvalCounters:
    """Dataset-level bookkeeping counters."""

    total_images: int = 0
    evaluated_images: int = 0
    failed_images: int = 0
    missing_gt_label_files: int = 0
    missing_pred_label_files: int = 0
    invalid_gt_label_lines: int = 0
    invalid_pred_label_lines: int = 0
    visualizations_saved: int = 0
    visualization_failures: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MatchResult:
    """Greedy matching output for one image."""

    matches: list[tuple[int, int, float]]
    matched_gt: set[int]
    matched_pred: set[int]
    best_iou_by_gt: dict[int, float]
    same_class_pred_count_by_gt: dict[int, int]


@dataclass(frozen=True)
class FailureCase:
    """One unmatched ground-truth object for CSV and visualization."""

    image_name: str
    image_path: Path
    class_id: int
    class_name: str
    size_group: str
    gt_xyxy: tuple[float, float, float, float]
    best_iou: float
    matched: bool
    reason: str


@dataclass
class ImageFailureRecord:
    """Failure context needed to draw one image."""

    image_path: Path
    gt_boxes: list[Box]
    pred_boxes: list[Box]
    failure_cases: list[FailureCase]


@dataclass
class AnalysisResult:
    """All analysis outputs before CSV serialization."""

    model_name: str
    overall_stats: MetricStats
    class_stats: dict[int, MetricStats]
    size_stats: dict[str, MetricStats]
    class_size_stats: dict[tuple[int, str], MetricStats]
    failure_cases: list[FailureCase]
    counters: EvalCounters


def parse_image_exts(value: str) -> set[str]:
    """Parse a comma-separated image extension list."""
    extensions: set[str] = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        extensions.add(item if item.startswith(".") else f".{item}")
    return extensions


def load_class_names(class_names_path: Path) -> dict[int, str]:
    """Load class id to class name mapping from project YAML."""
    with class_names_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {class_names_path}")

    classes = data.get("classes", [])
    if not isinstance(classes, list):
        raise ValueError(f"'classes' must be a list in {class_names_path}")

    class_names: dict[int, str] = {}
    for index, item in enumerate(classes):
        if isinstance(item, str):
            class_names[index] = item
            continue
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            class_id = item.get("id", index)
            if not isinstance(class_id, int):
                raise ValueError(f"Invalid class id in {class_names_path}: {class_id}")
            class_names[class_id] = item["name"]
            continue
        raise ValueError(f"Invalid class entry in {class_names_path}: {item}")

    if not class_names:
        raise ValueError(f"No classes found in {class_names_path}")
    return class_names


def list_images(image_dir: Path, image_exts: set[str], limit: int | None) -> list[Path]:
    """List images in deterministic order."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_exts
        ],
        key=lambda path: path.name.lower(),
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive when provided.")
        return image_paths[:limit]
    return image_paths


def read_image_size(image_path: Path) -> tuple[int, int]:
    """Read image width and height."""
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def yolo_cxcywh_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Convert normalized YOLO cxcywh to clipped pixel xyxy."""
    x1 = (x_center - width / 2.0) * image_width
    y1 = (y_center - height / 2.0) * image_height
    x2 = (x_center + width / 2.0) * image_width
    y2 = (y_center + height / 2.0) * image_height

    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def read_yolo_labels(
    label_path: Path,
    image_width: int,
    image_height: int,
    valid_class_ids: set[int],
) -> tuple[list[Box], int, bool]:
    """Read YOLO txt labels with optional confidence in the sixth column."""
    if not label_path.is_file():
        return [], 0, False

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return [], 0, True

    boxes: list[Box] = []
    invalid_lines = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) not in {5, 6}:
            invalid_lines += 1
            print(f"[WARN] Invalid YOLO line ignored: {label_path}:{line_number}")
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            confidence = float(parts[5]) if len(parts) == 6 else None
        except ValueError:
            invalid_lines += 1
            print(f"[WARN] Invalid YOLO values ignored: {label_path}:{line_number}")
            continue

        if class_id not in valid_class_ids or width <= 0.0 or height <= 0.0:
            invalid_lines += 1
            print(f"[WARN] Out-of-range YOLO label ignored: {label_path}:{line_number}")
            continue

        xyxy = yolo_cxcywh_to_xyxy(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
        )
        if xyxy is None:
            invalid_lines += 1
            print(f"[WARN] Degenerate YOLO label ignored: {label_path}:{line_number}")
            continue

        boxes.append(Box(class_id=class_id, xyxy=xyxy, confidence=confidence))

    return boxes, invalid_lines, True


def compute_iou(a: Box, b: Box) -> float:
    """Compute IoU between two pixel-space boxes."""
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = a.area + b.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def classify_size(
    box: Box,
    image_width: int,
    image_height: int,
    size_mode: str,
) -> str:
    """Classify a box as small, medium, or large."""
    if size_mode == "coco":
        area = box.area
        if area < COCO_SMALL_AREA:
            return "small"
        if area < COCO_MEDIUM_AREA:
            return "medium"
        return "large"

    if size_mode == "relative":
        image_area = float(image_width * image_height)
        if image_area <= 0.0:
            raise ValueError("Image area must be positive for relative size mode.")
        relative_area = box.area / image_area
        if relative_area < RELATIVE_SMALL_AREA:
            return "small"
        if relative_area < RELATIVE_MEDIUM_AREA:
            return "medium"
        return "large"

    raise ValueError(f"Unsupported size mode: {size_mode}")


def match_predictions_to_gt(
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    iou_threshold: float,
) -> MatchResult:
    """Greedily match same-class predictions to GT boxes by descending IoU."""
    candidates: list[tuple[float, int, int]] = []
    best_iou_by_gt = {index: 0.0 for index in range(len(gt_boxes))}
    same_class_pred_count_by_gt = {index: 0 for index in range(len(gt_boxes))}

    for gt_index, gt_box in enumerate(gt_boxes):
        for pred_index, pred_box in enumerate(pred_boxes):
            if gt_box.class_id != pred_box.class_id:
                continue
            same_class_pred_count_by_gt[gt_index] += 1
            iou = compute_iou(gt_box, pred_box)
            best_iou_by_gt[gt_index] = max(best_iou_by_gt[gt_index], iou)
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))

    candidates.sort(reverse=True, key=lambda item: item[0])
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou))

    return MatchResult(
        matches=matches,
        matched_gt=matched_gt,
        matched_pred=matched_pred,
        best_iou_by_gt=best_iou_by_gt,
        same_class_pred_count_by_gt=same_class_pred_count_by_gt,
    )


def _new_class_stats(class_names: dict[int, str]) -> dict[int, MetricStats]:
    return {class_id: MetricStats() for class_id in class_names}


def _new_size_stats() -> dict[str, MetricStats]:
    return {size_group: MetricStats() for size_group in SIZE_GROUPS}


def _new_class_size_stats(class_names: dict[int, str]) -> dict[tuple[int, str], MetricStats]:
    return {
        (class_id, size_group): MetricStats()
        for class_id in class_names
        for size_group in SIZE_GROUPS
    }


def _metric_row(stats: MetricStats) -> dict[str, Any]:
    return {
        "gt_count": stats.gt_count,
        "pred_count": stats.pred_count,
        "matched_count": stats.matched_count,
        "false_negative_count": stats.false_negative_count,
        "false_positive_count": stats.false_positive_count,
        "precision": f"{stats.precision:.6f}",
        "recall": f"{stats.recall:.6f}",
        "mean_iou": f"{stats.mean_iou:.6f}",
    }


def _failure_reason(
    pred_boxes: list[Box],
    same_class_pred_count: int,
    best_iou: float,
) -> str:
    if not pred_boxes:
        return "no_predictions"
    if same_class_pred_count == 0:
        return "no_same_class_prediction"
    if best_iou <= 0.0:
        return "no_overlap"
    return "low_iou"


def aggregate_statistics(
    image_name: str,
    image_path: Path,
    gt_boxes: list[Box],
    pred_boxes: list[Box],
    match_result: MatchResult,
    image_width: int,
    image_height: int,
    size_mode: str,
    class_names: dict[int, str],
    overall_stats: MetricStats,
    class_stats: dict[int, MetricStats],
    size_stats: dict[str, MetricStats],
    class_size_stats: dict[tuple[int, str], MetricStats],
    failure_cases: list[FailureCase],
) -> None:
    """Update all aggregate stats for one image."""
    overall_stats.gt_count += len(gt_boxes)
    overall_stats.pred_count += len(pred_boxes)
    overall_stats.matched_count += len(match_result.matches)
    overall_stats.iou_sum += sum(iou for _, _, iou in match_result.matches)

    for class_id in class_names:
        gt_count = sum(1 for box in gt_boxes if box.class_id == class_id)
        pred_count = sum(1 for box in pred_boxes if box.class_id == class_id)
        matched_count = sum(
            1 for gt_index, _, _ in match_result.matches if gt_boxes[gt_index].class_id == class_id
        )
        iou_sum = sum(
            iou for gt_index, _, iou in match_result.matches if gt_boxes[gt_index].class_id == class_id
        )
        stats = class_stats[class_id]
        stats.gt_count += gt_count
        stats.pred_count += pred_count
        stats.matched_count += matched_count
        stats.iou_sum += iou_sum

    for gt_index, pred_index, iou in match_result.matches:
        gt_box = gt_boxes[gt_index]
        size_group = classify_size(gt_box, image_width, image_height, size_mode)

        size_stats[size_group].gt_count += 1
        size_stats[size_group].pred_count += 1
        size_stats[size_group].matched_count += 1
        size_stats[size_group].iou_sum += iou

        class_size = class_size_stats[(gt_box.class_id, size_group)]
        class_size.gt_count += 1
        class_size.pred_count += 1
        class_size.matched_count += 1
        class_size.iou_sum += iou

    for gt_index, gt_box in enumerate(gt_boxes):
        if gt_index in match_result.matched_gt:
            continue

        size_group = classify_size(gt_box, image_width, image_height, size_mode)
        size_stats[size_group].gt_count += 1
        class_size_stats[(gt_box.class_id, size_group)].gt_count += 1

        best_iou = match_result.best_iou_by_gt.get(gt_index, 0.0)
        same_class_pred_count = match_result.same_class_pred_count_by_gt.get(gt_index, 0)
        failure_cases.append(
            FailureCase(
                image_name=image_name,
                image_path=image_path,
                class_id=gt_box.class_id,
                class_name=class_names.get(gt_box.class_id, f"class_{gt_box.class_id}"),
                size_group=size_group,
                gt_xyxy=gt_box.xyxy,
                best_iou=best_iou,
                matched=False,
                reason=_failure_reason(pred_boxes, same_class_pred_count, best_iou),
            )
        )

    for pred_index, pred_box in enumerate(pred_boxes):
        if pred_index in match_result.matched_pred:
            continue

        size_group = classify_size(pred_box, image_width, image_height, size_mode)
        size_stats[size_group].pred_count += 1
        class_size_stats[(pred_box.class_id, size_group)].pred_count += 1


def analyze_dataset(
    image_dir: Path,
    label_dir: Path,
    pred_dir: Path,
    class_names_path: Path,
    output_dir: Path,
    model_name: str,
    iou_threshold: float,
    size_mode: str,
    image_exts: set[str] | None = None,
    limit: int | None = None,
    save_visualizations: bool = False,
    vis_dir: Path | None = None,
    max_vis: int = 50,
) -> AnalysisResult:
    """Analyze YOLO predictions against manual val labels and write outputs."""
    if not label_dir.is_dir():
        raise FileNotFoundError(f"GT label directory does not exist: {label_dir}")
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction label directory does not exist: {pred_dir}")
    if iou_threshold < 0.0 or iou_threshold > 1.0:
        raise ValueError("--iou-threshold must be between 0 and 1.")
    if size_mode not in {"coco", "relative"}:
        raise ValueError("--size-mode must be 'coco' or 'relative'.")
    if max_vis < 0:
        raise ValueError("--max-vis must be greater than or equal to 0.")

    class_names = load_class_names(class_names_path)
    valid_class_ids = set(class_names)
    image_paths = list_images(image_dir, image_exts or set(DEFAULT_IMAGE_EXTS), limit)

    overall_stats = MetricStats()
    class_stats = _new_class_stats(class_names)
    size_stats = _new_size_stats()
    class_size_stats = _new_class_size_stats(class_names)
    failure_cases: list[FailureCase] = []
    image_failure_records: dict[str, ImageFailureRecord] = {}
    counters = EvalCounters(total_images=len(image_paths))

    for image_path in image_paths:
        try:
            image_width, image_height = read_image_size(image_path)
            if image_width <= 0 or image_height <= 0:
                raise ValueError(f"Invalid image size: {image_width}x{image_height}")

            label_path = label_dir / f"{image_path.stem}.txt"
            pred_path = pred_dir / f"{image_path.stem}.txt"

            gt_boxes, invalid_gt, gt_exists = read_yolo_labels(
                label_path=label_path,
                image_width=image_width,
                image_height=image_height,
                valid_class_ids=valid_class_ids,
            )
            pred_boxes, invalid_pred, pred_exists = read_yolo_labels(
                label_path=pred_path,
                image_width=image_width,
                image_height=image_height,
                valid_class_ids=valid_class_ids,
            )

            if not gt_exists:
                counters.missing_gt_label_files += 1
            if not pred_exists:
                counters.missing_pred_label_files += 1
            counters.invalid_gt_label_lines += invalid_gt
            counters.invalid_pred_label_lines += invalid_pred

            match_result = match_predictions_to_gt(gt_boxes, pred_boxes, iou_threshold)
            before_failure_count = len(failure_cases)
            aggregate_statistics(
                image_name=image_path.name,
                image_path=image_path,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                match_result=match_result,
                image_width=image_width,
                image_height=image_height,
                size_mode=size_mode,
                class_names=class_names,
                overall_stats=overall_stats,
                class_stats=class_stats,
                size_stats=size_stats,
                class_size_stats=class_size_stats,
                failure_cases=failure_cases,
            )
            new_failures = failure_cases[before_failure_count:]
            if new_failures:
                image_failure_records[image_path.name] = ImageFailureRecord(
                    image_path=image_path,
                    gt_boxes=gt_boxes,
                    pred_boxes=pred_boxes,
                    failure_cases=new_failures,
                )

            counters.evaluated_images += 1
        except Exception as exc:
            counters.failed_images += 1
            message = f"{image_path.as_posix()}: {exc}"
            counters.warnings.append(message)
            print(f"[WARN] Failed image skipped: {message}")

    result = AnalysisResult(
        model_name=model_name,
        overall_stats=overall_stats,
        class_stats=class_stats,
        size_stats=size_stats,
        class_size_stats=class_size_stats,
        failure_cases=failure_cases,
        counters=counters,
    )
    write_analysis_outputs(
        result=result,
        class_names=class_names,
        output_dir=output_dir,
        iou_threshold=iou_threshold,
        size_mode=size_mode,
    )

    if save_visualizations and max_vis > 0:
        target_vis_dir = vis_dir or output_dir / "visualizations"
        saved, failed = save_failure_visualizations(
            failure_records=image_failure_records,
            failure_cases=failure_cases,
            class_names=class_names,
            output_dir=target_vis_dir,
            max_vis=max_vis,
        )
        counters.visualizations_saved = saved
        counters.visualization_failures = failed
        write_summary_csv(
            output_dir / "summary.csv",
            result,
            iou_threshold=iou_threshold,
            size_mode=size_mode,
        )

    if counters.warnings:
        failure_log = output_dir / "small_object_analysis_failures.txt"
        failure_log.write_text("\n".join(counters.warnings) + "\n", encoding="utf-8")
        print(f"[INFO] Wrote warning log: {failure_log.as_posix()}")

    return result


def write_analysis_outputs(
    result: AnalysisResult,
    class_names: dict[int, str],
    output_dir: Path,
    iou_threshold: float,
    size_mode: str,
) -> None:
    """Write all CSV outputs for one model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(output_dir / "summary.csv", result, iou_threshold, size_mode)
    write_by_size_csv(output_dir / "by_size.csv", result)
    write_by_class_csv(output_dir / "by_class.csv", result, class_names)
    write_by_class_size_csv(output_dir / "by_class_size.csv", result, class_names)
    write_failure_cases_csv(output_dir / "failure_cases.csv", result.failure_cases)


def write_summary_csv(
    output_path: Path,
    result: AnalysisResult,
    iou_threshold: float,
    size_mode: str,
) -> None:
    """Write one-row summary CSV."""
    row = {
        "model_name": result.model_name,
        "size_mode": size_mode,
        "iou_threshold": f"{iou_threshold:.6f}",
        **_metric_row(result.overall_stats),
        "total_images": result.counters.total_images,
        "evaluated_images": result.counters.evaluated_images,
        "failed_images": result.counters.failed_images,
        "missing_gt_label_files": result.counters.missing_gt_label_files,
        "missing_pred_label_files": result.counters.missing_pred_label_files,
        "invalid_gt_label_lines": result.counters.invalid_gt_label_lines,
        "invalid_pred_label_lines": result.counters.invalid_pred_label_lines,
        "visualizations_saved": result.counters.visualizations_saved,
        "visualization_failures": result.counters.visualization_failures,
    }
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_by_size_csv(output_path: Path, result: AnalysisResult) -> None:
    """Write small/medium/large metrics CSV."""
    fieldnames = [
        "model_name",
        "size_group",
        "gt_count",
        "pred_count",
        "matched_count",
        "false_negative_count",
        "false_positive_count",
        "precision",
        "recall",
        "mean_iou",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for size_group in SIZE_GROUPS:
            writer.writerow(
                {
                    "model_name": result.model_name,
                    "size_group": size_group,
                    **_metric_row(result.size_stats[size_group]),
                }
            )


def write_by_class_csv(
    output_path: Path,
    result: AnalysisResult,
    class_names: dict[int, str],
) -> None:
    """Write per-class metrics CSV."""
    fieldnames = [
        "model_name",
        "class_id",
        "class_name",
        "gt_count",
        "pred_count",
        "matched_count",
        "false_negative_count",
        "false_positive_count",
        "precision",
        "recall",
        "mean_iou",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for class_id in sorted(class_names):
            writer.writerow(
                {
                    "model_name": result.model_name,
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    **_metric_row(result.class_stats[class_id]),
                }
            )


def write_by_class_size_csv(
    output_path: Path,
    result: AnalysisResult,
    class_names: dict[int, str],
) -> None:
    """Write class and size cross statistics CSV."""
    fieldnames = [
        "model_name",
        "class_id",
        "class_name",
        "size_group",
        "gt_count",
        "pred_count",
        "matched_count",
        "false_negative_count",
        "false_positive_count",
        "precision",
        "recall",
        "mean_iou",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for class_id in sorted(class_names):
            for size_group in SIZE_GROUPS:
                writer.writerow(
                    {
                        "model_name": result.model_name,
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "size_group": size_group,
                        **_metric_row(result.class_size_stats[(class_id, size_group)]),
                    }
                )


def _format_xyxy(xyxy: tuple[float, float, float, float]) -> str:
    return "[" + ",".join(f"{value:.2f}" for value in xyxy) + "]"


def write_failure_cases_csv(output_path: Path, failure_cases: list[FailureCase]) -> None:
    """Write unmatched GT cases for failure analysis."""
    fieldnames = [
        "image_name",
        "class_id",
        "class_name",
        "size_group",
        "gt_xyxy",
        "best_iou",
        "matched",
        "reason",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for case in failure_cases:
            writer.writerow(
                {
                    "image_name": case.image_name,
                    "class_id": case.class_id,
                    "class_name": case.class_name,
                    "size_group": case.size_group,
                    "gt_xyxy": _format_xyxy(case.gt_xyxy),
                    "best_iou": f"{case.best_iou:.6f}",
                    "matched": str(case.matched).lower(),
                    "reason": case.reason,
                }
            )


def save_failure_visualizations(
    failure_records: dict[str, ImageFailureRecord],
    failure_cases: list[FailureCase],
    class_names: dict[int, str],
    output_dir: Path,
    max_vis: int,
) -> tuple[int, int]:
    """Save visualizations, prioritizing small-object false negatives."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_names: list[str] = []
    seen_names: set[str] = set()

    sorted_cases = sorted(
        failure_cases,
        key=lambda case: (
            0 if case.size_group == "small" else 1,
            case.best_iou,
            case.image_name,
        ),
    )
    for case in sorted_cases:
        if case.image_name in seen_names:
            continue
        if case.image_name not in failure_records:
            continue
        selected_names.append(case.image_name)
        seen_names.add(case.image_name)
        if len(selected_names) >= max_vis:
            break

    saved = 0
    failed = 0
    for index, image_name in enumerate(selected_names, start=1):
        record = failure_records[image_name]
        output_path = output_dir / f"{Path(image_name).stem}_failure_{index:03d}.jpg"
        try:
            visualize_failure_case(record, class_names, output_path)
            saved += 1
        except Exception as exc:
            failed += 1
            print(f"[WARN] Failed to save visualization for {image_name}: {exc}")

    return saved, failed


def visualize_failure_case(
    record: ImageFailureRecord,
    class_names: dict[int, str],
    output_path: Path,
) -> None:
    """Draw false-negative GT boxes and predictions on one image."""
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(record.image_path) as image:
        canvas = image.convert("RGB")

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for pred_box in record.pred_boxes:
        class_name = class_names.get(pred_box.class_id, f"class_{pred_box.class_id}")
        if pred_box.confidence is None:
            label = f"Pred {class_name}"
        else:
            label = f"Pred {class_name} {pred_box.confidence:.2f}"
        _draw_labeled_box(draw, pred_box.xyxy, label, outline=(40, 120, 255), font=font)

    for case in record.failure_cases:
        label = f"FN GT {case.class_name} {case.size_group}"
        _draw_labeled_box(draw, case.gt_xyxy, label, outline=(255, 64, 64), font=font, width=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def _draw_labeled_box(
    draw: Any,
    xyxy: tuple[float, float, float, float],
    label: str,
    outline: tuple[int, int, int],
    font: Any,
    width: int = 2,
) -> None:
    """Draw one labeled rectangle with a solid text background."""
    x1, y1, x2, y2 = (int(round(value)) for value in xyxy)
    draw.rectangle((x1, y1, x2, y2), outline=outline, width=width)

    text_x = x1
    text_y = max(0, y1 - 12)
    text_bbox = draw.textbbox((text_x, text_y), label, font=font)
    draw.rectangle(text_bbox, fill=outline)
    draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)
