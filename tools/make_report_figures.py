"""Generate report-ready charts from CSV result tables."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Any

plt: Any | None = None


REQUIRED_TABLES = (
    "yolo_manual_baseline.csv",
    "yolo_auto_label_results.csv",
    "yolo_three_model_comparison.csv",
    "small_object_analysis.csv",
    "ablation_method.csv",
    "ablation_threshold.csv",
    "ablation_prompt.csv",
    "auto_label_quality.csv",
)

EXPECTED_FIGURES = (
    "yolo_main_metrics.png",
    "three_model_map_comparison.png",
    "small_object_recall.png",
    "ablation_method.png",
    "ablation_threshold.png",
    "ablation_prompt.png",
)

METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "map50": "mAP@0.5",
    "map50_95": "mAP@0.5:0.95",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate report charts from results/tables CSV files."
    )
    parser.add_argument(
        "--table-dir",
        default="results/tables",
        help="Directory containing normalized CSV result tables.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/charts",
        help="Directory where chart PNG files are written.",
    )
    parser.add_argument(
        "--report-table-dir",
        default="report/tables",
        help="Directory where report table CSV copies are written.",
    )
    parser.add_argument(
        "--report-figure-dir",
        default="report/figures",
        help="Directory where report figure PNG copies are written.",
    )
    return parser.parse_args()


def setup_matplotlib() -> None:
    """Load matplotlib with a non-interactive backend."""
    global plt
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ovdas_matplotlib_cache")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to generate report figures. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc
    plt = pyplot


def read_table(table_dir: Path, filename: str) -> list[dict[str, str]]:
    """Read one CSV table, returning an empty list if it is missing."""
    return read_csv_path(table_dir / filename)


def read_csv_path(path: Path) -> list[dict[str, str]]:
    """Read one CSV path, returning an empty list if it is missing."""
    if not path.is_file():
        print(f"[WARN] Missing table: {path.as_posix()}")
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def table_columns(rows: list[dict[str, str]]) -> set[str]:
    """Return the union of CSV columns from rows."""
    columns: set[str] = set()
    for row in rows:
        columns.update(row)
    return columns


def to_float(value: str | None) -> float | None:
    """Convert a CSV cell to float, returning None for blanks or invalid values."""
    if value is None:
        return None


def rows_have_numeric(rows: list[dict[str, str]], column: str) -> bool:
    """Return whether any row has a numeric value in one column."""
    return any(to_float(row.get(column)) is not None for row in rows)


def write_small_object_summary(table_dir: Path, rows: list[dict[str, str]]) -> None:
    """Write a consolidated small-object table from per-model by_size.csv files."""
    output_path = table_dir / "small_object_analysis.csv"
    fieldnames = [
        "model",
        "size_group",
        "gt_count",
        "pred_count",
        "matched_count",
        "false_negative_count",
        "false_positive_count",
        "precision",
        "recall",
        "mean_iou",
        "notes",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    print(f"[INFO] Wrote consolidated small-object table: {output_path.as_posix()}")


def load_small_object_rows(table_dir: Path) -> list[dict[str, str]]:
    """Load small-object rows, falling back to Day 11 per-model outputs."""
    rows = read_table(table_dir, "small_object_analysis.csv")
    if rows and rows_have_numeric(rows, "recall"):
        return rows

    sources = (
        ("manual", table_dir / "small_object_analysis_manual" / "by_size.csv"),
        ("dino_only", table_dir / "small_object_analysis_auto_dino_only" / "by_size.csv"),
        ("dino_sam", table_dir / "small_object_analysis_auto_dino_sam" / "by_size.csv"),
    )
    combined_rows: list[dict[str, str]] = []
    for model_name, path in sources:
        for row in read_csv_path(path):
            combined_rows.append(
                {
                    "model": model_name,
                    "size_group": row.get("size_group", row.get("size", "")),
                    "gt_count": row.get("gt_count", ""),
                    "pred_count": row.get("pred_count", ""),
                    "matched_count": row.get("matched_count", ""),
                    "false_negative_count": row.get("false_negative_count", ""),
                    "false_positive_count": row.get("false_positive_count", ""),
                    "precision": row.get("precision", ""),
                    "recall": row.get("recall", ""),
                    "mean_iou": row.get("mean_iou", ""),
                    "notes": "Consolidated from per-model small-object by_size.csv.",
                }
            )

    if combined_rows and rows_have_numeric(combined_rows, "recall"):
        write_small_object_summary(table_dir, combined_rows)
        return combined_rows

    return rows
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def save_placeholder(output_path: Path, title: str, message: str) -> None:
    """Save a placeholder figure when a source table has no numeric data."""
    if plt is None:
        raise RuntimeError("matplotlib is not initialized.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, wrap=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote placeholder chart: {output_path.as_posix()}")


def format_axis(ax: Any, ylabel: str = "Score") -> None:
    """Apply common chart axis styling."""
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_bar_labels(ax: Any) -> None:
    """Add compact numeric labels above bars."""
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=7)


def plot_grouped_metrics(
    rows: list[dict[str, str]],
    x_column: str,
    metric_columns: list[str],
    output_path: Path,
    title: str,
) -> None:
    """Plot grouped metric bars for a model or method table."""
    columns = table_columns(rows)
    missing_columns = [column for column in [x_column, *metric_columns] if column not in columns]
    if missing_columns:
        save_placeholder(
            output_path,
            title,
            f"Missing required columns: {', '.join(missing_columns)}.",
        )
        return

    data = [
        row
        for row in rows
        if any(to_float(row.get(metric)) is not None for metric in metric_columns)
    ]
    if not data:
        save_placeholder(output_path, title, "No numeric metric rows are available yet.")
        return

    if plt is None:
        raise RuntimeError("matplotlib is not initialized.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row.get(x_column, "")) for row in data]
    x = list(range(len(labels)))
    width = 0.8 / max(1, len(metric_columns))
    max_value = 0.0

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for index, metric in enumerate(metric_columns):
        values = [to_float(row.get(metric)) or 0.0 for row in data]
        if values:
            max_value = max(max_value, max(values))
        offset = (index - (len(metric_columns) - 1) / 2.0) * width
        ax.bar([value + offset for value in x], values, width, label=METRIC_LABELS.get(metric, metric))

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, max(1.0, max_value * 1.18))
    format_axis(ax)
    ax.legend(frameon=False, ncols=min(4, len(metric_columns)))
    add_bar_labels(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote chart: {output_path.as_posix()}")


def plot_yolo_main_metrics(table_dir: Path, output_dir: Path) -> None:
    """Generate grouped Precision/Recall/mAP chart for three YOLO models."""
    rows = read_table(table_dir, "yolo_three_model_comparison.csv")
    plot_grouped_metrics(
        rows=rows,
        x_column="model",
        metric_columns=["precision", "recall", "map50", "map50_95"],
        output_path=output_dir / "yolo_main_metrics.png",
        title="YOLOv8s Main Metrics on VisDrone Val",
    )


def plot_three_model_map(table_dir: Path, output_dir: Path) -> None:
    """Generate mAP comparison chart for the three YOLO models."""
    rows = read_table(table_dir, "yolo_three_model_comparison.csv")
    plot_grouped_metrics(
        rows=rows,
        x_column="model",
        metric_columns=["map50", "map50_95"],
        output_path=output_dir / "three_model_map_comparison.png",
        title="Three-Model mAP Comparison",
    )


def plot_small_object_recall(table_dir: Path, output_dir: Path) -> None:
    """Generate small/medium/large recall chart."""
    output_path = output_dir / "small_object_recall.png"
    title = "Recall by Object Size"
    rows = load_small_object_rows(table_dir)
    required = {"model", "size_group", "recall"}
    if not rows or not required.issubset(table_columns(rows)):
        save_placeholder(output_path, title, "small_object_analysis.csv has no usable recall data.")
        return

    data = [
        (row.get("model", ""), row.get("size_group", ""), to_float(row.get("recall")))
        for row in rows
    ]
    data = [(model, size_group, recall) for model, size_group, recall in data if recall is not None]
    if not data:
        save_placeholder(output_path, title, "Run Day 11 small-object analysis to fill recall values.")
        return

    model_names = list(dict.fromkeys(model for model, _, _ in data))
    size_order = [size for size in ["small", "medium", "large"] if any(item[1] == size for item in data)]
    pivot: dict[tuple[str, str], float] = {}
    for model, size_group, recall in data:
        pivot[(size_group, model)] = recall

    if plt is None:
        raise RuntimeError("matplotlib is not initialized.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = list(range(len(size_order)))
    width = 0.8 / max(1, len(model_names))
    max_value = 0.0
    for index, model_name in enumerate(model_names):
        values = [pivot.get((size_group, model_name), 0.0) for size_group in size_order]
        if values:
            max_value = max(max_value, max(values))
        offset = (index - (len(model_names) - 1) / 2.0) * width
        ax.bar([value + offset for value in x], values, width, label=model_name)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(size_order)
    ax.set_ylim(0.0, max(1.0, max_value * 1.18))
    format_axis(ax, ylabel="Recall")
    ax.legend(frameon=False)
    add_bar_labels(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote chart: {output_path.as_posix()}")


def plot_ablation_method(table_dir: Path, output_dir: Path) -> None:
    """Generate method ablation chart."""
    rows = read_table(table_dir, "ablation_method.csv")
    plot_grouped_metrics(
        rows=rows,
        x_column="method",
        metric_columns=["map50", "map50_95"],
        output_path=output_dir / "ablation_method.png",
        title="Ablation: Auto-Labeling Method",
    )


def plot_ablation_threshold(table_dir: Path, output_dir: Path) -> None:
    """Generate threshold ablation chart."""
    output_path = output_dir / "ablation_threshold.png"
    title = "Ablation: Grounding DINO Threshold"
    rows = read_table(table_dir, "ablation_threshold.csv")
    columns = table_columns(rows)
    if not rows or "map50" not in columns:
        save_placeholder(output_path, title, "ablation_threshold.csv has no usable mAP data.")
        return

    x_column = "box_threshold" if "box_threshold" in columns else "setting"
    if x_column not in columns:
        save_placeholder(output_path, title, "Missing threshold or setting column.")
        return

    metric_columns = [column for column in ["map50", "map50_95", "recall"] if column in columns]
    plot_grouped_metrics(rows, x_column, metric_columns, output_path, title)


def plot_ablation_prompt(table_dir: Path, output_dir: Path) -> None:
    """Generate prompt ablation chart."""
    output_path = output_dir / "ablation_prompt.png"
    title = "Ablation: Text Prompt"
    rows = read_table(table_dir, "ablation_prompt.csv")
    columns = table_columns(rows)
    if not rows or "map50" not in columns:
        save_placeholder(output_path, title, "ablation_prompt.csv has no usable mAP data.")
        return

    x_column = "prompt_name" if "prompt_name" in columns else "prompt"
    if x_column not in columns:
        save_placeholder(output_path, title, "Missing prompt_name or prompt column.")
        return

    metric_columns = [column for column in ["map50", "map50_95", "recall"] if column in columns]
    plot_grouped_metrics(rows, x_column, metric_columns, output_path, title)


def copy_report_artifacts(
    table_dir: Path,
    output_dir: Path,
    report_table_dir: Path,
    report_figure_dir: Path,
) -> None:
    """Copy normalized tables and generated figures into the report directory."""
    report_table_dir.mkdir(parents=True, exist_ok=True)
    report_figure_dir.mkdir(parents=True, exist_ok=True)

    for filename in REQUIRED_TABLES:
        source = table_dir / filename
        if source.is_file():
            shutil.copy2(source, report_table_dir / filename)
            print(f"[INFO] Copied table: {(report_table_dir / filename).as_posix()}")
        else:
            print(f"[WARN] Table not copied because it is missing: {source.as_posix()}")

    for filename in EXPECTED_FIGURES:
        source = output_dir / filename
        if source.is_file():
            shutil.copy2(source, report_figure_dir / filename)
            print(f"[INFO] Copied figure: {(report_figure_dir / filename).as_posix()}")
        else:
            print(f"[WARN] Figure not copied because it is missing: {source.as_posix()}")


def run(args: argparse.Namespace) -> None:
    """Generate all report charts and report copies."""
    setup_matplotlib()
    table_dir = Path(args.table_dir)
    output_dir = Path(args.output_dir)
    report_table_dir = Path(args.report_table_dir)
    report_figure_dir = Path(args.report_figure_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_yolo_main_metrics(table_dir, output_dir)
    plot_three_model_map(table_dir, output_dir)
    plot_small_object_recall(table_dir, output_dir)
    plot_ablation_method(table_dir, output_dir)
    plot_ablation_threshold(table_dir, output_dir)
    plot_ablation_prompt(table_dir, output_dir)
    copy_report_artifacts(table_dir, output_dir, report_table_dir, report_figure_dir)


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    try:
        run(args)
        return 0
    except (OSError, ValueError, RuntimeError, csv.Error) as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
