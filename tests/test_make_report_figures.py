from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tools import make_report_figures as report_figures


class MakeReportFiguresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        report_figures.setup_matplotlib()

    @staticmethod
    def write_csv(
        path: Path,
        fieldnames: list[str],
        rows: list[dict[str, str]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_new_day5_artifacts_are_declared(self) -> None:
        self.assertIn(
            "auto_label_val_three_method_comparison.csv",
            report_figures.REQUIRED_TABLES,
        )
        self.assertIn(
            "auto_label_val_failure_comparison.csv",
            report_figures.REQUIRED_TABLES,
        )
        self.assertIn(
            "auto_label_val_metrics.png",
            report_figures.EXPECTED_FIGURES,
        )
        self.assertIn(
            "auto_label_val_failure_comparison.png",
            report_figures.EXPECTED_FIGURES,
        )

    def test_plot_auto_label_val_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            table_dir = root / "tables"
            output_dir = root / "figures"

            self.write_csv(
                table_dir / "auto_label_val_three_method_comparison.csv",
                [
                    "method",
                    "precision",
                    "recall",
                    "f1",
                    "mean_iou",
                    "small_recall",
                    "medium_recall",
                    "large_recall",
                    "pred_count",
                    "matched_count",
                    "average_time_sec_per_image",
                ],
                [
                    {
                        "method": "DINO-only",
                        "precision": "0.673536",
                        "recall": "0.189877",
                        "f1": "0.296240",
                        "mean_iou": "0.881416",
                        "small_recall": "0.065082",
                        "medium_recall": "0.436062",
                        "large_recall": "0.784015",
                        "pred_count": "10482",
                        "matched_count": "7060",
                        "average_time_sec_per_image": "0.090981",
                    },
                    {
                        "method": "OVDAS-Tile",
                        "precision": "0.432059",
                        "recall": "0.408854",
                        "f1": "0.420136",
                        "mean_iou": "0.825779",
                        "small_recall": "0.276383",
                        "medium_recall": "0.689523",
                        "large_recall": "0.846813",
                        "pred_count": "35185",
                        "matched_count": "15202",
                        "average_time_sec_per_image": "0.803590",
                    },
                ],
            )

            report_figures.plot_auto_label_val_metrics(table_dir, output_dir)

            output = output_dir / "auto_label_val_metrics.png"
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

    def test_plot_auto_label_failure_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            table_dir = root / "tables"
            output_dir = root / "figures"

            self.write_csv(
                table_dir / "auto_label_val_failure_comparison.csv",
                [
                    "method",
                    "total_false_negatives",
                    "small_false_negatives",
                    "medium_false_negatives",
                    "large_false_negatives",
                    "no_overlap",
                    "no_same_class_prediction",
                    "low_iou",
                    "pedestrian_false_negatives",
                    "motor_false_negatives",
                    "car_false_negatives",
                ],
                [
                    {
                        "method": "DINO-only",
                        "total_false_negatives": "30122",
                        "small_false_negatives": "23608",
                        "medium_false_negatives": "6317",
                        "large_false_negatives": "197",
                        "no_overlap": "14399",
                        "no_same_class_prediction": "14813",
                        "low_iou": "905",
                        "pedestrian_false_negatives": "8757",
                        "motor_false_negatives": "4886",
                        "car_false_negatives": "8246",
                    },
                    {
                        "method": "OVDAS-Tile",
                        "total_false_negatives": "21980",
                        "small_false_negatives": "18247",
                        "medium_false_negatives": "3585",
                        "large_false_negatives": "148",
                        "no_overlap": "14329",
                        "no_same_class_prediction": "4737",
                        "low_iou": "2914",
                        "pedestrian_false_negatives": "7547",
                        "motor_false_negatives": "4877",
                        "car_false_negatives": "3849",
                    },
                ],
            )

            report_figures.plot_auto_label_failure_comparison(
                table_dir,
                output_dir,
            )

            output = output_dir / "auto_label_val_failure_comparison.png"
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
