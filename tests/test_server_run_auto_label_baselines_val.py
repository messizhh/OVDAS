from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "server_run_auto_label_baselines_val.sh"


class ServerRunAutoLabelBaselinesValScriptTest(unittest.TestCase):
    def run_script(
        self,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        merged_env = os.environ.copy()
        merged_env.update(
            {
                "PROJECT_ROOT": PROJECT_ROOT.as_posix(),
                "PYTHON_BIN": "python3",
                "LOG_DIR": "/tmp/auto_label_baselines_val_script_test_logs",
                "TABLE_DIR": "/tmp/auto_label_baselines_val_script_test_tables",
            }
        )
        if env:
            merged_env.update(env)
        return subprocess.run(
            ["bash", SCRIPT.as_posix(), *args],
            cwd=PROJECT_ROOT,
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_help_is_safe(self) -> None:
        result = self.run_script(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--preflight-only", result.stdout)
        self.assertIn("--skip-existing", result.stdout)

    def test_dry_run_uses_locked_val_defaults_and_independent_outputs(self) -> None:
        result = self.run_script(["--dry-run"])
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout + result.stderr

        self.assertIn("IMAGE_DIR: data/processed/visdrone/images/val", output)
        self.assertIn("GT_LABEL_DIR: data/processed/visdrone/labels/val", output)
        self.assertIn("EXPECTED_IMAGE_COUNT: 548", output)
        self.assertIn("EXPECTED_GT_LABEL_COUNT: 548", output)
        self.assertIn("OUTPUT_ROOT: outputs/auto_label_baselines/val", output)
        self.assertIn("GROUNDING_JSON_DIR: outputs/auto_label_baselines/val/grounding_json", output)
        self.assertIn("SAM_JSON_DIR: outputs/auto_label_baselines/val/sam_json", output)
        self.assertIn("DINO_LABEL_DIR: outputs/auto_label_baselines/val/dino_only_labels", output)
        self.assertIn("DINO_SAM_LABEL_DIR: outputs/auto_label_baselines/val/dino_sam_labels", output)
        self.assertIn("VIS_DIR: results/visualizations/auto_label_baselines/val", output)
        self.assertIn("DINO_SUMMARY_CSV: /tmp/auto_label_baselines_val_script_test_tables/dino_only_val_summary.csv", output)
        self.assertIn("DINO_CLASS_CSV: /tmp/auto_label_baselines_val_script_test_tables/dino_only_val_by_class.csv", output)
        self.assertIn("DINO_SIZE_CSV: /tmp/auto_label_baselines_val_script_test_tables/dino_only_val_by_size.csv", output)
        self.assertIn("DINO_SAM_SUMMARY_CSV: /tmp/auto_label_baselines_val_script_test_tables/dino_sam_val_summary.csv", output)
        self.assertIn("DINO_SAM_CLASS_CSV: /tmp/auto_label_baselines_val_script_test_tables/dino_sam_val_by_class.csv", output)
        self.assertIn("DINO_SAM_SIZE_CSV: /tmp/auto_label_baselines_val_script_test_tables/dino_sam_val_by_size.csv", output)
        self.assertIn("EFFICIENCY_CSV: /tmp/auto_label_baselines_val_script_test_tables/auto_label_baselines_val_efficiency.csv", output)

        self.assertIn("box_threshold=0.35", output)
        self.assertIn("text_threshold=0.25", output)
        self.assertIn("min_refine_area_px=0", output)
        self.assertIn("score_threshold=0.35", output)
        self.assertIn("evaluation_iou_threshold=0.50", output)
        self.assertIn("tools/run_grounding_dino_batch.py", output)
        self.assertIn("--score-threshold 0.35", output)
        self.assertIn("--min-refine-area-px 0", output)
        self.assertIn("--iou-threshold 0.50", output)
        self.assertIn("--skip-existing", output)
        self.assertIn("No preflight checks or model inference were executed", output)

        self.assertNotIn("tools/run_grounding_dino_tiled_batch.py", output)
        self.assertNotIn("outputs/ovdas_tile_full", output)
        self.assertNotIn("val_debug", output)
        self.assertNotIn("outputs/auto_label_baselines/train", output)

    def test_locked_configuration_cannot_drift(self) -> None:
        drift_cases = {
            "BOX_THRESHOLD": "0.25",
            "TEXT_THRESHOLD": "0.30",
            "MIN_REFINE_AREA_PX": "1024",
            "SCORE_THRESHOLD": "0.25",
            "IOU_THRESHOLD": "0.75",
        }
        for name, value in drift_cases.items():
            with self.subTest(name=name):
                result = self.run_script(["--dry-run"], env={name: value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"Locked {name} changed", result.stdout + result.stderr)

    def test_forbidden_output_paths_are_rejected(self) -> None:
        forbidden_cases = {
            "OUTPUT_ROOT": "outputs/auto_label_baselines/train",
            "DINO_SUMMARY_CSV": "results/tables/ovdas_tile_val_summary.csv",
            "SAM_JSON_DIR": "outputs/sam_refine_json/val_debug",
        }
        for name, value in forbidden_cases.items():
            with self.subTest(name=name):
                result = self.run_script(["--dry-run"], env={name: value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must not", result.stdout + result.stderr)

    def test_preflight_rejects_image_count_mismatch_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            image_dir = root / "images"
            gt_label_dir = root / "labels"
            image_dir.mkdir()
            gt_label_dir.mkdir()
            result = self.run_script(
                ["--preflight-only"],
                env={
                    "IMAGE_DIR": image_dir.as_posix(),
                    "GT_LABEL_DIR": gt_label_dir.as_posix(),
                },
            )

        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Expected 548 val images", output)
        self.assertNotIn("Stage 1/", output)
        self.assertNotIn("tools/run_grounding_dino_batch.py", output)

    def test_preflight_rejects_missing_gt_label_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "sample.jpg").write_bytes(b"dummy")
            result = self.run_script(
                ["--preflight-only"],
                env={
                    "IMAGE_DIR": image_dir.as_posix(),
                    "EXPECTED_IMAGE_COUNT": "1",
                    "EXPECTED_GT_LABEL_COUNT": "1",
                    "GT_LABEL_DIR": (root / "missing_labels").as_posix(),
                },
            )

        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing GT label directory", output)
        self.assertNotIn("Stage 1/", output)

    def test_preflight_rejects_gt_label_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            image_dir = root / "images"
            gt_label_dir = root / "labels"
            image_dir.mkdir()
            gt_label_dir.mkdir()
            for index in range(548):
                (image_dir / f"image_{index:04d}.jpg").write_bytes(b"dummy")
            for index in range(547):
                (gt_label_dir / f"image_{index:04d}.txt").write_text("", encoding="utf-8")

            result = self.run_script(
                ["--preflight-only"],
                env={
                    "IMAGE_DIR": image_dir.as_posix(),
                    "GT_LABEL_DIR": gt_label_dir.as_posix(),
                },
            )

        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Expected 548 val GT label files", output)
        self.assertNotIn("Stage 1/", output)

    def test_preflight_rejects_missing_model_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            image_dir = root / "images"
            gt_label_dir = root / "labels"
            image_dir.mkdir()
            gt_label_dir.mkdir()
            (image_dir / "sample.jpg").write_bytes(b"dummy")
            (gt_label_dir / "sample.txt").write_text("", encoding="utf-8")
            classes_config = root / "classes.yaml"
            classes_config.write_text("classes:\n  - id: 0\n    name: car\n", encoding="utf-8")
            dino_config = root / "dino_config.py"
            dino_config.write_text("# dummy\n", encoding="utf-8")
            sam_checkpoint = root / "sam.pth"
            sam_checkpoint.write_bytes(b"dummy")

            result = self.run_script(
                ["--preflight-only"],
                env={
                    "IMAGE_DIR": image_dir.as_posix(),
                    "GT_LABEL_DIR": gt_label_dir.as_posix(),
                    "EXPECTED_IMAGE_COUNT": "1",
                    "EXPECTED_GT_LABEL_COUNT": "1",
                    "CLASSES_CONFIG": classes_config.as_posix(),
                    "GROUNDING_DINO_CONFIG": dino_config.as_posix(),
                    "GROUNDING_DINO_CHECKPOINT": (root / "missing_dino.pth").as_posix(),
                    "SAM_CHECKPOINT": sam_checkpoint.as_posix(),
                },
            )

        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing Grounding DINO checkpoint", output)
        self.assertNotIn("Stage 1/", output)


if __name__ == "__main__":
    unittest.main()
