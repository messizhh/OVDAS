from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "server_run_ovdas_tile_val.sh"


class ServerRunOvdasTileValScriptTest(unittest.TestCase):
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
                "LOG_DIR": "/tmp/ovdas_tile_val_script_test_logs",
                "TABLE_DIR": "/tmp/ovdas_tile_val_script_test_tables",
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

    def test_dry_run_uses_locked_val_defaults_and_output_paths(self) -> None:
        result = self.run_script(["--dry-run"])
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout + result.stderr

        self.assertIn("IMAGE_DIR: data/processed/visdrone/images/val", output)
        self.assertIn("GT_LABEL_DIR: data/processed/visdrone/labels/val", output)
        self.assertIn("EXPECTED_IMAGE_COUNT: 548", output)
        self.assertIn("EXPECTED_GT_LABEL_COUNT: 548", output)
        self.assertIn("OUTPUT_ROOT: outputs/ovdas_tile_full/val", output)
        self.assertIn("GROUNDING_JSON_DIR: outputs/ovdas_tile_full/val/grounding_json", output)
        self.assertIn("SAM_JSON_DIR: outputs/ovdas_tile_full/val/sam_json", output)
        self.assertIn("LABEL_DIR: outputs/ovdas_tile_full/val/labels", output)
        self.assertIn("VIS_DIR: results/visualizations/ovdas_tile_full/val", output)
        self.assertIn("SUMMARY_CSV: /tmp/ovdas_tile_val_script_test_tables/ovdas_tile_val_summary.csv", output)
        self.assertIn("CLASS_CSV: /tmp/ovdas_tile_val_script_test_tables/ovdas_tile_val_by_class.csv", output)
        self.assertIn("SIZE_CSV: /tmp/ovdas_tile_val_script_test_tables/ovdas_tile_val_by_size.csv", output)
        self.assertIn("EFFICIENCY_CSV: /tmp/ovdas_tile_val_script_test_tables/ovdas_tile_val_efficiency.csv", output)
        self.assertIn("--iou-threshold 0.50", output)
        self.assertIn("--skip-existing", output)
        self.assertIn("No preflight checks or model inference were executed", output)
        self.assertNotIn("outputs/ovdas_tile_full/train", output)
        self.assertNotIn("val_debug", output)

    def test_locked_configuration_cannot_drift(self) -> None:
        result = self.run_script(["--dry-run"], env={"BOX_THRESHOLD": "0.35"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Locked BOX_THRESHOLD changed", result.stdout + result.stderr)

        result = self.run_script(["--dry-run"], env={"IOU_THRESHOLD": "0.75"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Locked IOU_THRESHOLD changed", result.stdout + result.stderr)

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
        self.assertNotIn("Stage 1/4", output)
        self.assertNotIn("tools/run_grounding_dino_tiled_batch.py", output)

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
        self.assertNotIn("Stage 1/4", output)

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
        self.assertNotIn("Stage 1/4", output)


if __name__ == "__main__":
    unittest.main()
