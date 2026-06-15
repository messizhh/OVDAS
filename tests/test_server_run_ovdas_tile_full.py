from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "server_run_ovdas_tile_full.sh"


class ServerRunOvdasTileFullScriptTest(unittest.TestCase):
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
                "LOG_DIR": "/tmp/ovdas_tile_script_test_logs",
                "TABLE_DIR": "/tmp/ovdas_tile_script_test_tables",
                "OUTPUT_ROOT": "/tmp/ovdas_tile_script_test_outputs",
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
        self.assertEqual(result.returncode, 0)
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--preflight-only", result.stdout)

    def test_dry_run_prints_locked_commands_without_preflight(self) -> None:
        result = self.run_script(
            ["--dry-run"],
            env={
                "OUTPUT_ROOT": "outputs/test dry run/ovdas_tile_full",
                "TABLE_DIR": "results/tables",
                "LOG_DIR": "logs",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("box_threshold=0.25", result.stdout)
        self.assertIn("--tile-size 640", result.stdout)
        self.assertIn("--enable-size-aware-filter", result.stdout)
        self.assertIn("No preflight checks or model inference were executed", result.stdout)

    def test_invalid_omp_threads_fails_before_dry_run(self) -> None:
        result = self.run_script(["--dry-run"], env={"OMP_NUM_THREADS": "0"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OMP_NUM_THREADS must be a positive integer", result.stderr)

    def test_preflight_rejects_image_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            image_dir = Path(temp_root) / "images"
            image_dir.mkdir()
            result = self.run_script(
                ["--preflight-only"],
                env={
                    "IMAGE_DIR": image_dir.as_posix(),
                    "EXPECTED_IMAGE_COUNT": "6471",
                },
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Expected 6471 train images", result.stdout + result.stderr)

    def test_preflight_rejects_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "sample.jpg").write_bytes(b"not a real image")
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
                    "EXPECTED_IMAGE_COUNT": "1",
                    "CLASSES_CONFIG": classes_config.as_posix(),
                    "GROUNDING_DINO_CONFIG": dino_config.as_posix(),
                    "GROUNDING_DINO_CHECKPOINT": (root / "missing_dino.pth").as_posix(),
                    "SAM_CHECKPOINT": sam_checkpoint.as_posix(),
                },
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing Grounding DINO checkpoint", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
