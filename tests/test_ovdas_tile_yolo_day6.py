from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
import yaml
from pathlib import Path

from tools.prepare_auto_yolo_dataset import place_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = PROJECT_ROOT / "scripts" / "server_prepare_ovdas_tile_yolo_dataset.sh"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "server_train_yolo_ovdas_tile.sh"
PREPARE_TOOL = PROJECT_ROOT / "tools" / "prepare_auto_yolo_dataset.py"
TEST_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if not TEST_PYTHON.is_file():
    TEST_PYTHON = Path("python3")

OVDAS_MARKER = ".ovdas_tile_yolo_dataset"
DATASET_ROOT = Path("data/processed/visdrone_ovdas_tile_yolo")
DATA_CONFIG = Path("configs/yolo_visdrone_ovdas_tile.yaml")
RUN_DIR = Path("runs/yolov8s_ovdas_tile_visdrone")

VISDRONE_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "bus",
    7: "motor",
}


def write_label(path: Path, text: str = "3 0.5 0.5 0.2 0.2\n") -> None:
    path.write_text(text, encoding="utf-8")


def touch_image(path: Path) -> None:
    path.write_bytes(b"not a real image")


def write_marker(path: Path, *, dataset_id: str = "ovdas_tile_yolo", status: str = "complete") -> None:
    path.write_text(
        "created_by=tools/prepare_auto_yolo_dataset.py\n"
        f"dataset_id={dataset_id}\n"
        f"status={status}\n",
        encoding="utf-8",
    )


def write_yolo_data_config(root: Path, overrides: dict[str, object] | None = None) -> None:
    data = {
        "path": DATASET_ROOT.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": VISDRONE_NAMES,
    }
    if overrides:
        data.update(overrides)
    config_path = root / DATA_CONFIG
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def create_training_project(root: Path, *, marker_status: str = "complete") -> Path:
    (root / "tools").symlink_to(PROJECT_ROOT / "tools", target_is_directory=True)
    write_yolo_data_config(root)
    (root / "configs" / "classes_visdrone.yaml").write_text(
        (PROJECT_ROOT / "configs" / "classes_visdrone.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    dataset_root = root / DATASET_ROOT
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        (dataset_root / relative).mkdir(parents=True)
    write_marker(dataset_root / OVDAS_MARKER, status=marker_status)
    (root / "yolov8s.pt").write_bytes(b"dummy model")
    return dataset_root


def populate_fixed_dataset(dataset_root: Path) -> None:
    for index in range(6471):
        stem = f"train_{index:04d}"
        touch_image(dataset_root / "images" / "train" / f"{stem}.jpg")
        write_label(dataset_root / "labels" / "train" / f"{stem}.txt", "")
    for index in range(548):
        stem = f"val_{index:04d}"
        touch_image(dataset_root / "images" / "val" / f"{stem}.jpg")
        write_label(dataset_root / "labels" / "val" / f"{stem}.txt", "")


def write_resume_args(root: Path, overrides: dict[str, object] | None = None) -> None:
    args = {
        "data": DATA_CONFIG.as_posix(),
        "epochs": 100,
        "imgsz": 1024,
        "batch": 16,
        "seed": 0,
        "project": "runs",
        "name": "yolov8s_ovdas_tile_visdrone",
    }
    if overrides:
        args.update(overrides)
    run_dir = root / RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "weights" / "last.pt").write_bytes(b"last checkpoint")
    (run_dir / "args.yaml").write_text(yaml.safe_dump(args, sort_keys=False), encoding="utf-8")


class OvdasTileYoloPrepareScriptTest(unittest.TestCase):
    def make_sources(
        self,
        root: Path,
        *,
        train_count: int = 2,
        val_count: int = 1,
        train_label_text: str = "3 0.5 0.5 0.2 0.2\n",
    ) -> dict[str, Path]:
        train_images = root / "src" / "images" / "train"
        val_images = root / "src" / "images" / "val"
        train_labels = root / "src" / "labels" / "train"
        val_labels = root / "src" / "labels" / "val"
        for path in (train_images, val_images, train_labels, val_labels):
            path.mkdir(parents=True)

        for index in range(train_count):
            stem = f"train_{index:04d}"
            touch_image(train_images / f"{stem}.jpg")
            write_label(train_labels / f"{stem}.txt", train_label_text)
        for index in range(val_count):
            stem = f"val_{index:04d}"
            touch_image(val_images / f"{stem}.jpg")
            write_label(val_labels / f"{stem}.txt")

        return {
            "train_images": train_images,
            "train_labels": train_labels,
            "val_images": val_images,
            "val_labels": val_labels,
        }

    def run_prepare(
        self,
        temp_root: Path,
        sources: dict[str, Path],
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        out_root = temp_root / "out" / "visdrone_ovdas_tile_yolo"
        config_path = temp_root / "configs" / "yolo_visdrone_ovdas_tile.yaml"
        env = os.environ.copy()
        env.update(
            {
                "PYTHON_BIN": "python3",
                "LOG_DIR": (temp_root / "logs").as_posix(),
                "TRAIN_IMAGE_DIR": sources["train_images"].as_posix(),
                "TRAIN_LABEL_DIR": sources["train_labels"].as_posix(),
                "VAL_IMAGE_DIR": sources["val_images"].as_posix(),
                "VAL_LABEL_DIR": sources["val_labels"].as_posix(),
                "OUT_ROOT": out_root.as_posix(),
                "CONFIG_PATH": config_path.as_posix(),
                "EXPECTED_TRAIN_IMAGES": "2",
                "EXPECTED_TRAIN_LABELS": "2",
                "EXPECTED_VAL_IMAGES": "1",
                "EXPECTED_VAL_LABELS": "1",
            }
        )
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", PREPARE_SCRIPT.as_posix(), "--rebuild"],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def run_prepare_tool(
        self,
        temp_root: Path,
        sources: dict[str, Path],
        args: list[str],
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                TEST_PYTHON.as_posix(),
                PREPARE_TOOL.as_posix(),
                "--train-image-dir",
                sources["train_images"].as_posix(),
                "--train-label-dir",
                sources["train_labels"].as_posix(),
                "--val-image-dir",
                sources["val_images"].as_posix(),
                "--val-label-dir",
                sources["val_labels"].as_posix(),
                "--out-root",
                (temp_root / "out" / "validate_only_target").as_posix(),
                "--config-path",
                (temp_root / "configs" / "validate_only.yaml").as_posix(),
                "--classes-config",
                "configs/classes_visdrone.yaml",
                "--expected-train-images",
                "2",
                "--expected-train-labels",
                "2",
                "--expected-val-images",
                "1",
                "--expected-val-labels",
                "1",
                *args,
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_valid_dataset_is_copied_into_real_directories_and_empty_labels_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sources = self.make_sources(temp_root)
            write_label(sources["train_labels"] / "train_0001.txt", "")

            result = self.run_prepare(temp_root, sources)

            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            out_root = temp_root / "out" / "visdrone_ovdas_tile_yolo"
            for relative in ("images/train", "images/val", "labels/train", "labels/val"):
                target = out_root / relative
                self.assertTrue(target.is_dir(), relative)
                self.assertFalse(target.is_symlink(), relative)
            self.assertIn("train_empty_label_files: 1", output)
            self.assertIn("train_total_boxes: 1", output)
            self.assertIn("class_3: 1", output)
            marker_text = (out_root / OVDAS_MARKER).read_text(encoding="utf-8")
            self.assertIn("dataset_id=ovdas_tile_yolo", marker_text)
            self.assertIn("status=complete", marker_text)

    def test_unmarked_existing_directory_rejects_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sources = self.make_sources(temp_root)
            out_root = temp_root / "out" / "visdrone_ovdas_tile_yolo"
            out_root.mkdir(parents=True)
            (out_root / "foreign.txt").write_text("do not delete", encoding="utf-8")

            result = self.run_prepare(temp_root, sources)

            output = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("Refusing to rebuild unmarked output root", output)
            self.assertTrue((out_root / "foreign.txt").is_file())

    def test_wrong_dataset_marker_rejects_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sources = self.make_sources(temp_root)
            out_root = temp_root / "out" / "visdrone_ovdas_tile_yolo"
            out_root.mkdir(parents=True)
            write_marker(out_root / OVDAS_MARKER, dataset_id="other_dataset")

            result = self.run_prepare(temp_root, sources)

            output = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("dataset_id", output)
            self.assertTrue((out_root / OVDAS_MARKER).is_file())

    def test_building_marker_allows_rebuild_of_partial_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sources = self.make_sources(temp_root)
            out_root = temp_root / "out" / "visdrone_ovdas_tile_yolo"
            out_root.mkdir(parents=True)
            (out_root / "partial.txt").write_text("partial", encoding="utf-8")
            write_marker(out_root / OVDAS_MARKER, status="building")

            result = self.run_prepare(temp_root, sources)

            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertFalse((out_root / "partial.txt").exists())
            marker_text = (out_root / OVDAS_MARKER).read_text(encoding="utf-8")
            self.assertIn("dataset_id=ovdas_tile_yolo", marker_text)
            self.assertIn("status=complete", marker_text)

    def test_validate_only_does_not_create_delete_copy_or_write_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sources = self.make_sources(temp_root)
            out_root = temp_root / "out" / "validate_only_target"
            out_root.mkdir(parents=True)
            sentinel = out_root / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")

            result = self.run_prepare_tool(
                temp_root,
                sources,
                ["--validate-only"],
            )

            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertTrue(sentinel.is_file())
            self.assertFalse((out_root / "images").exists())
            self.assertFalse((temp_root / "configs" / "validate_only.yaml").exists())
            self.assertIn("copied_files: 0", output)
            self.assertIn("train_total_boxes: 2", output)

    def test_validate_only_reports_count_stem_class_and_coordinate_errors(self) -> None:
        cases = {
            "count": ("Expected 2 train images", lambda sources: (sources["train_images"] / "train_0001.jpg").unlink()),
            "stem": (
                "Missing 1 train label file",
                lambda sources: (
                    (sources["train_labels"] / "train_0001.txt").unlink(),
                    write_label(sources["train_labels"] / "extra.txt"),
                ),
            ),
            "class": ("class_id 8", lambda sources: write_label(sources["train_labels"] / "train_0000.txt", "8 0.5 0.5 0.2 0.2\n")),
            "coord": ("cx must be in [0, 1]", lambda sources: write_label(sources["train_labels"] / "train_0000.txt", "3 1.2 0.5 0.2 0.2\n")),
        }
        for name, (expected, mutate) in cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_root = Path(temp_dir)
                    sources = self.make_sources(temp_root)
                    mutate(sources)

                    result = self.run_prepare_tool(temp_root, sources, ["--validate-only"])

                    output = result.stdout + result.stderr
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn(expected, output)
                    self.assertFalse((temp_root / "out" / "validate_only_target" / "images").exists())

    def test_count_mismatch_fails_before_copying(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sources = self.make_sources(temp_root, train_count=1)

            result = self.run_prepare(temp_root, sources)

            output = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("Expected 2 train images", output)
            self.assertFalse((temp_root / "out" / "visdrone_ovdas_tile_yolo").exists())

    def test_stem_mismatch_and_extra_label_fail_before_copying(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sources = self.make_sources(temp_root)
            (sources["train_labels"] / "train_0001.txt").unlink()
            write_label(sources["train_labels"] / "extra.txt")

            result = self.run_prepare(temp_root, sources)

            output = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("Missing 1 train label file", output)
            self.assertIn("Extra 1 train label file", output)
            self.assertFalse((temp_root / "out" / "visdrone_ovdas_tile_yolo").exists())

    def test_invalid_yolo_label_lines_fail_with_file_and_line_number(self) -> None:
        invalid_cases = {
            "bad_columns": "3 0.5 0.5 0.2\n",
            "bad_class": "8 0.5 0.5 0.2 0.2\n",
            "nan": "3 nan 0.5 0.2 0.2\n",
            "inf": "3 0.5 inf 0.2 0.2\n",
            "bad_center": "3 1.2 0.5 0.2 0.2\n",
            "bad_size": "3 0.5 0.5 0 0.2\n",
        }
        for name, label_text in invalid_cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_root = Path(temp_dir)
                    sources = self.make_sources(temp_root, train_label_text=label_text)

                    result = self.run_prepare(temp_root, sources)

                    output = result.stdout + result.stderr
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn("train_0000.txt:1", output)
                    self.assertFalse((temp_root / "out" / "visdrone_ovdas_tile_yolo").exists())

    def test_place_directory_keeps_default_forbid_symlink_argument_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_dir = temp_root / "source"
            target_dir = temp_root / "target"
            source_dir.mkdir()
            (source_dir / "sample.txt").write_text("sample", encoding="utf-8")

            created, reused, copied = place_directory(source_dir, target_dir, "copy", False)

            self.assertEqual((created, reused, copied), (0, 0, 1))
            self.assertEqual((target_dir / "sample.txt").read_text(encoding="utf-8"), "sample")


class OvdasTileYoloTrainScriptTest(unittest.TestCase):
    def run_train(
        self,
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: Path = PROJECT_ROOT,
    ) -> subprocess.CompletedProcess[str]:
        merged_env = os.environ.copy()
        merged_env.update({"PYTHON_BIN": TEST_PYTHON.as_posix(), "LOG_DIR": "/tmp/ovdas_tile_yolo_train_test_logs"})
        if env:
            merged_env.update(env)
        return subprocess.run(
            ["bash", TRAIN_SCRIPT.as_posix(), *args],
            cwd=cwd,
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_help_is_safe(self) -> None:
        result = self.run_train(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--resume", result.stdout)

    def test_dry_run_prints_locked_training_parameters_without_running_yolo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            marker = Path(temp_dir) / "yolo_was_called"
            fake_yolo = Path(temp_dir) / "fake_yolo.sh"
            fake_yolo.write_text(f"#!/usr/bin/env bash\ntouch {marker.as_posix()}\n", encoding="utf-8")
            fake_yolo.chmod(0o755)

            result = self.run_train(["--dry-run"], env={"YOLO_BIN": fake_yolo.as_posix()})

            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertIn("model=yolov8s.pt", output)
            self.assertIn("data=configs/yolo_visdrone_ovdas_tile.yaml", output)
            self.assertIn("epochs=100", output)
            self.assertIn("imgsz=1024", output)
            self.assertIn("batch=16", output)
            self.assertIn("seed=0", output)
            self.assertIn("name=yolov8s_ovdas_tile_visdrone", output)
            self.assertFalse(marker.exists())

    def test_locked_training_configuration_cannot_drift(self) -> None:
        drift_cases = {"EPOCHS": "99", "SEED": "42", "DATA_CONFIG": "configs/yolo_visdrone_auto_dino_only.yaml"}
        for key, value in drift_cases.items():
            with self.subTest(key=key):
                result = self.run_train(["--dry-run"], env={key: value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"Locked {key} changed", result.stdout + result.stderr)

    def test_invalid_omp_threads_fails(self) -> None:
        result = self.run_train(["--dry-run"], env={"OMP_NUM_THREADS": "0"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OMP_NUM_THREADS must be a positive integer", result.stdout + result.stderr)

    def test_existing_run_directory_is_protected_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            create_training_project(root)
            (root / RUN_DIR).mkdir(parents=True)

            result = self.run_train(["--preflight-only"], env={"PROJECT_ROOT": root.as_posix()}, cwd=root)

            output = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("Run directory already exists", output)

    def test_training_rejects_missing_building_or_wrong_dataset_marker(self) -> None:
        cases = {
            "missing": (None, "Missing OVDAS-Tile dataset marker"),
            "building": ("building", "status=complete"),
            "wrong": ("wrong_id", "dataset_id"),
        }
        for name, (marker_case, expected) in cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    dataset_root = create_training_project(root)
                    marker_path = dataset_root / OVDAS_MARKER
                    if marker_case is None:
                        marker_path.unlink()
                    elif marker_case == "building":
                        write_marker(marker_path, status="building")
                    else:
                        write_marker(marker_path, dataset_id="other_dataset")

                    result = self.run_train(["--preflight-only"], env={"PROJECT_ROOT": root.as_posix()}, cwd=root)

                    output = result.stdout + result.stderr
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn(expected, output)

    def test_training_rejects_yolo_yaml_drift(self) -> None:
        drift_cases = {
            "path": {"path": "data/processed/other"},
            "train": {"train": "train_images"},
            "val": {"val": "val_images"},
            "names": {"names": {0: "car"}},
        }
        for name, overrides in drift_cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    create_training_project(root)
                    write_yolo_data_config(root, overrides)

                    result = self.run_train(["--preflight-only"], env={"PROJECT_ROOT": root.as_posix()}, cwd=root)

                    output = result.stdout + result.stderr
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn("Locked YOLO data config changed", output)

    def test_preflight_only_does_not_call_yolo_on_valid_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = create_training_project(root)
            populate_fixed_dataset(dataset_root)
            yolo_marker = root / "yolo_called"
            fake_yolo = root / "fake_yolo.sh"
            fake_yolo.write_text(f"#!/usr/bin/env bash\ntouch {yolo_marker.as_posix()}\n", encoding="utf-8")
            fake_yolo.chmod(0o755)

            result = self.run_train(
                ["--preflight-only"],
                env={"PROJECT_ROOT": root.as_posix(), "YOLO_BIN": fake_yolo.as_posix()},
                cwd=root,
            )

            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertIn("Preflight completed", output)
            self.assertIn("train_images: 6471", output)
            self.assertFalse(yolo_marker.exists())

    def test_training_preflight_finds_tampered_copied_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = create_training_project(root)
            touch_image(dataset_root / "images" / "train" / "train_0000.jpg")
            write_label(dataset_root / "labels" / "train" / "train_0000.txt", "8 0.5 0.5 0.2 0.2\n")

            result = self.run_train(["--preflight-only"], env={"PROJECT_ROOT": root.as_posix()}, cwd=root)

            output = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn("Expected 6471 train images", output)

    def test_resume_rejects_custom_checkpoint_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            create_training_project(root)

            result = self.run_train(
                ["--resume", "--preflight-only"],
                env={"PROJECT_ROOT": root.as_posix(), "RESUME_MODEL": "runs/other/weights/last.pt"},
                cwd=root,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Locked RESUME_MODEL changed", result.stdout + result.stderr)

    def test_resume_rejects_missing_or_mismatched_args_yaml(self) -> None:
        cases = {
            "missing": (None, "Missing resume args"),
            "mismatch": ({"seed": 42}, "Resume args mismatch for seed"),
        }
        for name, (overrides, expected) in cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    create_training_project(root)
                    write_resume_args(root, overrides if overrides is not None else {})
                    if overrides is None:
                        (root / RUN_DIR / "args.yaml").unlink()

                    result = self.run_train(["--resume", "--preflight-only"], env={"PROJECT_ROOT": root.as_posix()}, cwd=root)

                    output = result.stdout + result.stderr
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn(expected, output)

    def test_legal_resume_dry_run_and_preflight_do_not_start_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = create_training_project(root)
            populate_fixed_dataset(dataset_root)
            write_resume_args(root)
            yolo_marker = root / "yolo_called"
            fake_yolo = root / "fake_yolo.sh"
            fake_yolo.write_text(f"#!/usr/bin/env bash\ntouch {yolo_marker.as_posix()}\n", encoding="utf-8")
            fake_yolo.chmod(0o755)

            dry_run = self.run_train(
                ["--resume", "--dry-run"],
                env={"PROJECT_ROOT": root.as_posix(), "YOLO_BIN": fake_yolo.as_posix()},
                cwd=root,
            )
            preflight = self.run_train(
                ["--resume", "--preflight-only"],
                env={"PROJECT_ROOT": root.as_posix(), "YOLO_BIN": fake_yolo.as_posix()},
                cwd=root,
            )

            self.assertEqual(dry_run.returncode, 0, dry_run.stdout + dry_run.stderr)
            self.assertEqual(preflight.returncode, 0, preflight.stdout + preflight.stderr)
            self.assertIn("resume=True", dry_run.stdout + dry_run.stderr)
            self.assertIn("Preflight completed", preflight.stdout + preflight.stderr)
            self.assertFalse(yolo_marker.exists())


if __name__ == "__main__":
    unittest.main()
