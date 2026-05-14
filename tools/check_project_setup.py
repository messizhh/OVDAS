"""Lightweight project setup checker for OVDAS Day 1."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "configs",
    "data",
    "data/samples/images",
    "data/samples/labels",
    "src",
    "tools",
    "scripts",
    "outputs",
    "results",
    "figures",
    "report",
    "tests",
    "logs",
]

REQUIRED_FILES = [
    "AGENTS.md",
    "README.md",
    "requirements.txt",
    "environment.yml",
    "configs/default.yaml",
    "configs/classes_visdrone.yaml",
    "data/README.md",
    ".gitignore",
]

DEFAULT_YAML_KEYS = [
    "project",
    "data",
    "classes",
    "grounding_dino",
    "sam",
    "auto_label",
    "evaluation",
    "yolo",
    "paths",
]

VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "bus",
    "motor",
]

ABSOLUTE_PATH_MARKERS = [
    "/home/",
    "/mnt/",
    "C:\\",
    "D:\\",
]


def check_required_dirs() -> list[str]:
    """Return errors for missing required project directories."""
    errors: list[str] = []
    for relative_path in REQUIRED_DIRS:
        path = PROJECT_ROOT / relative_path
        if not path.is_dir():
            errors.append(f"Missing directory: {relative_path}/")
    return errors


def check_required_files() -> list[str]:
    """Return errors for missing required project files."""
    errors: list[str] = []
    for relative_path in REQUIRED_FILES:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            errors.append(f"Missing file: {relative_path}")
    return errors


def load_yaml(relative_path: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Load a YAML file and return its mapping data plus any errors."""
    if yaml is None:
        return None, ["Missing Python dependency: pyyaml"]

    path = PROJECT_ROOT / relative_path
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, [f"Cannot read YAML file because it is missing: {relative_path}"]
    except yaml.YAMLError as exc:
        return None, [f"Invalid YAML in {relative_path}: {exc}"]

    if not isinstance(data, dict):
        return None, [f"YAML root must be a mapping: {relative_path}"]

    return data, []


def check_default_yaml(data: dict[str, Any] | None) -> list[str]:
    """Return errors for missing top-level keys in configs/default.yaml."""
    if data is None:
        return []

    errors: list[str] = []
    for key in DEFAULT_YAML_KEYS:
        if key not in data:
            errors.append(f"Missing top-level key in configs/default.yaml: {key}")
    return errors


def extract_class_names(classes_data: dict[str, Any] | None) -> set[str]:
    """Extract class names from configs/classes_visdrone.yaml."""
    if classes_data is None:
        return set()

    classes = classes_data.get("classes", [])
    names: set[str] = set()
    if not isinstance(classes, list):
        return names

    for item in classes:
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.add(item["name"])

    return names


def check_visdrone_classes(classes_data: dict[str, Any] | None) -> list[str]:
    """Return errors for missing required VisDrone target classes."""
    names = extract_class_names(classes_data)
    missing = [name for name in VISDRONE_CLASSES if name not in names]
    if missing:
        return [
            "Missing classes in configs/classes_visdrone.yaml: "
            + ", ".join(missing)
        ]
    return []


def check_absolute_path_markers(relative_paths: list[str]) -> list[str]:
    """Return errors for obvious local absolute path markers in config files."""
    errors: list[str] = []
    for relative_path in relative_paths:
        path = PROJECT_ROOT / relative_path
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue

        for marker in ABSOLUTE_PATH_MARKERS:
            if marker in text:
                errors.append(
                    f"Config file contains local absolute path marker "
                    f"{marker!r}: {relative_path}"
                )
    return errors


def run_checks() -> list[str]:
    """Run all project setup checks and return collected errors."""
    errors: list[str] = []

    errors.extend(check_required_dirs())
    errors.extend(check_required_files())

    default_config, default_errors = load_yaml("configs/default.yaml")
    classes_config, classes_errors = load_yaml("configs/classes_visdrone.yaml")
    errors.extend(default_errors)
    errors.extend(classes_errors)

    errors.extend(check_default_yaml(default_config))
    errors.extend(check_visdrone_classes(classes_config))
    errors.extend(
        check_absolute_path_markers(
            ["configs/default.yaml", "configs/classes_visdrone.yaml"]
        )
    )

    return errors


def main() -> int:
    """Run the command-line checker."""
    errors = run_checks()
    if not errors:
        print("Project setup check passed.")
        return 0

    print("Project setup check failed:")
    for error in errors:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
