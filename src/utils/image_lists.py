"""Helpers for deterministic image-list based experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def normalize_extensions(image_exts: Iterable[str]) -> list[str]:
    """Return lowercase extensions with a leading dot, preserving order."""
    normalized: list[str] = []
    seen: set[str] = set()
    for item in image_exts:
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext in seen:
            continue
        normalized.append(ext)
        seen.add(ext)
    if not normalized:
        raise ValueError("At least one image extension is required.")
    return normalized


def read_image_list(image_list: Path) -> list[str]:
    """Read image names or paths from a txt or JSON list file."""
    if not image_list.is_file():
        raise FileNotFoundError(f"Image list does not exist: {image_list}")

    if image_list.suffix.lower() == ".json":
        data = json.loads(image_list.read_text(encoding="utf-8"))
        return _entries_from_json(data, image_list)

    entries: list[str] = []
    for line in image_list.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        entries.append(value)
    return entries


def _entries_from_json(data: Any, image_list: Path) -> list[str]:
    """Extract image-list entries from supported JSON shapes."""
    if isinstance(data, list):
        raw_entries = data
    elif isinstance(data, dict) and isinstance(data.get("images"), list):
        raw_entries = data["images"]
    else:
        raise ValueError(
            "JSON image list must be a list or an object with an 'images' list: "
            f"{image_list}"
        )

    entries: list[str] = []
    for item in raw_entries:
        if isinstance(item, str):
            value = item
        elif isinstance(item, dict):
            value = _entry_from_dict(item)
        else:
            raise ValueError(f"Invalid image-list entry in {image_list}: {item}")
        value = value.strip()
        if value:
            entries.append(value)
    return entries


def _entry_from_dict(item: dict[str, Any]) -> str:
    """Return the first supported path/name field from one JSON entry."""
    for key in ("image_path", "path", "file_path", "file_name", "name", "image_name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"Image-list object has no supported path/name field: {item}")


def resolve_image_entries(
    image_dir: Path,
    entries: list[str],
    image_exts: Iterable[str],
) -> list[Path]:
    """Resolve image-list entries against an image directory."""
    exts = normalize_extensions(image_exts)
    image_paths: list[Path] = []
    for entry in entries:
        image_paths.append(resolve_one_image_entry(image_dir, entry, exts))
    return image_paths


def resolve_one_image_entry(image_dir: Path, entry: str, image_exts: list[str]) -> Path:
    """Resolve one image-list entry to an existing file."""
    raw = Path(entry)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(raw)
        candidates.append(image_dir / raw)
        candidates.append(image_dir / raw.name)

    if raw.suffix:
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    else:
        for candidate in candidates:
            for ext in image_exts:
                with_ext = candidate.with_suffix(ext)
                if with_ext.is_file():
                    return with_ext

    checked = ", ".join(path.as_posix() for path in candidates[:3])
    raise FileNotFoundError(f"Cannot resolve image-list entry '{entry}'. Checked: {checked}")
