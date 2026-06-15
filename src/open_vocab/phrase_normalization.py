"""Shared phrase cleaning and VisDrone class alias helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PHRASE_ALIASES = {
    "car van": "car",
    "car van truck": "car",
    "van truck": "van",
    "pedestrian people": "people",
}


def clean_phrase(value: Any) -> str:
    """Normalize an open-vocabulary phrase for class lookup."""
    text = str(value).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = text.strip(" .。,:;[](){}\"'")
    return " ".join(text.split())


def normalize_phrase_alias(phrase: str) -> tuple[str, bool]:
    """Apply a known alias after phrase cleaning."""
    cleaned = clean_phrase(phrase)
    alias_target = PHRASE_ALIASES.get(cleaned)
    if alias_target is None:
        return cleaned, False
    return alias_target, True


def load_class_mapping(classes_config: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load phrase-to-class-id mapping from classes_visdrone.yaml."""
    with classes_config.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {classes_config}")

    classes = data.get("classes", [])
    if not isinstance(classes, list):
        raise ValueError(f"'classes' must be a list in {classes_config}")

    phrase_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}
    for index, item in enumerate(classes):
        if isinstance(item, str):
            class_id = index
            name = item
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            raw_id = item.get("id", index)
            if not isinstance(raw_id, int):
                raise ValueError(f"Invalid class id in {classes_config}: {raw_id}")
            class_id = raw_id
            name = item["name"]
        else:
            raise ValueError(f"Invalid class entry in {classes_config}: {item}")

        cleaned = clean_phrase(name)
        if not cleaned:
            raise ValueError(f"Empty class name in {classes_config}: {item}")
        phrase_to_id[cleaned] = class_id
        id_to_name[class_id] = cleaned

    if not phrase_to_id:
        raise ValueError(f"No classes found in {classes_config}")
    return phrase_to_id, id_to_name


def resolve_class_id(
    phrase: str,
    phrase_to_id: dict[str, int],
) -> tuple[int | None, bool]:
    """Resolve a cleaned phrase to class id and whether an alias was used."""
    cleaned = clean_phrase(phrase)
    class_id = phrase_to_id.get(cleaned)
    if class_id is not None:
        return class_id, False

    alias_target = PHRASE_ALIASES.get(cleaned)
    if alias_target is not None:
        return phrase_to_id.get(alias_target), True

    combined_class_id = resolve_combined_class_id(cleaned, phrase_to_id)
    if combined_class_id is None:
        return None, False
    return combined_class_id, True


def resolve_combined_class_id(
    cleaned_phrase: str,
    phrase_to_id: dict[str, int],
) -> int | None:
    """Resolve a multi-class phrase by full-word class matches in config order."""
    phrase_words = cleaned_phrase.split()
    if not phrase_words:
        return None

    for class_name, class_id in phrase_to_id.items():
        class_words = class_name.split()
        if class_words and contains_word_sequence(phrase_words, class_words):
            return class_id
    return None


def contains_word_sequence(words: list[str], target: list[str]) -> bool:
    """Return whether target appears as a complete-word contiguous sequence."""
    if len(target) > len(words):
        return False
    end = len(words) - len(target) + 1
    for start in range(end):
        if words[start : start + len(target)] == target:
            return True
    return False
