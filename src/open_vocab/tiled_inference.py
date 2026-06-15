"""Tiling, coordinate mapping, and class-aware NMS for OVDAS-Tile."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.open_vocab.phrase_normalization import clean_phrase, normalize_phrase_alias, resolve_class_id


@dataclass(frozen=True)
class Tile:
    """One image tile in original image coordinates."""

    index: int
    xyxy: tuple[int, int, int, int]

    @property
    def width(self) -> int:
        """Return tile width."""
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> int:
        """Return tile height."""
        return self.xyxy[3] - self.xyxy[1]


def validate_tiling_params(tile_size: int, overlap_ratio: float) -> None:
    """Validate tiling parameters."""
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if overlap_ratio < 0.0 or overlap_ratio >= 1.0:
        raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}")


def axis_positions(length: int, tile_size: int, overlap_ratio: float) -> list[int]:
    """Return deterministic tile starts for one axis with edge coverage."""
    validate_tiling_params(tile_size, overlap_ratio)
    if length <= 0:
        raise ValueError(f"Image dimension must be positive, got {length}")
    if length <= tile_size:
        return [0]

    stride = max(1, int(round(tile_size * (1.0 - overlap_ratio))))
    positions = [0]
    while positions[-1] + tile_size < length:
        next_pos = positions[-1] + stride
        if next_pos + tile_size >= length:
            next_pos = length - tile_size
        if next_pos <= positions[-1]:
            break
        positions.append(next_pos)
    return positions


def generate_tiles(
    image_width: int,
    image_height: int,
    tile_size: int = 640,
    overlap_ratio: float = 0.20,
) -> list[Tile]:
    """Generate row-major tiles that fully cover an arbitrary image."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size: {image_width}x{image_height}")
    x_positions = axis_positions(image_width, tile_size, overlap_ratio)
    y_positions = axis_positions(image_height, tile_size, overlap_ratio)

    tiles: list[Tile] = []
    for y in y_positions:
        for x in x_positions:
            tiles.append(
                Tile(
                    index=len(tiles),
                    xyxy=(
                        x,
                        y,
                        min(image_width, x + tile_size),
                        min(image_height, y + tile_size),
                    ),
                )
            )
    return tiles


def clip_bbox_xyxy(
    value: Any,
    image_width: int,
    image_height: int,
) -> list[float] | None:
    """Convert a bbox-like value to a clipped valid xyxy pixel box."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None

    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def map_tile_bbox_to_image(
    tile_bbox_xyxy: Any,
    tile_xyxy: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> list[float] | None:
    """Map one tile-local bbox back to original image coordinates."""
    tile_bbox = clip_bbox_xyxy(
        tile_bbox_xyxy,
        image_width=tile_xyxy[2] - tile_xyxy[0],
        image_height=tile_xyxy[3] - tile_xyxy[1],
    )
    if tile_bbox is None:
        return None

    x_offset, y_offset = tile_xyxy[0], tile_xyxy[1]
    mapped = [
        tile_bbox[0] + x_offset,
        tile_bbox[1] + y_offset,
        tile_bbox[2] + x_offset,
        tile_bbox[3] + y_offset,
    ]
    return clip_bbox_xyxy(mapped, image_width, image_height)


def bbox_area_xyxy(bbox: list[float]) -> float:
    """Return bbox area."""
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
    """Compute IoU for two xyxy boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = bbox_area_xyxy(a) + bbox_area_xyxy(b) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def normalize_detection_for_merge(
    detection: dict[str, Any],
    image_width: int,
    image_height: int,
    phrase_to_id: dict[str, int],
    id_to_name: dict[int, str],
    source: str,
    tile: Tile | None,
) -> dict[str, Any] | None:
    """Normalize one detection before class-aware NMS."""
    bbox = clip_bbox_xyxy(detection.get("bbox_xyxy"), image_width, image_height)
    if bbox is None:
        return None

    original_phrase = str(detection.get("phrase", ""))
    cleaned_phrase = clean_phrase(original_phrase)
    normalized_phrase, alias_used = normalize_phrase_alias(cleaned_phrase)
    class_id, used_class_alias = resolve_class_id(cleaned_phrase, phrase_to_id)
    if class_id is not None:
        normalized_phrase = id_to_name.get(class_id, normalized_phrase)

    try:
        score = float(detection.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0

    normalized = dict(detection)
    normalized.update(
        {
            "bbox_xyxy": bbox,
            "score": score,
            "phrase": normalized_phrase,
            "original_phrase": original_phrase,
            "normalized_phrase": normalized_phrase,
            "class_id": class_id,
            "class_name": id_to_name.get(class_id) if class_id is not None else None,
            "phrase_alias_used": alias_used or used_class_alias,
            "source": source,
            "tile_xyxy": list(tile.xyxy) if tile is not None else None,
            "tile_index": tile.index if tile is not None else None,
            "original_score": score,
            "merged": False,
            "merged_count": 0,
        }
    )
    return normalized


def class_aware_nms(detections: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    """Run NMS independently for each normalized class."""
    if iou_threshold < 0.0 or iou_threshold > 1.0:
        raise ValueError(f"iou_threshold must be between 0 and 1, got {iou_threshold}")
    if not detections:
        return []

    groups: dict[str, list[dict[str, Any]]] = {}
    for detection in detections:
        class_id = detection.get("class_id")
        if class_id is None:
            key = f"phrase:{detection.get('normalized_phrase', '')}"
        else:
            key = f"class:{class_id}"
        groups.setdefault(key, []).append(detection)

    kept_all: list[dict[str, Any]] = []
    for group in groups.values():
        pending = sorted(group, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        kept_group: list[dict[str, Any]] = []
        while pending:
            current = pending.pop(0)
            survivors: list[dict[str, Any]] = []
            for candidate in pending:
                iou = bbox_iou_xyxy(current["bbox_xyxy"], candidate["bbox_xyxy"])
                if iou > iou_threshold:
                    current["merged"] = True
                    current["merged_count"] = int(current.get("merged_count", 0)) + 1
                    continue
                survivors.append(candidate)
            kept_group.append(current)
            pending = survivors
        kept_all.extend(kept_group)

    kept_all.sort(
        key=lambda item: (
            Path(str(item.get("image_path", ""))).name,
            int(item.get("tile_index") if item.get("tile_index") is not None else -1),
            -float(item.get("score", 0.0)),
        )
    )
    return kept_all


def build_tiled_result(
    image_path: Path,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap_ratio: float,
    include_full_image: bool,
    merge_iou: float,
    detections: list[dict[str, Any]],
    raw_detection_count: int,
    full_detection_count: int,
    tile_detection_count: int,
    inference_time_sec: float,
) -> dict[str, Any]:
    """Build a Grounding DINO compatible tiled result JSON object."""
    return {
        "image_path": image_path.as_posix(),
        "prompt": prompt,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "device": device,
        "image_width": image_width,
        "image_height": image_height,
        "inference_time_sec": inference_time_sec,
        "detections": detections,
        "tiled_inference": {
            "method": "OVDAS-Tile candidates",
            "tile_size": tile_size,
            "overlap_ratio": overlap_ratio,
            "include_full_image": include_full_image,
            "merge_iou": merge_iou,
            "raw_detection_count": raw_detection_count,
            "full_detection_count": full_detection_count,
            "tile_detection_count": tile_detection_count,
            "merged_detection_count": len(detections),
            "inference_time_sec": inference_time_sec,
        },
    }


def save_tiled_result_json(result: dict[str, Any], output_json: Path) -> None:
    """Save one tiled Grounding DINO JSON result."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def tile_to_json(tile: Tile) -> dict[str, Any]:
    """Convert one tile dataclass to a JSON-friendly dict."""
    return asdict(tile)
