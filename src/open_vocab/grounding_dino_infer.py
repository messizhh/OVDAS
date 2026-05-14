"""Grounding DINO single-image inference helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class GroundingDinoDetection:
    """One Grounding DINO detection result."""

    bbox_cxcywh_norm: list[float]
    bbox_xyxy: list[float]
    score: float
    phrase: str


@dataclass
class GroundingDinoResult:
    """Grounding DINO result for one image."""

    image_path: str
    prompt: str
    box_threshold: float
    text_threshold: float
    device: str
    detections: list[GroundingDinoDetection]


def validate_input_paths(
    image_path: Path,
    config_file: Path,
    checkpoint: Path,
) -> None:
    """Validate image, model config, and checkpoint paths."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if not config_file.is_file():
        raise FileNotFoundError(f"Grounding DINO config file does not exist: {config_file}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Grounding DINO checkpoint does not exist: {checkpoint}")


def resolve_device(device: str) -> str:
    """Validate and normalize the requested inference device."""
    normalized = device.strip().lower()
    if normalized == "cpu":
        return "cpu"

    if normalized.startswith("cuda"):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is required to use CUDA device selection.") from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{device}', but CUDA is not available in this environment."
            )
        return device

    return device


def load_grounding_dino_api() -> tuple[Any, Any, Any]:
    """Load Grounding DINO inference API lazily."""
    try:
        from groundingdino.util.inference import load_image, load_model, predict
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Grounding DINO is not installed. Install it on the machine where "
            "single-image inference will run, then retry this command."
        ) from exc

    return load_model, load_image, predict


def tensor_to_list(value: Any) -> list[Any]:
    """Convert a tensor-like object to a plain Python list."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def read_image_size(image_path: Path) -> tuple[int, int]:
    """Read image width and height with Pillow."""
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required to read and visualize images.") from exc

    with Image.open(image_path) as image:
        return image.size


def cxcywh_norm_to_xyxy(
    box: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Convert normalized cxcywh box to clipped pixel xyxy box."""
    x_center, y_center, width, height = box
    x1 = (x_center - width / 2.0) * image_width
    y1 = (y_center - height / 2.0) * image_height
    x2 = (x_center + width / 2.0) * image_width
    y2 = (y_center + height / 2.0) * image_height

    return [
        max(0.0, min(float(image_width), x1)),
        max(0.0, min(float(image_height), y1)),
        max(0.0, min(float(image_width), x2)),
        max(0.0, min(float(image_height), y2)),
    ]


def run_grounding_dino_single(
    image_path: Path,
    prompt: str,
    config_file: Path,
    checkpoint: Path,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> GroundingDinoResult:
    """Run Grounding DINO inference for one image."""
    validate_input_paths(image_path, config_file, checkpoint)
    resolved_device = resolve_device(device)
    load_model, load_image, predict = load_grounding_dino_api()

    model = load_model(
        model_config_path=str(config_file),
        model_checkpoint_path=str(checkpoint),
        device=resolved_device,
    )
    image_source, image_tensor = load_image(str(image_path))
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=resolved_device,
    )

    image_width, image_height = read_image_size(image_path)
    box_rows = tensor_to_list(boxes)
    score_rows = tensor_to_list(logits)
    phrase_rows = list(phrases)

    detections: list[GroundingDinoDetection] = []
    for box, score, phrase in zip(box_rows, score_rows, phrase_rows):
        box_values = [float(value) for value in box]
        detections.append(
            GroundingDinoDetection(
                bbox_cxcywh_norm=box_values,
                bbox_xyxy=cxcywh_norm_to_xyxy(box_values, image_width, image_height),
                score=float(score),
                phrase=str(phrase),
            )
        )

    return GroundingDinoResult(
        image_path=image_path.as_posix(),
        prompt=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=resolved_device,
        detections=detections,
    )


def save_result_json(result: GroundingDinoResult, output_json: Path) -> None:
    """Save a Grounding DINO result to JSON."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_visualization(
    image_path: Path,
    detections: list[GroundingDinoDetection],
    output_image: Path,
) -> None:
    """Save a visualization image with predicted boxes and phrases."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required to save visualization images.") from exc

    with Image.open(image_path) as image:
        canvas = image.convert("RGB")

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    output_image.parent.mkdir(parents=True, exist_ok=True)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            continue
        label = f"{detection.phrase} {detection.score:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=2)
        draw.text((x1, max(0.0, y1 - 12.0)), label, fill=(255, 255, 255), font=font)

    canvas.save(output_image, quality=95)
