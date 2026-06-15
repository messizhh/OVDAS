"""SAM bbox refinement helpers for Grounding DINO detections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SamRefineOutput:
    """Refined result plus in-memory masks for visualization."""

    result: dict[str, Any]
    masks: list[Any | None]
    total_detections: int
    refined_detections: int


def validate_refine_inputs(image_path: Path, dino_json_path: Path, checkpoint: Path) -> None:
    """Validate input image, Grounding DINO JSON, and SAM checkpoint paths."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if not dino_json_path.is_file():
        raise FileNotFoundError(f"Grounding DINO JSON does not exist: {dino_json_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM checkpoint does not exist: {checkpoint}")


def load_dino_json(dino_json_path: Path) -> dict[str, Any]:
    """Load one Grounding DINO JSON result."""
    try:
        data = json.loads(dino_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {dino_json_path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Grounding DINO JSON root must be an object: {dino_json_path}")
    detections = data.get("detections", [])
    if not isinstance(detections, list):
        raise ValueError(f"'detections' must be a list in {dino_json_path}")
    return data


def save_refined_json(result: dict[str, Any], output_json: Path) -> None:
    """Save one refined SAM JSON result."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_device(device: str) -> str:
    """Validate and normalize the requested SAM device."""
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


def load_sam_api() -> tuple[Any, Any]:
    """Load the Segment Anything API lazily."""
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Segment Anything is not installed. Install the 'segment_anything' package "
            "on the GPU server before running SAM refinement."
        ) from exc
    return SamPredictor, sam_model_registry


def load_image_rgb(image_path: Path) -> tuple[Any, int, int]:
    """Load an image as an RGB array for SAM and return width and height."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv-python is required to load images for SAM.") from exc

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise OSError(f"Failed to read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    return image_rgb, int(width), int(height)


def coerce_bbox_xyxy(value: Any, image_width: int, image_height: int) -> list[float] | None:
    """Convert a bbox_xyxy-like value to a clipped valid pixel box."""
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


def mask_to_bbox_xyxy(mask: Any) -> list[float] | None:
    """Convert a binary mask to a pixel xyxy bbox."""
    if mask is None or not bool(mask.any()):
        return None

    y_indices, x_indices = mask.nonzero()
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    x1 = float(x_indices.min())
    y1 = float(y_indices.min())
    x2 = float(x_indices.max() + 1)
    y2 = float(y_indices.max() + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def build_result(
    image_path: Path,
    dino_json_path: Path,
    dino_data: dict[str, Any],
    detections: list[dict[str, Any]],
    model_type: str,
    checkpoint: Path,
    device: str,
    save_mask: bool,
    min_refine_area_px: float = 0.0,
) -> dict[str, Any]:
    """Build the top-level refined JSON while preserving source fields."""
    result = dict(dino_data)
    result["image_path"] = dino_data.get("image_path", image_path.as_posix())
    result["detections"] = detections
    result["sam_refine"] = {
        "input_image": image_path.as_posix(),
        "input_dino_json": dino_json_path.as_posix(),
        "model_type": model_type,
        "checkpoint": checkpoint.as_posix(),
        "device": device,
        "save_mask": save_mask,
        "min_refine_area_px": min_refine_area_px,
    }
    return result


def build_empty_refine_output(
    image_path: Path,
    dino_json_path: Path,
    dino_data: dict[str, Any],
    model_type: str,
    checkpoint: Path,
    device: str,
    save_mask: bool,
    min_refine_area_px: float = 0.0,
) -> SamRefineOutput:
    """Build a valid refined result for images with no detections."""
    result = build_result(
        image_path=image_path,
        dino_json_path=dino_json_path,
        dino_data=dino_data,
        detections=[],
        model_type=model_type,
        checkpoint=checkpoint,
        device=device,
        save_mask=save_mask,
        min_refine_area_px=min_refine_area_px,
    )
    return SamRefineOutput(
        result=result,
        masks=[],
        total_detections=0,
        refined_detections=0,
    )


def save_mask_png(mask: Any, output_path: Path) -> None:
    """Save one binary mask as a PNG image."""
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required to save mask PNG files.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_image = (mask.astype("uint8") * 255)
    Image.fromarray(mask_image).save(output_path)


def bbox_area_xyxy(bbox: list[float]) -> float:
    """Return bbox area in pixels."""
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def fallback_detection(
    detection: dict[str, Any],
    fallback_bbox: Any,
    status: str,
) -> dict[str, Any]:
    """Record a fallback refine result without dropping the detection."""
    detection.update(
        {
            "refined_bbox_xyxy": fallback_bbox,
            "mask_area": 0,
            "sam_score": None,
            "mask_path": None,
            "refine_status": status,
        }
    )
    return detection


class SamBoxRefiner:
    """Reusable SAM predictor for bbox-guided mask refinement."""

    def __init__(self, checkpoint: Path, model_type: str, device: str) -> None:
        """Load SAM once for repeated image refinement."""
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM checkpoint does not exist: {checkpoint}")

        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = resolve_device(device)

        SamPredictor, sam_model_registry = load_sam_api()
        if model_type not in sam_model_registry:
            available = ", ".join(sorted(sam_model_registry.keys()))
            raise ValueError(f"Unknown SAM model type '{model_type}'. Available: {available}")

        sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam_model.to(device=self.device)
        self._predictor = SamPredictor(sam_model)

    def refine_loaded_json(
        self,
        image_path: Path,
        dino_json_path: Path,
        dino_data: dict[str, Any],
        save_mask: bool = False,
        mask_output_dir: Path | None = None,
        min_refine_area_px: float = 1024.0,
    ) -> SamRefineOutput:
        """Refine detections from an already loaded Grounding DINO result."""
        if save_mask and mask_output_dir is None:
            raise ValueError("--mask-output-dir is required when --save-mask is set.")
        if min_refine_area_px < 0.0:
            raise ValueError("--min-refine-area-px must be greater than or equal to 0.")

        image_rgb, image_width, image_height = load_image_rgb(image_path)

        source_detections = dino_data.get("detections", [])
        if not isinstance(source_detections, list):
            raise ValueError(f"'detections' must be a list in {dino_json_path}")

        refined_detections: list[dict[str, Any]] = []
        masks: list[Any | None] = []
        refined_count = 0
        image_is_set = False

        for index, source_detection in enumerate(source_detections):
            if isinstance(source_detection, dict):
                detection = dict(source_detection)
            else:
                detection = {"raw_detection": source_detection}

            bbox = coerce_bbox_xyxy(detection.get("bbox_xyxy"), image_width, image_height)
            if bbox is None:
                refined_detections.append(
                    fallback_detection(
                        detection=detection,
                        fallback_bbox=detection.get("bbox_xyxy"),
                        status="invalid_bbox_fallback",
                    )
                )
                masks.append(None)
                continue
            if bbox_area_xyxy(bbox) < min_refine_area_px:
                refined_detections.append(
                    fallback_detection(
                        detection=detection,
                        fallback_bbox=bbox,
                        status="skipped_small",
                    )
                )
                masks.append(None)
                continue

            try:
                if not image_is_set:
                    self._predictor.set_image(image_rgb)
                    image_is_set = True
                mask, sam_score = self.predict_mask(bbox)
                mask_area = int(mask.sum())
                refined_bbox = mask_to_bbox_xyxy(mask)
            except Exception as exc:
                refined_detections.append(
                    fallback_detection(
                        detection=detection,
                        fallback_bbox=bbox,
                        status=f"sam_error_fallback: {exc}",
                    )
                )
                masks.append(None)
                continue

            mask_path: str | None = None
            if save_mask and mask_output_dir is not None and refined_bbox is not None:
                mask_file = mask_output_dir / f"{image_path.stem}_mask_{index:04d}.png"
                save_mask_png(mask, mask_file)
                mask_path = mask_file.as_posix()

            detection.update(
                {
                    "refined_bbox_xyxy": refined_bbox if refined_bbox is not None else bbox,
                    "mask_area": mask_area,
                    "sam_score": sam_score,
                    "mask_path": mask_path,
                    "refine_status": "refined" if refined_bbox is not None else "empty_mask_fallback",
                }
            )
            if refined_bbox is not None:
                refined_count += 1
            refined_detections.append(detection)
            masks.append(mask if refined_bbox is not None else None)

        result = build_result(
            image_path=image_path,
            dino_json_path=dino_json_path,
            dino_data=dino_data,
            detections=refined_detections,
            model_type=self.model_type,
            checkpoint=self.checkpoint,
            device=self.device,
            save_mask=save_mask,
            min_refine_area_px=min_refine_area_px,
        )
        return SamRefineOutput(
            result=result,
            masks=masks,
            total_detections=len(refined_detections),
            refined_detections=refined_count,
        )

    def predict_mask(self, bbox_xyxy: list[float]) -> tuple[Any, float]:
        """Predict one SAM mask from one xyxy bbox."""
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError("numpy is required to run SAM prediction.") from exc

        box_array = np.array(bbox_xyxy, dtype=np.float32)
        masks, scores, _ = self._predictor.predict(
            box=box_array,
            multimask_output=False,
        )
        if len(masks) == 0:
            raise RuntimeError("SAM returned no masks.")

        score_values = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        best_index = int(np.argmax(scores)) if len(score_values) > 1 else 0
        return masks[best_index].astype(bool), float(score_values[best_index])


def run_sam_refine_single(
    image_path: Path,
    dino_json_path: Path,
    checkpoint: Path,
    model_type: str,
    device: str,
    save_mask: bool = False,
    mask_output_dir: Path | None = None,
    min_refine_area_px: float = 1024.0,
) -> SamRefineOutput:
    """Run SAM refinement for one image and one Grounding DINO JSON."""
    validate_refine_inputs(image_path, dino_json_path, checkpoint)
    if save_mask and mask_output_dir is None:
        raise ValueError("--mask-output-dir is required when --save-mask is set.")

    dino_data = load_dino_json(dino_json_path)
    source_detections = dino_data.get("detections", [])
    if not source_detections:
        return build_empty_refine_output(
            image_path=image_path,
            dino_json_path=dino_json_path,
            dino_data=dino_data,
            model_type=model_type,
            checkpoint=checkpoint,
            device=device,
            save_mask=save_mask,
            min_refine_area_px=min_refine_area_px,
        )

    refiner = SamBoxRefiner(checkpoint=checkpoint, model_type=model_type, device=device)
    return refiner.refine_loaded_json(
        image_path=image_path,
        dino_json_path=dino_json_path,
        dino_data=dino_data,
        save_mask=save_mask,
        mask_output_dir=mask_output_dir,
        min_refine_area_px=min_refine_area_px,
    )


def draw_box(draw: Any, bbox: Any, color: tuple[int, int, int], width: int) -> None:
    """Draw one bbox when it is valid."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return
    if x2 <= x1 or y2 <= y1:
        return
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)


def save_refine_visualization(
    image_path: Path,
    detections: list[dict[str, Any]],
    masks: list[Any | None],
    output_image: Path,
) -> None:
    """Save a before/after visualization for SAM bbox refinement."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required to save SAM refinement visualization.") from exc

    with Image.open(image_path) as image:
        canvas = image.convert("RGBA")

    for mask in masks:
        if mask is None:
            continue
        mask_layer = Image.fromarray(mask.astype("uint8") * 80, mode="L")
        overlay = Image.new("RGBA", canvas.size, (64, 220, 120, 0))
        overlay.putalpha(mask_layer)
        canvas = Image.alpha_composite(canvas, overlay)

    canvas_rgb = canvas.convert("RGB")
    draw = ImageDraw.Draw(canvas_rgb)
    font = ImageFont.load_default()

    for detection in detections:
        draw_box(draw, detection.get("bbox_xyxy"), color=(255, 64, 64), width=2)
        draw_box(draw, detection.get("refined_bbox_xyxy"), color=(64, 220, 120), width=2)

        refined_bbox = detection.get("refined_bbox_xyxy")
        label_box = refined_bbox if refined_bbox is not None else detection.get("bbox_xyxy")
        if isinstance(label_box, (list, tuple)) and len(label_box) == 4:
            try:
                x1 = float(label_box[0])
                y1 = float(label_box[1])
            except (TypeError, ValueError):
                continue
            phrase = str(detection.get("phrase", "object"))
            score = detection.get("score")
            score_text = f" {float(score):.2f}" if isinstance(score, (float, int)) else ""
            status = str(detection.get("refine_status", ""))
            label = f"{phrase}{score_text} | {status}"
            draw.text((x1, max(0.0, y1 - 12.0)), label, fill=(255, 255, 255), font=font)

    output_image.parent.mkdir(parents=True, exist_ok=True)
    canvas_rgb.save(output_image, quality=95)
