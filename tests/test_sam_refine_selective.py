from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.segmentation.sam_refine import SamBoxRefiner


class RaisingPredictor:
    def set_image(self, image: object) -> None:
        raise AssertionError("SAM set_image should not be called for skipped small boxes.")


class SelectiveSamTest(unittest.TestCase):
    def test_selective_sam_marks_small_box_skipped_without_calling_sam(self) -> None:
        try:
            import cv2  # noqa: F401
            from PIL import Image
        except ModuleNotFoundError as exc:
            self.skipTest(f"Optional image dependency missing: {exc}")

        with TemporaryDirectory() as temp_root:
            tmp_path = Path(temp_root)
            image_path = tmp_path / "sample.jpg"
            Image.new("RGB", (100, 100), color=(0, 0, 0)).save(image_path)
            dino_json_path = tmp_path / "sample_grounding_dino.json"
            dino_json_path.write_text("{}", encoding="utf-8")

            refiner = SamBoxRefiner.__new__(SamBoxRefiner)
            refiner.checkpoint = Path("dummy_sam.pth")
            refiner.model_type = "vit_h"
            refiner.device = "cpu"
            refiner._predictor = RaisingPredictor()

            output = refiner.refine_loaded_json(
                image_path=image_path,
                dino_json_path=dino_json_path,
                dino_data={
                    "image_path": image_path.as_posix(),
                    "detections": [
                        {
                            "bbox_xyxy": [10, 10, 20, 20],
                            "score": 0.9,
                            "phrase": "car",
                        }
                    ],
                },
                min_refine_area_px=1024,
            )

        detection = output.result["detections"][0]
        self.assertEqual(detection["refine_status"], "skipped_small")
        self.assertEqual(detection["refined_bbox_xyxy"], [10.0, 10.0, 20.0, 20.0])
        self.assertEqual(detection["mask_area"], 0)
        self.assertIsNone(detection["sam_score"])
        self.assertEqual(output.total_detections, 1)
        self.assertEqual(output.refined_detections, 0)


if __name__ == "__main__":
    unittest.main()
