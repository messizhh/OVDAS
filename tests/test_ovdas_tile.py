from __future__ import annotations

import unittest
from pathlib import Path

from src.open_vocab.phrase_normalization import load_class_mapping, resolve_class_id
from src.open_vocab.tiled_inference import (
    build_tiled_result,
    class_aware_nms,
    clip_bbox_xyxy,
    generate_tiles,
    map_tile_bbox_to_image,
    normalize_detection_for_merge,
)
from tools.generate_yolo_labels_from_auto import Summary, convert_detections


class OvdasTileTest(unittest.TestCase):
    def class_maps(self) -> tuple[dict[str, int], dict[int, str]]:
        return load_class_mapping(Path("configs/classes_visdrone.yaml"))

    def test_generate_640_tiles_uses_20_percent_overlap_on_regular_steps(self) -> None:
        tiles = generate_tiles(2000, 640, tile_size=640, overlap_ratio=0.20)
        x_starts = [tile.xyxy[0] for tile in tiles]
        self.assertEqual(x_starts[0], 0)
        self.assertEqual(x_starts[1] - x_starts[0], 512)
        self.assertEqual((640 - 512) / 640, 0.20)

    def test_edge_tiles_cover_complete_image(self) -> None:
        width, height = 1500, 900
        tiles = generate_tiles(width, height, tile_size=640, overlap_ratio=0.20)
        self.assertEqual(min(tile.xyxy[0] for tile in tiles), 0)
        self.assertEqual(min(tile.xyxy[1] for tile in tiles), 0)
        self.assertEqual(max(tile.xyxy[2] for tile in tiles), width)
        self.assertEqual(max(tile.xyxy[3] for tile in tiles), height)

    def test_tile_bbox_maps_to_original_coordinates(self) -> None:
        mapped = map_tile_bbox_to_image(
            [10, 20, 30, 40],
            tile_xyxy=(512, 128, 1152, 768),
            image_width=1600,
            image_height=900,
        )
        self.assertEqual(mapped, [522.0, 148.0, 542.0, 168.0])

    def test_bbox_clipping_handles_out_of_bounds(self) -> None:
        self.assertEqual(clip_bbox_xyxy([-5, -1, 20, 30], 100, 100), [0.0, 0.0, 20.0, 30.0])
        self.assertIsNone(clip_bbox_xyxy([10, 10, 5, 20], 100, 100))

    def test_class_aware_nms_suppresses_same_class_only(self) -> None:
        detections = [
            {"bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "score": 0.9, "class_id": 3},
            {"bbox_xyxy": [1.0, 1.0, 11.0, 11.0], "score": 0.8, "class_id": 3},
            {"bbox_xyxy": [1.0, 1.0, 11.0, 11.0], "score": 0.7, "class_id": 4},
        ]
        kept = class_aware_nms(detections, iou_threshold=0.5)
        self.assertEqual(len(kept), 2)
        class_ids = sorted(item["class_id"] for item in kept)
        self.assertEqual(class_ids, [3, 4])
        kept_class_3 = next(item for item in kept if item["class_id"] == 3)
        self.assertIs(kept_class_3["merged"], True)
        self.assertEqual(kept_class_3["merged_count"], 1)

    def test_small_image_gets_one_tile_covering_whole_image(self) -> None:
        tiles = generate_tiles(320, 240, tile_size=640, overlap_ratio=0.20)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].xyxy, (0, 0, 320, 240))

    def test_empty_predictions_nms_returns_empty_list(self) -> None:
        self.assertEqual(class_aware_nms([], iou_threshold=0.5), [])

    def test_combined_phrases_resolve_by_visdrone_class_order(self) -> None:
        phrase_to_id, id_to_name = self.class_maps()
        cases = {
            "truck bus": "truck",
            "car van truck bus": "car",
            "car truck": "car",
            "van truck bus": "van",
            "car truck bus": "car",
            "van bus": "van",
        }

        for phrase, expected_class_name in cases.items():
            with self.subTest(phrase=phrase):
                class_id, used_alias = resolve_class_id(phrase, phrase_to_id)
                self.assertEqual(class_id, phrase_to_id[expected_class_name])
                assert class_id is not None
                self.assertEqual(id_to_name[class_id], expected_class_name)
                self.assertIs(used_alias, True)

    def test_exact_match_and_explicit_alias_priority_stay_unchanged(self) -> None:
        phrase_to_id, _ = self.class_maps()

        class_id, used_alias = resolve_class_id("car", phrase_to_id)
        self.assertEqual(class_id, phrase_to_id["car"])
        self.assertIs(used_alias, False)

        explicit_alias_cases = {
            "car van": "car",
            "van truck": "van",
            "pedestrian people": "people",
        }
        for phrase, expected_class_name in explicit_alias_cases.items():
            with self.subTest(phrase=phrase):
                class_id, used_alias = resolve_class_id(phrase, phrase_to_id)
                self.assertEqual(class_id, phrase_to_id[expected_class_name])
                self.assertIs(used_alias, True)

    def test_unknown_and_substring_phrases_do_not_match_classes(self) -> None:
        phrase_to_id, _ = self.class_maps()
        for phrase in ("airplane ship", "cargo vanishing busstop", "motorcycle"):
            with self.subTest(phrase=phrase):
                class_id, used_alias = resolve_class_id(phrase, phrase_to_id)
                self.assertIsNone(class_id)
                self.assertIs(used_alias, False)

    def test_normalized_detection_keeps_json_compatible_fields(self) -> None:
        phrase_to_id = {"car": 3, "van": 4}
        id_to_name = {3: "car", 4: "van"}
        detection = {
            "bbox_xyxy": [1, 2, 10, 12],
            "bbox_cxcywh_norm": [0.1, 0.2, 0.3, 0.4],
            "score": 0.88,
            "phrase": "car van",
        }
        normalized = normalize_detection_for_merge(
            detection=detection,
            image_width=100,
            image_height=100,
            phrase_to_id=phrase_to_id,
            id_to_name=id_to_name,
            source="tile",
            tile=generate_tiles(100, 100, tile_size=64, overlap_ratio=0.20)[0],
        )
        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["bbox_xyxy"], [1.0, 2.0, 10.0, 12.0])
        self.assertEqual(normalized["bbox_cxcywh_norm"], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(normalized["phrase"], "car")
        self.assertEqual(normalized["original_phrase"], "car van")
        self.assertEqual(normalized["source"], "tile")
        self.assertEqual(normalized["tile_xyxy"], [0, 0, 64, 64])
        self.assertEqual(normalized["tile_index"], 0)
        self.assertEqual(normalized["original_score"], 0.88)
        self.assertIs(normalized["merged"], False)

    def test_combined_phrase_normalized_detection_fields_are_consistent(self) -> None:
        phrase_to_id, id_to_name = self.class_maps()
        detection = {
            "bbox_xyxy": [1, 2, 10, 12],
            "score": 0.88,
            "phrase": "car truck bus",
        }
        normalized = normalize_detection_for_merge(
            detection=detection,
            image_width=100,
            image_height=100,
            phrase_to_id=phrase_to_id,
            id_to_name=id_to_name,
            source="tile",
            tile=generate_tiles(100, 100, tile_size=64, overlap_ratio=0.20)[0],
        )

        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized["original_phrase"], "car truck bus")
        self.assertEqual(normalized["normalized_phrase"], "car")
        self.assertEqual(normalized["class_id"], phrase_to_id["car"])
        self.assertEqual(normalized["class_name"], "car")
        self.assertIs(normalized["phrase_alias_used"], True)

    def test_yolo_label_generation_keeps_combined_phrase_detection(self) -> None:
        class Args:
            enable_size_aware_filter = False
            score_threshold = 0.35
            bbox_key = "bbox_xyxy"
            fallback_bbox_key = "bbox_xyxy"
            min_box_area = 4.0

        phrase_to_id, id_to_name = self.class_maps()
        summary = Summary()
        labels = convert_detections(
            detections=[
                {
                    "bbox_xyxy": [10, 10, 30, 40],
                    "score": 0.9,
                    "phrase": "truck bus",
                }
            ],
            image_width=100,
            image_height=100,
            phrase_to_id=phrase_to_id,
            id_to_name=id_to_name,
            args=Args(),
            summary=summary,
        )

        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].class_id, phrase_to_id["truck"])
        self.assertEqual(labels[0].class_name, "truck")
        self.assertEqual(summary.skipped_unknown_class, 0)
        self.assertEqual(summary.mapped_alias_labels, 1)

    def test_tiled_result_json_preserves_existing_top_level_fields(self) -> None:
        result = build_tiled_result(
            image_path=Path("image.jpg"),
            prompt="car.",
            box_threshold=0.25,
            text_threshold=0.25,
            device="cuda",
            image_width=100,
            image_height=80,
            tile_size=640,
            overlap_ratio=0.20,
            include_full_image=True,
            merge_iou=0.5,
            detections=[],
            raw_detection_count=0,
            full_detection_count=0,
            tile_detection_count=0,
            inference_time_sec=1.25,
        )
        for key in ("image_path", "prompt", "box_threshold", "text_threshold", "device", "detections"):
            self.assertIn(key, result)
        self.assertEqual(result["image_width"], 100)
        self.assertEqual(result["image_height"], 80)
        self.assertIs(result["tiled_inference"]["include_full_image"], True)


if __name__ == "__main__":
    unittest.main()
