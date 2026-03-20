import numpy as np
import pytest
from submission.utils import (
    compute_tiles,
    map_tile_boxes_to_image,
    classify_detections,
    compute_final_score,
)


class TestComputeTiles:
    def test_small_image_no_tiling(self):
        tiles = compute_tiles(800, 600, tile_size=1280, overlap=0.2)
        assert len(tiles) == 1
        assert tiles[0] == (0, 0, 800, 600)

    def test_large_image_produces_tiles(self):
        tiles = compute_tiles(3000, 2000, tile_size=1280, overlap=0.2)
        assert len(tiles) > 1
        for x, y, w, h in tiles:
            assert w <= 1280
            assert h <= 1280

    def test_tiles_cover_full_image(self):
        img_w, img_h = 3000, 2000
        tiles = compute_tiles(img_w, img_h, tile_size=1280, overlap=0.2)
        covered = np.zeros((img_h, img_w), dtype=bool)
        for x, y, w, h in tiles:
            covered[y:y+h, x:x+w] = True
        assert covered.all()


class TestMapTileBoxes:
    def test_offset_applied(self):
        tile_boxes = np.array([[10.0, 20.0, 50.0, 60.0]])
        tile_offset = (100, 200)
        result = map_tile_boxes_to_image(tile_boxes, tile_offset)
        expected = np.array([[110.0, 220.0, 150.0, 260.0]])
        np.testing.assert_array_equal(result, expected)

    def test_empty_boxes(self):
        tile_boxes = np.zeros((0, 4))
        result = map_tile_boxes_to_image(tile_boxes, (100, 200))
        assert len(result) == 0


class TestClassifyDetections:
    def test_high_confidence_classifier_wins(self):
        probs = np.zeros(357)
        probs[9] = 0.8
        result = classify_detections(
            yolo_category=5,
            yolo_class_conf=0.3,
            classifier_probs=probs,
            reference_embeddings=np.zeros((357, 1408)),
            crop_embedding=np.zeros(1408),
            classifier_threshold=0.7,
            reference_threshold=0.8,
        )
        assert result == 9

    def test_yolo_fallback(self):
        result = classify_detections(
            yolo_category=42,
            yolo_class_conf=0.5,
            classifier_probs=np.ones(357) / 357,
            reference_embeddings=np.random.RandomState(42).randn(357, 1408).astype(np.float32),
            crop_embedding=np.random.RandomState(99).randn(1408).astype(np.float32),
            classifier_threshold=0.7,
            reference_threshold=0.8,
        )
        assert result == 42

    def test_reference_embedding_match(self):
        ref_embeddings = np.zeros((357, 1408), dtype=np.float32)
        ref_embeddings[77] = np.ones(1408) / np.sqrt(1408)  # normalized
        crop_emb = np.ones(1408, dtype=np.float32) / np.sqrt(1408)  # same direction
        result = classify_detections(
            yolo_category=5,
            yolo_class_conf=0.3,
            classifier_probs=np.ones(357) / 357,
            reference_embeddings=ref_embeddings,
            crop_embedding=crop_emb,
            classifier_threshold=0.7,
            reference_threshold=0.8,
        )
        assert result == 77


class TestComputeFinalScore:
    def test_score_multiplication(self):
        score = compute_final_score(yolo_conf=0.9, classification_conf=0.8)
        assert abs(score - 0.72) < 1e-6

    def test_score_clamped(self):
        score = compute_final_score(yolo_conf=1.5, classification_conf=0.8)
        assert 0.0 <= score <= 1.0
