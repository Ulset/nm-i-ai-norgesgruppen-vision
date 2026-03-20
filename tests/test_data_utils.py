import json
import pytest
from pathlib import Path
from training.data_utils import build_product_category_mapping, create_train_val_split, coco_to_yolo_label


class TestProductCategoryMapping:
    def test_exact_name_match(self):
        annotations = {
            "categories": [
                {"id": 0, "name": "COFFEE MATE 180G NESTLE", "supercategory": "product"},
                {"id": 1, "name": "WASA KNEKKEBRØD 300G", "supercategory": "product"},
            ],
            "images": [],
            "annotations": [],
        }
        metadata = {
            "products": [
                {"product_code": "123", "product_name": "COFFEE MATE 180G NESTLE"},
                {"product_code": "456", "product_name": "WASA KNEKKEBRØD 300G"},
            ]
        }
        mapping = build_product_category_mapping(annotations, metadata)
        assert mapping["123"] == 0
        assert mapping["456"] == 1

    def test_empty_name_category_excluded(self):
        annotations = {
            "categories": [
                {"id": 0, "name": "PRODUCT A", "supercategory": "product"},
                {"id": 300, "name": "", "supercategory": "product"},
            ],
            "images": [],
            "annotations": [],
        }
        metadata = {
            "products": [
                {"product_code": "789", "product_name": ""},
            ]
        }
        mapping = build_product_category_mapping(annotations, metadata)
        assert "789" not in mapping

    def test_unmatched_product_ignored(self):
        annotations = {
            "categories": [
                {"id": 0, "name": "PRODUCT A", "supercategory": "product"},
            ],
            "images": [],
            "annotations": [],
        }
        metadata = {
            "products": [
                {"product_code": "999", "product_name": "NONEXISTENT PRODUCT"},
            ]
        }
        mapping = build_product_category_mapping(annotations, metadata)
        assert "999" not in mapping


class TestTrainValSplit:
    def test_split_ratio(self):
        images = [{"id": i, "file_name": f"img_{i:05d}.jpg"} for i in range(100)]
        annotations = [
            {"id": i, "image_id": i % 100, "category_id": i % 10, "bbox": [0, 0, 10, 10]}
            for i in range(500)
        ]
        train_ids, val_ids = create_train_val_split(images, annotations, val_ratio=0.2, seed=42)
        assert len(train_ids) == 80
        assert len(val_ids) == 20
        assert len(set(train_ids) & set(val_ids)) == 0

    def test_split_is_deterministic(self):
        images = [{"id": i, "file_name": f"img_{i:05d}.jpg"} for i in range(50)]
        annotations = [{"id": i, "image_id": i % 50, "category_id": 0, "bbox": [0, 0, 10, 10]} for i in range(100)]
        split1 = create_train_val_split(images, annotations, val_ratio=0.2, seed=42)
        split2 = create_train_val_split(images, annotations, val_ratio=0.2, seed=42)
        assert split1 == split2

    def test_all_images_assigned(self):
        images = [{"id": i, "file_name": f"img_{i:05d}.jpg"} for i in range(50)]
        annotations = [{"id": i, "image_id": i % 50, "category_id": 0, "bbox": [0, 0, 10, 10]} for i in range(100)]
        train_ids, val_ids = create_train_val_split(images, annotations, val_ratio=0.2, seed=42)
        assert set(train_ids) | set(val_ids) == {i for i in range(50)}


class TestCocoToYoloLabel:
    def test_basic_conversion(self):
        annotation = {"category_id": 5, "bbox": [100, 200, 50, 80]}
        result = coco_to_yolo_label(annotation, image_width=1000, image_height=800)
        assert result == "5 0.125000 0.300000 0.050000 0.100000"

    def test_clamps_to_unit_range(self):
        annotation = {"category_id": 0, "bbox": [950, 750, 100, 100]}
        result = coco_to_yolo_label(annotation, image_width=1000, image_height=800)
        parts = result.split()
        assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])
