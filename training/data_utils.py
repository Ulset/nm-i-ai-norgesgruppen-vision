"""Utilities for data preparation: product mapping, train/val split, COCO→YOLO conversion."""

import json
import random
from pathlib import Path


def build_product_category_mapping(
    annotations: dict, metadata: dict
) -> dict[str, int]:
    """Map product_code → category_id via exact name matching.

    Args:
        annotations: Parsed COCO annotations dict (needs 'categories' key).
        metadata: Parsed product_images/metadata.json dict (needs 'products' key).

    Returns:
        Dict mapping product_code (str) to category_id (int).
        Excludes empty-name categories and unmatched products.
    """
    name_to_cat_id = {}
    for cat in annotations["categories"]:
        name = cat["name"].strip()
        if name:
            name_to_cat_id[name] = cat["id"]

    mapping = {}
    for product in metadata["products"]:
        product_name = product["product_name"].strip()
        if product_name and product_name in name_to_cat_id:
            mapping[product["product_code"]] = name_to_cat_id[product_name]

    return mapping


def create_train_val_split(
    images: list[dict],
    annotations: list[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split image IDs into train and val sets.

    Args:
        images: List of COCO image dicts (need 'id' key).
        annotations: List of COCO annotation dicts (available for stratification).
        val_ratio: Fraction of images for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_image_ids, val_image_ids).
    """
    image_ids = sorted([img["id"] for img in images])
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    val_count = int(len(image_ids) * val_ratio)
    val_ids = image_ids[:val_count]
    train_ids = image_ids[val_count:]

    return train_ids, val_ids


def coco_to_yolo_label(annotation: dict, image_width: int, image_height: int) -> str:
    """Convert a single COCO annotation to YOLO format label line.

    COCO bbox: [x, y, width, height] in pixels
    YOLO bbox: [class_id, x_center, y_center, width, height] normalized to [0, 1]
    """
    cat_id = annotation["category_id"]
    x, y, w, h = annotation["bbox"]

    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    w_norm = w / image_width
    h_norm = h / image_height

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def load_coco_annotations(annotations_path: Path) -> dict:
    """Load and return parsed COCO annotations JSON."""
    with open(annotations_path) as f:
        return json.load(f)


def load_product_metadata(metadata_path: Path) -> dict:
    """Load and return parsed product metadata JSON."""
    with open(metadata_path) as f:
        return json.load(f)
