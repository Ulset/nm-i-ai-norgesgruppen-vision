"""Inference utilities for the submission sandbox.

IMPORTANT: No blocked imports (os, sys, subprocess, pickle, yaml, threading, etc.)
Use pathlib for file operations, json for config files.
"""

import json
import math
import numpy as np
from pathlib import Path
from PIL import Image


def compute_tiles(
    img_width: int,
    img_height: int,
    tile_size: int = 1280,
    overlap: float = 0.2,
) -> list[tuple[int, int, int, int]]:
    """Compute tile coordinates for an image.

    Returns:
        List of (x, y, width, height) tuples for each tile.
    """
    if img_width <= tile_size and img_height <= tile_size:
        return [(0, 0, img_width, img_height)]

    stride = int(tile_size * (1 - overlap))
    tiles = []

    y = 0
    while y < img_height:
        x = 0
        h = min(tile_size, img_height - y)
        while x < img_width:
            w = min(tile_size, img_width - x)
            tiles.append((x, y, w, h))
            if x + w >= img_width:
                break
            x += stride
        if y + h >= img_height:
            break
        y += stride

    return tiles


def map_tile_boxes_to_image(
    boxes: np.ndarray,
    tile_offset: tuple[int, int],
) -> np.ndarray:
    """Map bounding boxes from tile coordinates to full image coordinates.

    Args:
        boxes: Array of shape (N, 4) in xyxy format.
        tile_offset: (x_offset, y_offset) of tile in image.

    Returns:
        Boxes offset to image coordinates.
    """
    if len(boxes) == 0:
        return boxes
    offset = np.array([tile_offset[0], tile_offset[1], tile_offset[0], tile_offset[1]])
    return boxes + offset


def classify_detections(
    yolo_category: int,
    yolo_class_conf: float,
    classifier_probs: np.ndarray,
    reference_embeddings: np.ndarray,
    crop_embedding: np.ndarray,
    classifier_threshold: float = 0.7,
    reference_threshold: float = 0.8,
) -> int:
    """Decide final category_id using three signals.

    Priority:
    1. Classifier softmax top-1 if confidence > classifier_threshold
    2. Reference embedding nearest neighbor if cosine sim > reference_threshold
    3. YOLO category as fallback
    4. unknown_product (355) if all signals very low confidence
    """
    clf_top1 = int(np.argmax(classifier_probs))
    clf_conf = float(classifier_probs[clf_top1])

    if clf_conf > classifier_threshold:
        return clf_top1

    if reference_embeddings is not None and crop_embedding is not None:
        norms_ref = np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
        norms_ref = np.maximum(norms_ref, 1e-8)
        ref_normed = reference_embeddings / norms_ref

        norm_crop = np.linalg.norm(crop_embedding)
        if norm_crop > 1e-8:
            crop_normed = crop_embedding / norm_crop
            similarities = ref_normed @ crop_normed
            best_ref_idx = int(np.argmax(similarities))
            best_ref_sim = float(similarities[best_ref_idx])

            if best_ref_sim > reference_threshold:
                return best_ref_idx

    if yolo_class_conf > 0.15 or clf_conf > 0.15:
        return yolo_category

    return 355  # Configurable via UNKNOWN_PRODUCT_ID in run.py


def classify_detections_dinov2(
    crop_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_threshold: float = 0.6,
    unknown_product_id: int = 355,
) -> tuple[int, float]:
    """Classify using DINOv2 cosine similarity only (no softmax logits).

    Returns:
        (category_id, similarity_score)
    """
    if reference_embeddings is None or crop_embedding is None:
        return unknown_product_id, 0.0

    # Embeddings should already be L2-normalized, but ensure it
    norm_crop = np.linalg.norm(crop_embedding)
    if norm_crop < 1e-8:
        return unknown_product_id, 0.0
    crop_normed = crop_embedding / norm_crop

    norms_ref = np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
    norms_ref = np.maximum(norms_ref, 1e-8)
    ref_normed = reference_embeddings / norms_ref

    similarities = ref_normed @ crop_normed
    best_idx = int(np.argmax(similarities))
    best_sim = float(similarities[best_idx])

    if best_sim >= reference_threshold:
        return best_idx, best_sim
    return unknown_product_id, best_sim


def compute_final_score(yolo_conf: float, classification_conf: float) -> float:
    """Compute the final confidence score for a detection.
    score = yolo_conf * classification_conf, clamped to [0, 1].
    """
    return max(0.0, min(1.0, yolo_conf * classification_conf))


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from xyxy to xywh (COCO format for output)."""
    if len(boxes) == 0:
        return boxes
    xywh = boxes.copy()
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh


def load_image(path: Path) -> Image.Image:
    """Load image from path, convert to RGB."""
    return Image.open(path).convert("RGB")


def image_id_from_filename(filename: str) -> int:
    """Extract numeric image_id from filename like img_00042.jpg -> 42."""
    stem = Path(filename).stem
    return int(stem.split("_")[-1])
