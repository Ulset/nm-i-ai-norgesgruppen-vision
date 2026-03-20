# submission/run.py
"""Submission entry point for NorgesGruppen object detection.

Executed as: python run.py --input /data/images --output /output/predictions.json

Two-stage pipeline:
  1. YOLOv8x tiled inference → bounding boxes + baseline categories
  2. EfficientNet-B2 crop classification + reference embedding fallback

Uses ONNX models with CUDAExecutionProvider for GPU acceleration.
Falls back to YOLO-only mode if classifier fails to load.

IMPORTANT: No blocked imports (os, sys, subprocess, pickle, yaml, etc.)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image

import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion

from utils import (
    compute_tiles,
    map_tile_boxes_to_image,
    classify_detections,
    compute_final_score,
    xyxy_to_xywh,
    image_id_from_filename,
)

# Inference config
YOLO_INPUT_SIZE = 1280
TILE_SIZE = 1280
TILE_OVERLAP = 0.2
WBF_IOU_THR = 0.55
CLASSIFIER_INPUT_SIZE = 224
CLASSIFIER_THRESHOLD = 0.7
REFERENCE_THRESHOLD = 0.8
CROP_BATCH_SIZE = 32
CONFIDENCE_FLOOR = 0.05
UNKNOWN_PRODUCT_ID = 355


def load_onnx_session(model_path: str) -> ort.InferenceSession:
    """Load ONNX model with GPU preference."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


def run_yolo_on_tile(
    session: ort.InferenceSession,
    tile_img: Image.Image,
    input_size: int = 1280,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run YOLO ONNX model on a single tile.

    Returns:
        boxes: (N, 4) in xyxy format (tile coordinates)
        scores: (N,) confidence scores
        class_ids: (N,) predicted class IDs
    """
    orig_w, orig_h = tile_img.size
    img_resized = tile_img.resize((input_size, input_size), Image.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: arr})

    # YOLO ONNX output: (1, 4+nc, num_detections)
    output = outputs[0]
    if output.ndim == 3:
        output = output[0].T  # (N, 4+nc)

    if len(output) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    boxes_cxcywh = output[:, :4]
    class_scores = output[:, 4:]

    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    scale_x = orig_w / input_size
    scale_y = orig_h / input_size
    boxes_xyxy[:, [0, 2]] *= scale_x
    boxes_xyxy[:, [1, 3]] *= scale_y

    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    mask = scores > CONFIDENCE_FLOOR
    return boxes_xyxy[mask], scores[mask], class_ids[mask]


def run_classifier_batch(
    session: ort.InferenceSession,
    crops: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Run classifier on a batch of crops.

    Args:
        crops: List of (224, 224, 3) uint8 arrays.

    Returns:
        probs: (N, num_classes) softmax probabilities
        embeddings: (N, embed_dim) penultimate layer embeddings
    """
    if not crops:
        return np.zeros((0, 357)), np.zeros((0, 1408))

    batch = np.stack(crops).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch = (batch - mean) / std
    batch = np.transpose(batch, (0, 3, 1, 2))  # NCHW

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})

    probs = outputs[0]
    embeddings = outputs[1] if len(outputs) > 1 else np.zeros((len(crops), 1408))

    # Softmax if not already applied
    if probs.min() < 0 or probs.sum(axis=1).mean() > 1.5:
        exp_probs = np.exp(probs - probs.max(axis=1, keepdims=True))
        probs = exp_probs / exp_probs.sum(axis=1, keepdims=True)

    return probs, embeddings


def process_image(
    img_path: Path,
    yolo_session: ort.InferenceSession,
    clf_session: ort.InferenceSession | None,
    reference_embeddings: np.ndarray | None,
    class_idx_to_cat_id: dict[int, int] | None = None,
) -> list[dict]:
    """Run full two-stage pipeline on a single image."""
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    image_id = image_id_from_filename(img_path.name)

    # Stage 1: Tiled YOLO inference
    tiles = compute_tiles(img_w, img_h, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)

    all_boxes = []
    all_scores = []
    all_labels = []

    for tx, ty, tw, th in tiles:
        tile_img = img.crop((tx, ty, tx + tw, ty + th))
        boxes, scores, class_ids = run_yolo_on_tile(yolo_session, tile_img)

        if len(boxes) == 0:
            continue

        boxes = map_tile_boxes_to_image(boxes, (tx, ty))

        boxes_norm = boxes.copy()
        boxes_norm[:, [0, 2]] /= img_w
        boxes_norm[:, [1, 3]] /= img_h
        boxes_norm = np.clip(boxes_norm, 0, 1)

        all_boxes.append(boxes_norm)
        all_scores.append(scores)
        all_labels.append(class_ids)

    # Also run on full image resized
    full_boxes, full_scores, full_labels = run_yolo_on_tile(yolo_session, img)
    if len(full_boxes) > 0:
        full_boxes_norm = full_boxes.copy()
        full_boxes_norm[:, [0, 2]] /= img_w
        full_boxes_norm[:, [1, 3]] /= img_h
        full_boxes_norm = np.clip(full_boxes_norm, 0, 1)
        all_boxes.append(full_boxes_norm)
        all_scores.append(full_scores)
        all_labels.append(full_labels)

    if not all_boxes:
        return []

    # WBF merge
    boxes_list = [b.tolist() for b in all_boxes]
    scores_list = [s.tolist() for s in all_scores]
    labels_list = [l.tolist() for l in all_labels]

    merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=WBF_IOU_THR,
        skip_box_thr=CONFIDENCE_FLOOR,
    )

    # Convert back to pixel coordinates
    merged_boxes[:, [0, 2]] *= img_w
    merged_boxes[:, [1, 3]] *= img_h

    # Stage 2: Classification (if available)
    predictions = []

    if clf_session is not None:
        crops = []
        crop_indices = []
        for i in range(len(merged_boxes)):
            x1, y1, x2, y2 = merged_boxes[i]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crop = img.crop((x1, y1, x2, y2)).resize(
                (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE), Image.LANCZOS
            )
            crops.append(np.array(crop))
            crop_indices.append(i)

        # Batch inference
        all_probs = []
        all_embeds = []
        for batch_start in range(0, len(crops), CROP_BATCH_SIZE):
            batch = crops[batch_start:batch_start + CROP_BATCH_SIZE]
            probs, embeds = run_classifier_batch(clf_session, batch)
            all_probs.append(probs)
            all_embeds.append(embeds)

        if all_probs:
            all_probs = np.concatenate(all_probs, axis=0)
            all_embeds = np.concatenate(all_embeds, axis=0)

            for j, i in enumerate(crop_indices):
                yolo_cat = int(merged_labels[i])
                yolo_conf = float(merged_scores[i])
                yolo_class_conf = yolo_conf

                # CRITICAL: Remap classifier probs from ImageFolder index to category_id.
                # ImageFolder sorts folder names lexicographically, so
                # index 0 != category_id 0. We need this mapping.
                clf_probs = all_probs[j]
                if class_idx_to_cat_id is not None:
                    remapped = np.zeros(357, dtype=np.float32)
                    for idx, cat_id in class_idx_to_cat_id.items():
                        if idx < len(clf_probs):
                            remapped[cat_id] = clf_probs[idx]
                    clf_probs = remapped

                final_cat = classify_detections(
                    yolo_category=yolo_cat,
                    yolo_class_conf=yolo_class_conf,
                    classifier_probs=clf_probs,
                    reference_embeddings=reference_embeddings,
                    crop_embedding=all_embeds[j],
                    classifier_threshold=CLASSIFIER_THRESHOLD,
                    reference_threshold=REFERENCE_THRESHOLD,
                )

                clf_conf = float(clf_probs.max())
                final_score = compute_final_score(yolo_conf, max(clf_conf, yolo_class_conf))

                box_xywh = xyxy_to_xywh(merged_boxes[i:i+1])[0]
                predictions.append({
                    "image_id": image_id,
                    "category_id": final_cat,
                    "bbox": [round(float(v), 1) for v in box_xywh],
                    "score": round(final_score, 3),
                })
    else:
        # YOLO-only fallback
        for i in range(len(merged_boxes)):
            box_xywh = xyxy_to_xywh(merged_boxes[i:i+1])[0]
            predictions.append({
                "image_id": image_id,
                "category_id": int(merged_labels[i]),
                "bbox": [round(float(v), 1) for v in box_xywh],
                "score": round(float(merged_scores[i]), 3),
            })

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    script_dir = Path(__file__).resolve().parent
    yolo_path = script_dir / "yolo_detector.onnx"
    clf_path = script_dir / "classifier.onnx"
    ref_path = script_dir / "reference_embeddings.npy"

    # Load YOLO (required)
    yolo_session = load_onnx_session(str(yolo_path))

    # Load classifier (optional — graceful degradation)
    clf_session = None
    reference_embeddings = None
    class_idx_to_cat_id = None
    try:
        if clf_path.exists():
            clf_session = load_onnx_session(str(clf_path))
        if ref_path.exists():
            reference_embeddings = np.load(str(ref_path))
        mapping_path = script_dir / "class_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                raw = json.load(f)
            class_idx_to_cat_id = {int(k): int(v) for k, v in raw.items()}
    except Exception:
        clf_session = None
        reference_embeddings = None
        class_idx_to_cat_id = None

    # Process all images
    all_predictions = []
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_files:
        preds = process_image(img_path, yolo_session, clf_session, reference_embeddings, class_idx_to_cat_id)
        all_predictions.extend(preds)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_predictions, f)


if __name__ == "__main__":
    main()
