# submission/run.py
"""Submission entry point for NorgesGruppen object detection.

Executed as: python run.py --input /data/images --output /output/predictions.json

Two-stage pipeline with ensemble + TTA:
  1. Multi-model YOLO ensemble with tiled inference + TTA
  2. EfficientNet-B2 / DINOv2 crop classification + reference embedding fallback

Uses ONNX models with CUDAExecutionProvider for GPU acceleration.
Auto-discovers available models. Falls back gracefully if any component is missing.

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
    classify_detections_dinov2,
    compute_final_score,
    xyxy_to_xywh,
    image_id_from_filename,
)
from baked_data import load_reference_embeddings, load_class_mapping

# Inference config defaults
TILE_SIZE = 1280
TILE_OVERLAP = 0.2
WBF_IOU_THR = 0.55
CLASSIFIER_INPUT_SIZE = 224
CLASSIFIER_THRESHOLD = 0.7
REFERENCE_THRESHOLD = 0.8
CROP_BATCH_SIZE = 32
CONFIDENCE_FLOOR = 0.05
UNKNOWN_PRODUCT_ID = 355
ENABLE_TTA = True  # Horizontal flip TTA

# Override defaults from config.json if present
_script_dir = Path(__file__).resolve().parent
_config_path = _script_dir / "config.json"
if _config_path.exists():
    _cfg = json.loads(_config_path.read_text())
    CONFIDENCE_FLOOR = _cfg.get("confidence_floor", CONFIDENCE_FLOOR)
    CLASSIFIER_THRESHOLD = _cfg.get("classifier_threshold", CLASSIFIER_THRESHOLD)
    REFERENCE_THRESHOLD = _cfg.get("reference_threshold", REFERENCE_THRESHOLD)
    WBF_IOU_THR = _cfg.get("wbf_iou_thr", WBF_IOU_THR)
    TILE_OVERLAP = _cfg.get("tile_overlap", TILE_OVERLAP)
    ENABLE_TTA = _cfg.get("enable_tta", ENABLE_TTA)


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

    # YOLO ONNX output: (1, 4+nc, N) or (1, N, 4+nc) depending on version
    output = outputs[0]
    if output.ndim == 3:
        # Auto-detect format: smaller dim is 4+nc, larger is num_detections
        if output.shape[1] < output.shape[2]:
            output = output[0].T  # (1, 4+nc, N) -> (N, 4+nc)
        else:
            output = output[0]    # (1, N, 4+nc) -> (N, 4+nc)

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


def run_yolo_on_image(
    session: ort.InferenceSession,
    img: Image.Image,
    input_size: int = 1280,
    tile_size: int = 1280,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Run YOLO with tiled inference on a single image.

    Returns lists of boxes/scores/labels (normalized to [0,1]) for WBF.
    """
    img_w, img_h = img.size

    all_boxes = []
    all_scores = []
    all_labels = []

    # Tiled inference
    tiles = compute_tiles(img_w, img_h, tile_size=tile_size, overlap=TILE_OVERLAP)
    for tx, ty, tw, th in tiles:
        tile_img = img.crop((tx, ty, tx + tw, ty + th))
        boxes, scores, class_ids = run_yolo_on_tile(session, tile_img, input_size)
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

    # Full image pass
    full_boxes, full_scores, full_labels = run_yolo_on_tile(session, img, input_size)
    if len(full_boxes) > 0:
        full_boxes_norm = full_boxes.copy()
        full_boxes_norm[:, [0, 2]] /= img_w
        full_boxes_norm[:, [1, 3]] /= img_h
        full_boxes_norm = np.clip(full_boxes_norm, 0, 1)
        all_boxes.append(full_boxes_norm)
        all_scores.append(full_scores)
        all_labels.append(full_labels)

    return all_boxes, all_scores, all_labels


def run_yolo_tta(
    session: ort.InferenceSession,
    img: Image.Image,
    input_size: int = 1280,
    tile_size: int = 1280,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Run YOLO with TTA (horizontal flip) on a single image.

    Returns combined lists for WBF merging.
    """
    img_w, img_h = img.size

    # Original
    all_boxes, all_scores, all_labels = run_yolo_on_image(
        session, img, input_size, tile_size
    )

    if ENABLE_TTA:
        # Horizontal flip
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_boxes, flip_scores, flip_labels = run_yolo_on_image(
            session, img_flip, input_size, tile_size
        )
        # Mirror boxes back: x1_new = 1 - x2_old, x2_new = 1 - x1_old
        for i in range(len(flip_boxes)):
            b = flip_boxes[i].copy()
            b[:, 0], b[:, 2] = 1 - flip_boxes[i][:, 2], 1 - flip_boxes[i][:, 0]
            flip_boxes[i] = b

        all_boxes.extend(flip_boxes)
        all_scores.extend(flip_scores)
        all_labels.extend(flip_labels)

    return all_boxes, all_scores, all_labels


def run_dinov2_batch(
    session: ort.InferenceSession,
    crops: list[np.ndarray],
) -> np.ndarray:
    """Run DINOv2 on a batch of crops, return L2-normalized embeddings."""
    if not crops:
        return np.zeros((0, 384))

    batch = np.stack(crops).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch = (batch - mean) / std
    batch = np.transpose(batch, (0, 3, 1, 2))  # NCHW

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})
    embeddings = outputs[0]  # (batch, 384)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return embeddings / norms


def run_classifier_batch(
    session: ort.InferenceSession,
    crops: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Run classifier on a batch of crops."""
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
    yolo_sessions: list[tuple[ort.InferenceSession, int, int]],
    clf_session: ort.InferenceSession | None,
    reference_embeddings: np.ndarray | None,
    class_idx_to_cat_id: dict[int, int] | None = None,
    dinov2_session: ort.InferenceSession | None = None,
    use_dinov2: bool = False,
) -> list[dict]:
    """Run full ensemble + two-stage pipeline on a single image.

    Args:
        yolo_sessions: List of (session, input_size, tile_size) tuples for ensemble.
        dinov2_session: Optional DINOv2 ONNX session for embedding-based classification.
        use_dinov2: If True, use DINOv2 for crop embeddings instead of EfficientNet.
    """
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    image_id = image_id_from_filename(img_path.name)

    # Stage 1: Ensemble YOLO inference with TTA
    all_boxes = []
    all_scores = []
    all_labels = []

    for session, input_size, tile_size in yolo_sessions:
        boxes, scores, labels = run_yolo_tta(
            session, img, input_size, tile_size
        )
        all_boxes.extend(boxes)
        all_scores.extend(scores)
        all_labels.extend(labels)

    if not all_boxes:
        return []

    # WBF merge all detections from all models + tiles + TTA
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

    has_classifier = clf_session is not None or (use_dinov2 and dinov2_session is not None)
    if has_classifier:
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

        # DINOv2 embedding path
        if use_dinov2 and dinov2_session is not None:
            all_dinov2_embeds = []
            for batch_start in range(0, len(crops), CROP_BATCH_SIZE):
                batch = crops[batch_start:batch_start + CROP_BATCH_SIZE]
                embeds = run_dinov2_batch(dinov2_session, batch)
                all_dinov2_embeds.append(embeds)

            if all_dinov2_embeds:
                all_dinov2_embeds = np.concatenate(all_dinov2_embeds, axis=0)
                for j, i in enumerate(crop_indices):
                    yolo_conf = float(merged_scores[i])
                    final_cat, ref_sim = classify_detections_dinov2(
                        crop_embedding=all_dinov2_embeds[j],
                        reference_embeddings=reference_embeddings,
                        reference_threshold=REFERENCE_THRESHOLD,
                    )
                    final_score = compute_final_score(yolo_conf, max(ref_sim, yolo_conf))
                    box_xywh = xyxy_to_xywh(merged_boxes[i:i+1])[0]
                    predictions.append({
                        "image_id": image_id,
                        "category_id": final_cat,
                        "bbox": [round(float(v), 1) for v in box_xywh],
                        "score": round(final_score, 3),
                    })

        # EfficientNet classifier path
        elif clf_session is not None:
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

                    # Remap classifier probs from ImageFolder index to category_id
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

    # Load YOLO models (ensemble — auto-discover all available)
    yolo_sessions = []

    for onnx_file in sorted(script_dir.glob("yolo_*.onnx")):
        sess = load_onnx_session(str(onnx_file))
        # Detect input size from model metadata
        input_shape = sess.get_inputs()[0].shape  # e.g., [1, 3, 1280, 1280]
        imgsz = input_shape[2] if isinstance(input_shape[2], int) else 1280
        tile_size = imgsz  # tile same as input for consistency
        yolo_sessions.append((sess, imgsz, tile_size))
        print(f"Loaded YOLO: {onnx_file.name} @ {imgsz}px")

    # Load classifier (optional — graceful degradation)
    clf_session = None
    dinov2_session = None
    reference_embeddings = None
    class_idx_to_cat_id = None
    use_dinov2 = False
    try:
        # Check for DINOv2 model first (preferred for embedding-based classification)
        dinov2_path = script_dir / "dinov2_fp16.onnx"
        if dinov2_path.exists():
            dinov2_session = load_onnx_session(str(dinov2_path))
            use_dinov2 = True
            print("Loaded DINOv2 for embedding-based classification")

        # Also load EfficientNet classifier if available
        clf_path = script_dir / "classifier.onnx"
        if clf_path.exists():
            clf_session = load_onnx_session(str(clf_path))
            print("Loaded EfficientNet classifier")

        # Load baked-in reference data (embedded in baked_data.py)
        reference_embeddings = load_reference_embeddings()
        class_idx_to_cat_id = load_class_mapping()
    except Exception:
        clf_session = None
        dinov2_session = None
        reference_embeddings = None
        class_idx_to_cat_id = None

    # Process all images
    all_predictions = []
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_files:
        preds = process_image(
            img_path, yolo_sessions, clf_session,
            reference_embeddings, class_idx_to_cat_id,
            dinov2_session=dinov2_session,
            use_dinov2=use_dinov2,
        )
        all_predictions.extend(preds)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_predictions, f)


if __name__ == "__main__":
    main()
