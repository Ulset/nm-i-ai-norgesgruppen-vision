#!/usr/bin/env python3
"""Generate pseudo-labels from a trained YOLO model and merge with existing ground truth.

Runs inference on training images, filters by confidence, and adds high-confidence
detections that don't overlap with existing labels. This expands the training set
with missed detections without duplicating existing annotations.

Usage:
  python scripts/pseudo_label.py --model runs/detect/yolo11x_1280/weights/best.pt
  python scripts/pseudo_label.py --model best.pt --confidence 0.85 --imgsz 1280
"""

import argparse
import json
from pathlib import Path

# PyTorch 2.6+ defaults weights_only=True which breaks ultralytics 8.1.0 model loading.
import torch

_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(
    *args, **{**kwargs, "weights_only": False}
)

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUM_CLASSES = 357


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into a list of (class_id, cx, cy, w, h) tuples."""
    boxes = []
    if not label_path.exists():
        return boxes
    text = label_path.read_text().strip()
    if not text:
        return boxes
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((cls_id, cx, cy, w, h))
    return boxes


def compute_iou_yolo(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two YOLO-format boxes (cx, cy, w, h) in normalized coordinates."""
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2

    # Convert to x1, y1, x2, y2
    x1_min = cx1 - w1 / 2
    y1_min = cy1 - h1 / 2
    x1_max = cx1 + w1 / 2
    y1_max = cy1 + h1 / 2

    x2_min = cx2 - w2 / 2
    y2_min = cy2 - h2 / 2
    x2_max = cx2 + w2 / 2
    y2_max = cy2 + h2 / 2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def has_significant_overlap(
    pseudo_box: tuple[float, float, float, float],
    existing_boxes: list[tuple[int, float, float, float, float]],
    iou_threshold: float,
) -> bool:
    """Check if a pseudo-label box overlaps significantly with any existing box."""
    for _, ecx, ecy, ew, eh in existing_boxes:
        iou = compute_iou_yolo(pseudo_box, (ecx, ecy, ew, eh))
        if iou >= iou_threshold:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels and merge with existing ground truth"
    )
    parser.add_argument(
        "--model", required=True, help="Path to trained YOLO .pt file"
    )
    parser.add_argument(
        "--images",
        default=str(PROJECT_ROOT / "data" / "coco_dataset" / "train" / "images"),
        help="Directory containing training images",
    )
    parser.add_argument(
        "--existing-labels",
        default=str(PROJECT_ROOT / "data" / "yolo_dataset" / "labels" / "train"),
        help="Directory containing existing YOLO labels",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "yolo_dataset_pseudo"),
        help="Output directory for pseudo-labeled dataset",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.8,
        help="Minimum confidence threshold for pseudo-labels",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for filtering overlap with existing labels",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="Inference image size"
    )
    args = parser.parse_args()

    images_dir = Path(args.images)
    existing_labels_dir = Path(args.existing_labels)
    output_dir = Path(args.output_dir)
    output_labels_dir = output_dir / "labels" / "train"
    output_images_dir = output_dir / "images" / "train"

    # Also copy val split if it exists
    existing_val_labels_dir = existing_labels_dir.parent / "val"
    existing_val_images_dir = Path(args.existing_labels).parent.parent / "images" / "val"

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Load the model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]
    )
    print(f"Found {len(image_files)} images in {images_dir}")

    # Run inference and merge
    total_existing = 0
    total_pseudo_added = 0
    total_pseudo_rejected = 0
    images_with_new_labels = 0

    for i, img_path in enumerate(image_files):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Processing image {i + 1}/{len(image_files)}: {img_path.name}")

        # Run inference
        results = model.predict(
            str(img_path),
            imgsz=args.imgsz,
            conf=args.confidence,
            verbose=False,
        )

        # Load existing labels
        label_stem = img_path.stem
        existing_label_path = existing_labels_dir / f"{label_stem}.txt"
        existing_boxes = parse_yolo_label(existing_label_path)
        total_existing += len(existing_boxes)

        # Extract pseudo-labels from predictions (already filtered by confidence)
        pseudo_added = 0
        new_boxes = []

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                img_h, img_w = result.orig_shape

                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    # Get xyxy coordinates and convert to YOLO normalized format
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = ((x1 + x2) / 2) / img_w
                    cy = ((y1 + y2) / 2) / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h

                    # Clamp to [0, 1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(0.0, min(1.0, w))
                    h = max(0.0, min(1.0, h))

                    pseudo_box = (cx, cy, w, h)

                    # Check overlap with existing labels
                    if has_significant_overlap(
                        pseudo_box, existing_boxes, args.iou_threshold
                    ):
                        total_pseudo_rejected += 1
                        continue

                    new_boxes.append((cls_id, cx, cy, w, h))
                    pseudo_added += 1

        # Merge existing + new pseudo-labels
        all_boxes = existing_boxes + new_boxes
        total_pseudo_added += pseudo_added
        if pseudo_added > 0:
            images_with_new_labels += 1

        # Write merged label file
        output_label_path = output_labels_dir / f"{label_stem}.txt"
        if all_boxes:
            lines = [
                f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                for cls_id, cx, cy, w, h in all_boxes
            ]
            output_label_path.write_text("\n".join(lines) + "\n")
        else:
            output_label_path.write_text("")

        # Symlink image
        output_image_path = output_images_dir / img_path.name
        if not output_image_path.exists():
            output_image_path.symlink_to(img_path.resolve())

    # Handle val split: symlink val images and copy val labels
    output_val_labels_dir = output_dir / "labels" / "val"
    output_val_images_dir = output_dir / "images" / "val"

    if existing_val_labels_dir.exists():
        output_val_labels_dir.mkdir(parents=True, exist_ok=True)
        output_val_images_dir.mkdir(parents=True, exist_ok=True)

        # Symlink val labels
        for label_file in existing_val_labels_dir.iterdir():
            if label_file.suffix == ".txt":
                dst = output_val_labels_dir / label_file.name
                if not dst.exists():
                    dst.symlink_to(label_file.resolve())

        # Symlink val images
        if existing_val_images_dir.exists():
            for img_file in existing_val_images_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    dst = output_val_images_dir / img_file.name
                    if not dst.exists():
                        dst.symlink_to(img_file.resolve())

    # Generate dataset_pseudo.yaml
    # Load category names from the original COCO annotations if available
    coco_ann_path = PROJECT_ROOT / "data" / "coco_dataset" / "train" / "annotations.json"
    if coco_ann_path.exists():
        with open(coco_ann_path) as f:
            coco_data = json.load(f)
        cat_names = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        names_dict = {i: cat_names.get(i, f"class_{i}") for i in range(NUM_CLASSES)}
    else:
        names_dict = {i: f"class_{i}" for i in range(NUM_CLASSES)}

    yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val: images/val

nc: {NUM_CLASSES}
names: {names_dict}
"""
    yaml_path = output_dir / "dataset_pseudo.yaml"
    yaml_path.write_text(yaml_content)

    print("\n" + "=" * 60)
    print("Pseudo-labeling complete!")
    print(f"  Images processed:        {len(image_files)}")
    print(f"  Existing labels:         {total_existing}")
    print(f"  Pseudo-labels added:     {total_pseudo_added}")
    print(f"  Pseudo-labels rejected:  {total_pseudo_rejected} (IoU >= {args.iou_threshold})")
    print(f"  Images with new labels:  {images_with_new_labels}")
    print(f"  Output directory:        {output_dir}")
    print(f"  Dataset config:          {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
