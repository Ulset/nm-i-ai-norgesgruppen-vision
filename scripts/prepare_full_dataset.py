# scripts/prepare_full_dataset.py
# NOTE: This is a TRAINING-ONLY script. It runs locally/on GCP, NOT in the sandbox.
"""Generate YOLO-format dataset using ALL images for training (no validation holdout).

This is for final submission training where we want to maximize training data.
A small dummy val set (20 images copied from train) is created so ultralytics
doesn't complain about a missing val split.

Creates:
  data/yolo_dataset_full/images/{train,val}/  — copied images
  data/yolo_dataset_full/labels/{train,val}/  — YOLO label .txt files
  dataset_full.yaml                           — ultralytics dataset config

Usage:
  python scripts/prepare_full_dataset.py
  python -m scripts.prepare_full_dataset
"""

import json
import shutil
from pathlib import Path
from training.data_utils import (
    load_coco_annotations,
    coco_to_yolo_label,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COCO_DIR = DATA_DIR / "coco_dataset" / "train"
YOLO_DIR = DATA_DIR / "yolo_dataset_full"
NUM_CLASSES = 357
DUMMY_VAL_COUNT = 20


def main():
    annotations = load_coco_annotations(COCO_DIR / "annotations.json")
    images = annotations["images"]
    anns = annotations["annotations"]

    img_lookup = {img["id"]: img for img in images}

    img_to_anns: dict[int, list] = {}
    for ann in anns:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    # Use ALL image IDs for training
    all_ids = sorted([img["id"] for img in images])

    # --- Write train split (all images) ---
    for split_name, image_ids in [("train", all_ids)]:
        img_dir = YOLO_DIR / "images" / split_name
        lbl_dir = YOLO_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_id in image_ids:
            img_info = img_lookup[img_id]
            src = COCO_DIR / "images" / img_info["file_name"]

            dst_name = Path(img_info["file_name"]).stem + ".jpg"
            dst = img_dir / dst_name
            if not dst.exists():
                shutil.copy2(src, dst)

            label_file = lbl_dir / (Path(img_info["file_name"]).stem + ".txt")
            img_anns = img_to_anns.get(img_id, [])
            lines = []
            for ann in img_anns:
                line = coco_to_yolo_label(ann, img_info["width"], img_info["height"])
                lines.append(line)
            label_file.write_text("\n".join(lines) + "\n" if lines else "")

    # --- Create dummy val split (copy first N images from train) ---
    dummy_val_ids = all_ids[:DUMMY_VAL_COUNT]
    val_img_dir = YOLO_DIR / "images" / "val"
    val_lbl_dir = YOLO_DIR / "labels" / "val"
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_id in dummy_val_ids:
        img_info = img_lookup[img_id]
        stem = Path(img_info["file_name"]).stem

        # Copy image from train to val
        train_img = YOLO_DIR / "images" / "train" / (stem + ".jpg")
        val_img = val_img_dir / (stem + ".jpg")
        if not val_img.exists():
            shutil.copy2(train_img, val_img)

        # Copy label from train to val
        train_lbl = YOLO_DIR / "labels" / "train" / (stem + ".txt")
        val_lbl = val_lbl_dir / (stem + ".txt")
        if not val_lbl.exists() and train_lbl.exists():
            shutil.copy2(train_lbl, val_lbl)

    # --- Generate dataset_full.yaml ---
    cat_names = {cat["id"]: cat["name"] for cat in annotations["categories"]}
    names_dict = {i: cat_names.get(i, f"class_{i}") for i in range(NUM_CLASSES)}

    yaml_content = f"""path: {YOLO_DIR.resolve()}
train: images/train
val: images/val

nc: {NUM_CLASSES}
names: {names_dict}
"""
    dataset_yaml = PROJECT_ROOT / "dataset_full.yaml"
    dataset_yaml.write_text(yaml_content)

    print(f"Full YOLO dataset created (no val holdout):")
    print(f"  Train: {len(all_ids)} images (100% of data)")
    print(f"  Val (dummy): {len(dummy_val_ids)} images (copied from train)")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"  Config: {dataset_yaml}")


if __name__ == "__main__":
    main()
