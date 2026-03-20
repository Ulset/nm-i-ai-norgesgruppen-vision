# training/prepare_yolo_dataset.py
# NOTE: This is a TRAINING-ONLY script. It runs locally/on GCP, NOT in the sandbox.
# shutil is used here for file copying — it is blocked in sandbox submission code.
"""Generate YOLO-format dataset from COCO annotations.

Creates:
  data/yolo_dataset/images/{train,val}/  — copied images
  data/yolo_dataset/labels/{train,val}/  — YOLO label .txt files
  dataset.yaml                           — ultralytics dataset config

Usage:
  python training/prepare_yolo_dataset.py
"""

import json
import shutil
from pathlib import Path
from training.data_utils import (
    load_coco_annotations,
    create_train_val_split,
    coco_to_yolo_label,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COCO_DIR = DATA_DIR / "coco_dataset" / "train"
YOLO_DIR = DATA_DIR / "yolo_dataset"
NUM_CLASSES = 357


def main():
    annotations = load_coco_annotations(COCO_DIR / "annotations.json")
    images = annotations["images"]
    anns = annotations["annotations"]

    img_lookup = {img["id"]: img for img in images}

    img_to_anns: dict[int, list] = {}
    for ann in anns:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    train_ids, val_ids = create_train_val_split(images, anns, val_ratio=0.2, seed=42)

    # Save val image IDs for evaluation filtering
    val_ids_path = YOLO_DIR / "val_image_ids.json"
    val_ids_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_ids_path, "w") as f:
        json.dump(val_ids, f)

    for split, image_ids in [("train", train_ids), ("val", val_ids)]:
        img_dir = YOLO_DIR / "images" / split
        lbl_dir = YOLO_DIR / "labels" / split
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

    cat_names = {cat["id"]: cat["name"] for cat in annotations["categories"]}
    names_dict = {i: cat_names.get(i, f"class_{i}") for i in range(NUM_CLASSES)}

    yaml_content = f"""path: {YOLO_DIR.resolve()}
train: images/train
val: images/val

nc: {NUM_CLASSES}
names: {names_dict}
"""
    dataset_yaml = PROJECT_ROOT / "dataset.yaml"
    dataset_yaml.write_text(yaml_content)

    print(f"YOLO dataset created:")
    print(f"  Train: {len(train_ids)} images")
    print(f"  Val: {len(val_ids)} images")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"  Config: {dataset_yaml}")


if __name__ == "__main__":
    main()
