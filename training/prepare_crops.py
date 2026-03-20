# training/prepare_crops.py
"""Extract product crops from training images for classifier training.

Creates:
  data/crops/{train,val}/{category_id}/crop_{annotation_id}.jpg
  data/crops/train/{category_id}/ref_{product_code}_{angle}.jpg

Usage:
  python -m training.prepare_crops
"""

from pathlib import Path
from PIL import Image
from training.data_utils import (
    load_coco_annotations,
    load_product_metadata,
    build_product_category_mapping,
    create_train_val_split,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COCO_DIR = DATA_DIR / "coco_dataset" / "train"
PRODUCT_DIR = DATA_DIR / "product_images"
CROPS_DIR = DATA_DIR / "crops"
CROP_SIZE = 224


def extract_training_crops():
    """Extract annotated bounding box crops from shelf images."""
    annotations = load_coco_annotations(COCO_DIR / "annotations.json")
    images = annotations["images"]
    anns = annotations["annotations"]

    img_lookup = {img["id"]: img for img in images}
    train_ids, val_ids = create_train_val_split(images, anns, val_ratio=0.2, seed=42)
    val_id_set = set(val_ids)

    # Sort annotations by image_id for efficient loading
    sorted_anns = sorted(anns, key=lambda a: a["image_id"])

    current_img_id = None
    current_img = None
    count = 0

    for ann in sorted_anns:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        split = "val" if img_id in val_id_set else "train"

        out_dir = CROPS_DIR / split / str(cat_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"crop_{ann['id']}.jpg"
        if out_path.exists():
            continue

        # Load image (cached — reload only when image_id changes)
        if img_id != current_img_id:
            img_info = img_lookup[img_id]
            img_path = COCO_DIR / "images" / img_info["file_name"]
            current_img = Image.open(img_path).convert("RGB")
            current_img_id = img_id

        x, y, w, h = ann["bbox"]
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(current_img.width, int(x + w))
        y2 = min(current_img.height, int(y + h))

        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        crop = current_img.crop((x1, y1, x2, y2)).resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
        crop.save(out_path, quality=90)
        count += 1

    print(f"Extracted {count} crops from training images")


def copy_reference_images():
    """Copy product reference images into crops/train/{category_id}/ structure."""
    annotations = load_coco_annotations(COCO_DIR / "annotations.json")
    metadata = load_product_metadata(PRODUCT_DIR / "metadata.json")
    mapping = build_product_category_mapping(annotations, metadata)

    count = 0
    for product_code, cat_id in mapping.items():
        product_dir = PRODUCT_DIR / product_code
        if not product_dir.is_dir():
            continue

        out_dir = CROPS_DIR / "train" / str(cat_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in product_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            out_path = out_dir / f"ref_{product_code}_{img_path.stem}.jpg"
            if out_path.exists():
                continue

            img = Image.open(img_path).convert("RGB").resize(
                (CROP_SIZE, CROP_SIZE), Image.LANCZOS
            )
            img.save(out_path, quality=90)
            count += 1

    print(f"Copied {count} product reference images")


def main():
    extract_training_crops()
    copy_reference_images()

    for split in ("train", "val"):
        split_dir = CROPS_DIR / split
        if split_dir.exists():
            cats = list(split_dir.iterdir())
            total = sum(len(list(c.iterdir())) for c in cats if c.is_dir())
            print(f"  {split}: {total} crops across {len(cats)} categories")


if __name__ == "__main__":
    main()
