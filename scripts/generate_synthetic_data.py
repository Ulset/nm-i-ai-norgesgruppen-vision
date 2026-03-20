#!/usr/bin/env python3
# scripts/generate_synthetic_data.py
# NOTE: This is a TRAINING-ONLY script. It runs locally/on GCP, NOT in the sandbox.
"""Generate synthetic training images via copy-paste augmentation.

For underrepresented classes (<N annotations), pastes product reference images
onto random shelf backgrounds with slight geometric perturbations and produces
YOLO-format labels.

Creates:
  data/yolo_dataset_synthetic/images/train/  — synthetic images
  data/yolo_dataset_synthetic/labels/train/  — YOLO label .txt files
  dataset_synthetic.yaml                     — ultralytics dataset config

Usage:
  python scripts/generate_synthetic_data.py
  python scripts/generate_synthetic_data.py --num-synthetic 200 --min-annotations 15
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUM_CLASSES = 357


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic training images via copy-paste augmentation"
    )
    parser.add_argument(
        "--coco-annotations",
        type=Path,
        default=PROJECT_ROOT / "data" / "coco_dataset" / "train" / "annotations.json",
        help="Path to COCO annotations JSON",
    )
    parser.add_argument(
        "--product-images",
        type=Path,
        default=PROJECT_ROOT / "data" / "product_images",
        help="Path to product reference images directory",
    )
    parser.add_argument(
        "--shelf-images",
        type=Path,
        default=PROJECT_ROOT / "data" / "coco_dataset" / "train" / "images",
        help="Path to shelf background images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "yolo_dataset_synthetic",
        help="Output directory for synthetic dataset",
    )
    parser.add_argument(
        "--num-synthetic",
        type=int,
        default=100,
        help="Number of synthetic images to generate",
    )
    parser.add_argument(
        "--min-annotations",
        type=int,
        default=10,
        help="Generate synthetic data for classes below this annotation count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def load_coco_data(annotations_path: Path) -> dict:
    """Load COCO annotations and compute per-category annotation counts."""
    with open(annotations_path) as f:
        coco = json.load(f)

    cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])
    cat_name_to_id = {cat["name"].strip(): cat["id"] for cat in coco["categories"]}
    cat_id_to_name = {cat["id"]: cat["name"].strip() for cat in coco["categories"]}

    # Compute median bbox size per category for realistic placement
    cat_bbox_sizes: dict[int, list[tuple[int, int]]] = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        w, h = ann["bbox"][2], ann["bbox"][3]
        cat_bbox_sizes.setdefault(cid, []).append((w, h))

    cat_median_bbox: dict[int, tuple[int, int]] = {}
    for cid, sizes in cat_bbox_sizes.items():
        ws = sorted(s[0] for s in sizes)
        hs = sorted(s[1] for s in sizes)
        cat_median_bbox[cid] = (ws[len(ws) // 2], hs[len(hs) // 2])

    # Global median for categories with no per-class bbox stats
    all_ws = sorted(ann["bbox"][2] for ann in coco["annotations"])
    all_hs = sorted(ann["bbox"][3] for ann in coco["annotations"])
    global_median_bbox = (all_ws[len(all_ws) // 2], all_hs[len(all_hs) // 2])

    return {
        "coco": coco,
        "cat_counts": cat_counts,
        "cat_name_to_id": cat_name_to_id,
        "cat_id_to_name": cat_id_to_name,
        "cat_median_bbox": cat_median_bbox,
        "global_median_bbox": global_median_bbox,
    }


def build_barcode_to_category(
    coco_data: dict, product_images_dir: Path
) -> dict[str, int]:
    """Map product barcode directories to COCO category IDs via metadata.json."""
    metadata_path = product_images_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"WARNING: {metadata_path} not found. Cannot map barcodes to categories.")
        return {}

    with open(metadata_path) as f:
        metadata = json.load(f)

    cat_name_to_id = coco_data["cat_name_to_id"]
    mapping: dict[str, int] = {}

    for product in metadata.get("products", []):
        product_name = product.get("product_name", "").strip()
        product_code = product.get("product_code", "").strip()
        if product_name and product_code and product_name in cat_name_to_id:
            mapping[product_code] = cat_name_to_id[product_name]

    return mapping


def load_product_reference_images(
    product_images_dir: Path,
    barcode_to_cat: dict[str, int],
    target_cat_ids: set[int],
) -> dict[int, list[Path]]:
    """Load paths to product reference images for target categories.

    Prefers 'front' and 'main' views as they are most representative of
    the product appearance on shelves.

    Returns:
        Dict mapping category_id to list of image paths.
    """
    # Invert mapping: cat_id -> list of barcodes
    cat_to_barcodes: dict[int, list[str]] = {}
    for barcode, cat_id in barcode_to_cat.items():
        if cat_id in target_cat_ids:
            cat_to_barcodes.setdefault(cat_id, []).append(barcode)

    # Preferred image views, in order
    preferred_views = ["front", "main", "left", "right"]

    cat_images: dict[int, list[Path]] = {}
    for cat_id, barcodes in cat_to_barcodes.items():
        paths = []
        for barcode in barcodes:
            barcode_dir = product_images_dir / barcode
            if not barcode_dir.is_dir():
                continue
            # Collect preferred views first, then all others
            found = []
            for view in preferred_views:
                img_path = barcode_dir / f"{view}.jpg"
                if img_path.exists():
                    found.append(img_path)
            # If no preferred views, take whatever is available
            if not found:
                found = list(barcode_dir.glob("*.jpg"))
            paths.extend(found)
        if paths:
            cat_images[cat_id] = paths

    return cat_images


def apply_color_jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    """Apply random brightness, contrast, and saturation jitter."""
    # Brightness
    factor = rng.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast
    factor = rng.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # Saturation
    factor = rng.uniform(0.8, 1.2)
    img = ImageEnhance.Color(img).enhance(factor)

    # Sharpness
    factor = rng.uniform(0.8, 1.2)
    img = ImageEnhance.Sharpness(img).enhance(factor)

    return img


def paste_product_on_shelf(
    shelf_img: Image.Image,
    product_img: Image.Image,
    target_w: int,
    target_h: int,
    rng: random.Random,
) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    """Paste a product image onto a shelf background with augmentation.

    Returns:
        Tuple of (augmented_image, (x_center_norm, y_center_norm, w_norm, h_norm))
        or None if placement failed.
    """
    shelf_w, shelf_h = shelf_img.size

    # Apply random scale variation (0.7x to 1.3x)
    scale = rng.uniform(0.7, 1.3)
    paste_w = max(20, int(target_w * scale))
    paste_h = max(20, int(target_h * scale))

    # Ensure product fits in shelf image
    if paste_w >= shelf_w or paste_h >= shelf_h:
        paste_w = min(paste_w, shelf_w - 10)
        paste_h = min(paste_h, shelf_h - 10)

    if paste_w < 20 or paste_h < 20:
        return None

    # Resize product image
    product_resized = product_img.resize((paste_w, paste_h), Image.LANCZOS)

    # Apply color jitter
    product_resized = apply_color_jitter(product_resized, rng)

    # Apply slight rotation (-10 to +10 degrees)
    angle = rng.uniform(-10, 10)
    product_rotated = product_resized.rotate(
        angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0)
    )

    # Create an alpha mask: non-black pixels are opaque
    # Convert to numpy for thresholding, then back to PIL
    arr = np.array(product_rotated)
    # Pixels where any channel > 15 are considered product (not background)
    mask_arr = np.any(arr > 15, axis=2).astype(np.uint8) * 255

    # Apply slight Gaussian blur to mask edges for smoother blending
    mask = Image.fromarray(mask_arr, mode="L")
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1))

    rot_w, rot_h = product_rotated.size

    if rot_w >= shelf_w or rot_h >= shelf_h:
        return None

    # Random placement position
    max_x = shelf_w - rot_w
    max_y = shelf_h - rot_h
    if max_x <= 0 or max_y <= 0:
        return None

    x = rng.randint(0, max_x)
    y = rng.randint(0, max_y)

    # Optional slight transparency
    alpha_factor = rng.uniform(0.85, 1.0)
    if alpha_factor < 1.0:
        mask = Image.fromarray(
            (np.array(mask).astype(np.float32) * alpha_factor).astype(np.uint8),
            mode="L",
        )

    # Paste product onto shelf
    result = shelf_img.copy()
    result.paste(product_rotated, (x, y), mask)

    # Compute YOLO-format bbox
    cx = (x + rot_w / 2) / shelf_w
    cy = (y + rot_h / 2) / shelf_h
    bw = rot_w / shelf_w
    bh = rot_h / shelf_h

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))

    return result, (cx, cy, bw, bh)


def generate_synthetic_dataset(args: argparse.Namespace) -> None:
    """Main synthetic data generation pipeline."""
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print("Loading COCO annotations...")
    coco_data = load_coco_data(args.coco_annotations)

    print("Building barcode-to-category mapping...")
    barcode_to_cat = build_barcode_to_category(coco_data, args.product_images)
    print(f"  Mapped {len(barcode_to_cat)} barcodes to category IDs")

    # Find underrepresented classes
    cat_counts = coco_data["cat_counts"]
    all_cat_ids = {cat["id"] for cat in coco_data["coco"]["categories"]}
    underrep_ids = {
        cid for cid in all_cat_ids if cat_counts.get(cid, 0) < args.min_annotations
    }
    print(f"  Found {len(underrep_ids)} classes with < {args.min_annotations} annotations")

    # Load reference images for underrepresented classes
    print("Loading product reference images...")
    cat_images = load_product_reference_images(
        args.product_images, barcode_to_cat, underrep_ids
    )
    covered_ids = set(cat_images.keys())
    uncovered_ids = underrep_ids - covered_ids
    print(
        f"  Have reference images for {len(covered_ids)}/{len(underrep_ids)} "
        f"underrepresented classes"
    )
    if uncovered_ids:
        uncovered_names = [
            coco_data["cat_id_to_name"].get(cid, f"class_{cid}")
            for cid in sorted(uncovered_ids)
        ]
        print(
            f"  No reference images for {len(uncovered_ids)} classes: "
            f"{uncovered_names[:10]}..."
        )

    if not covered_ids:
        print("ERROR: No reference images available for any underrepresented class. Exiting.")
        return

    # Load shelf background image paths
    shelf_paths = sorted(args.shelf_images.glob("*.jpg"))
    if not shelf_paths:
        shelf_paths = sorted(args.shelf_images.glob("*.png"))
    if not shelf_paths:
        print(f"ERROR: No shelf images found in {args.shelf_images}. Exiting.")
        return
    print(f"  Found {len(shelf_paths)} shelf background images")

    # Create output directories
    img_out_dir = args.output_dir / "images" / "train"
    lbl_out_dir = args.output_dir / "labels" / "train"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    # Determine how many images to generate per class, weighted by scarcity
    covered_list = sorted(covered_ids)
    class_weights = {}
    for cid in covered_list:
        count = cat_counts.get(cid, 0)
        # Weight inversely proportional to annotation count
        class_weights[cid] = max(1, args.min_annotations - count)

    total_weight = sum(class_weights.values())
    class_budgets = {}
    for cid in covered_list:
        budget = max(1, int(args.num_synthetic * class_weights[cid] / total_weight))
        class_budgets[cid] = budget

    total_planned = sum(class_budgets.values())
    print(f"\nPlanned: {total_planned} synthetic images across {len(covered_list)} classes")

    # Generate synthetic images
    generated_count = 0
    failed_count = 0
    class_generated: Counter = Counter()

    for cat_id in covered_list:
        budget = class_budgets[cat_id]
        ref_paths = cat_images[cat_id]
        median_bbox = coco_data["cat_median_bbox"].get(
            cat_id, coco_data["global_median_bbox"]
        )
        cat_name = coco_data["cat_id_to_name"].get(cat_id, f"class_{cat_id}")

        for i in range(budget):
            # Pick random shelf background
            shelf_path = rng.choice(shelf_paths)
            try:
                shelf_img = Image.open(shelf_path).convert("RGB")
            except Exception:
                failed_count += 1
                continue

            # Determine number of product instances to paste (1-3 per image)
            num_instances = rng.randint(1, 3)
            labels = []
            current_img = shelf_img.copy()

            for _ in range(num_instances):
                # Pick random reference image
                ref_path = rng.choice(ref_paths)
                try:
                    product_img = Image.open(ref_path).convert("RGB")
                except Exception:
                    continue

                result = paste_product_on_shelf(
                    current_img,
                    product_img,
                    target_w=median_bbox[0],
                    target_h=median_bbox[1],
                    rng=rng,
                )
                if result is None:
                    continue

                current_img, (cx, cy, bw, bh) = result
                labels.append(f"{cat_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not labels:
                failed_count += 1
                continue

            # Save synthetic image and labels
            img_name = f"synth_{cat_id:04d}_{i:04d}.jpg"
            lbl_name = f"synth_{cat_id:04d}_{i:04d}.txt"

            current_img.save(str(img_out_dir / img_name), quality=95)
            (lbl_out_dir / lbl_name).write_text("\n".join(labels) + "\n")

            generated_count += 1
            class_generated[cat_id] += 1

        if class_generated[cat_id] > 0:
            print(
                f"  {cat_name} (ID {cat_id}): "
                f"{cat_counts.get(cat_id, 0)} real + {class_generated[cat_id]} synthetic"
            )

    print(f"\nGeneration complete:")
    print(f"  Generated: {generated_count} images")
    print(f"  Failed: {failed_count} images")
    print(f"  Output: {args.output_dir}")

    # Generate dataset_synthetic.yaml combining original + synthetic data
    original_yolo_dir = PROJECT_ROOT / "data" / "yolo_dataset"

    yaml_content = build_dataset_yaml(
        original_yolo_dir, args.output_dir, PROJECT_ROOT
    )
    yaml_path = PROJECT_ROOT / "dataset_synthetic.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  Dataset config: {yaml_path}")

    # Print summary statistics
    print(f"\nPer-class summary:")
    for cat_id in sorted(class_generated.keys()):
        name = coco_data["cat_id_to_name"].get(cat_id, f"class_{cat_id}")
        real = cat_counts.get(cat_id, 0)
        synth = class_generated[cat_id]
        print(f"  {name}: {real} real + {synth} synthetic = {real + synth} total")


def build_dataset_yaml(
    original_yolo_dir: Path,
    synthetic_dir: Path,
    project_root: Path,
) -> str:
    """Build a YAML config that combines original and synthetic training data.

    The YAML format uses a list of training directories so that ultralytics
    picks up images from both the original and synthetic datasets.
    """
    coco_path = project_root / "data" / "coco_dataset" / "train" / "annotations.json"
    with open(coco_path) as f:
        coco = json.load(f)

    cat_names = {cat["id"]: cat["name"] for cat in coco["categories"]}
    names_dict = {i: cat_names.get(i, f"class_{i}") for i in range(NUM_CLASSES)}

    train_paths = [
        str(original_yolo_dir.resolve() / "images" / "train"),
        str(synthetic_dir.resolve() / "images" / "train"),
    ]
    val_path = str(original_yolo_dir.resolve() / "images" / "val")

    yaml_content = (
        f"path: {project_root.resolve()}\n"
        f"train:\n"
        f"  - {train_paths[0]}\n"
        f"  - {train_paths[1]}\n"
        f"val: {val_path}\n"
        f"\n"
        f"nc: {NUM_CLASSES}\n"
        f"names: {names_dict}\n"
    )

    return yaml_content


def main() -> None:
    args = parse_args()
    generate_synthetic_dataset(args)


if __name__ == "__main__":
    main()
