# Object Detection Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage object detection + classification pipeline for the NM i AI NorgesGruppen challenge, targeting 0.75-0.85 composite mAP.

**Architecture:** YOLOv8x detector with tiled inference for high-res shelf images, followed by EfficientNet-B2 crop classifier with product reference embedding fallback. Both models exported as ONNX FP16. Inference runs in a sandboxed Docker container (L4 GPU, 8GB RAM, 300s timeout, no network).

**Tech Stack:** Python 3.11, ultralytics 8.1.0, timm 0.9.12, onnxruntime-gpu, ensemble-boxes (WBF), pycocotools, numpy, Pillow, pathlib. Training on GCP (`ainm26osl-716`).

**Spec:** `docs/superpowers/specs/2026-03-20-object-detection-pipeline-design.md`

---

## File Structure

```
norges-gruppen-computer-vision/
├── data/                              # gitignored — training data
│   ├── coco_dataset/train/
│   │   ├── images/                    # 248 shelf images
│   │   └── annotations.json          # COCO format annotations
│   ├── product_images/                # 344 product reference folders
│   ├── yolo_dataset/                  # Generated: YOLO format for ultralytics
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   └── crops/                         # Generated: extracted product crops
│       ├── train/                     # One subfolder per category_id
│       └── val/
├── training/
│   ├── data_utils.py                  # Train/val split, COCO→YOLO conversion, product mapping
│   ├── prepare_yolo_dataset.py        # Script: generate YOLO format dataset
│   ├── prepare_crops.py               # Script: extract product crops from training data
│   ├── train_yolo.py                  # Script: YOLOv8x fine-tuning (runs on GCP)
│   ├── train_classifier.py            # Script: EfficientNet-B2 training (runs on GCP)
│   ├── build_reference_embeddings.py  # Script: pre-compute reference embeddings (runs on GCP)
│   └── export_models.py              # Script: export ONNX FP16 (runs on GCP)
├── submission/
│   ├── run.py                         # Entry point for sandbox
│   └── utils.py                       # Tiled inference, WBF, classification logic
├── scripts/
│   ├── setup_gcp_vm.sh                # Provision GPU VM
│   ├── upload_data.sh                 # Upload data to VM
│   ├── build_submission.sh            # Package submission.zip
│   └── evaluate_local.py             # Local mAP evaluation
├── tests/
│   ├── test_data_utils.py             # Tests for data utilities
│   ├── test_submission_utils.py       # Tests for inference utilities
│   └── test_evaluate.py              # Tests for evaluation script
├── dataset.yaml                       # YOLO dataset config (generated)
├── TASK.md
├── CLAUDE.md
└── .gitignore
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `.gitignore`
- Create: `CLAUDE.md`
- Create: `training/__init__.py`
- Create: `submission/__init__.py`
- Create: `scripts/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create .gitignore**

```gitignore
# Data
data/
*.zip

# Model weights
*.pt
*.pth
*.onnx
*.safetensors
*.npy

# Training outputs
runs/
wandb/

# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# OS
.DS_Store
__MACOSX/

# IDE
.idea/
.vscode/

# Submission artifacts
submission.zip
```

- [ ] **Step 2: Create CLAUDE.md**

```markdown
# NorgesGruppen Object Detection — NM i AI 2026

## Competition
- **Task**: Detect and classify grocery products on store shelves
- **Score**: 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5
- **Submission**: .zip with run.py + model weights, runs in sandboxed Docker (L4 GPU, 300s timeout)
- **Submission URL**: https://app.ainm.no/submit/norgesgruppen-data

## GCP Infrastructure
- **Project**: `ainm26osl-716` (unlimited compute)
- **User**: `devstar7161@gcplab.me`
- **Service account**: `59268370848-compute@developer.gserviceaccount.com`
- **gcloud profile**: `nmiai-unlimited`
- **Region**: `europe-north1`

## Key Constraints
- Sandbox: Python 3.11, L4 GPU (24GB VRAM), 8GB RAM, 300s timeout, NO network
- Blocked imports: os, sys, subprocess, pickle, yaml, threading, multiprocessing
- Use `pathlib` instead of `os`, `json` instead of `yaml`
- Max 3 weight files, 420MB total, 10 Python files
- Pin `ultralytics==8.1.0`, `timm==0.9.12` for training (match sandbox versions)
- Train with nc=357 (IDs 0-355 + 356 safety)
- 3 submissions/day max — validate locally first

## Running
```bash
# Prepare YOLO dataset from COCO annotations
python training/prepare_yolo_dataset.py

# Extract product crops for classifier training
python training/prepare_crops.py

# Local evaluation
python scripts/evaluate_local.py --predictions predictions.json --ground-truth data/coco_dataset/train/annotations.json

# Build submission zip
bash scripts/build_submission.sh
```

## Git Commits
- **NEVER add Co-Authored-By lines** to commit messages
```

- [ ] **Step 3: Create __init__.py files for all packages**

Create empty `__init__.py` files for `training/`, `submission/`, `scripts/`, and `tests/`. These are needed so cross-module imports work from the first test run.

- [ ] **Step 4: Commit**

```bash
git add .gitignore CLAUDE.md training/__init__.py submission/__init__.py scripts/__init__.py tests/__init__.py
git commit -m "Add project scaffolding"
```

---

## Task 2: Data Utilities — Product Mapping & Train/Val Split

**Files:**
- Create: `training/data_utils.py`
- Create: `tests/test_data_utils.py`

- [ ] **Step 1: Write tests for product-to-category mapping**

```python
# tests/test_data_utils.py
import json
import pytest
from pathlib import Path
from training.data_utils import build_product_category_mapping, create_train_val_split


class TestProductCategoryMapping:
    def test_exact_name_match(self, tmp_path):
        """Products match categories by exact name."""
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

    def test_empty_name_category_excluded(self, tmp_path):
        """Category with empty name should be excluded from mapping."""
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
        """Products with no matching category name are not in mapping."""
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
        """80/20 split produces correct counts."""
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
        """Same seed produces same split."""
        images = [{"id": i, "file_name": f"img_{i:05d}.jpg"} for i in range(50)]
        annotations = [{"id": i, "image_id": i % 50, "category_id": 0, "bbox": [0, 0, 10, 10]} for i in range(100)]

        split1 = create_train_val_split(images, annotations, val_ratio=0.2, seed=42)
        split2 = create_train_val_split(images, annotations, val_ratio=0.2, seed=42)
        assert split1 == split2

    def test_all_images_assigned(self):
        """Every image ends up in either train or val."""
        images = [{"id": i, "file_name": f"img_{i:05d}.jpg"} for i in range(50)]
        annotations = [{"id": i, "image_id": i % 50, "category_id": 0, "bbox": [0, 0, 10, 10]} for i in range(100)]

        train_ids, val_ids = create_train_val_split(images, annotations, val_ratio=0.2, seed=42)
        assert set(train_ids) | set(val_ids) == {i for i in range(50)}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'training'`

- [ ] **Step 3: Implement data_utils.py**

```python
# training/data_utils.py
"""Utilities for data preparation: product mapping, train/val split, COCO→YOLO conversion."""

import json
import random
from pathlib import Path
from collections import defaultdict


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
        if name:  # Exclude empty-name categories (e.g., category 300)
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

    Uses random shuffle with fixed seed for reproducibility.

    Args:
        images: List of COCO image dicts (need 'id' key).
        annotations: List of COCO annotation dicts (unused but available for stratification).
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

    Returns:
        String like "42 0.5123 0.3456 0.1234 0.0987"
    """
    cat_id = annotation["category_id"]
    x, y, w, h = annotation["bbox"]

    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    w_norm = w / image_width
    h_norm = h / image_height

    # Clamp to [0, 1]
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
```

- [ ] **Step 4: Add COCO→YOLO conversion tests**

Add to `tests/test_data_utils.py`:

```python
from training.data_utils import coco_to_yolo_label


class TestCocoToYoloLabel:
    def test_basic_conversion(self):
        """Convert COCO bbox to YOLO normalized format."""
        annotation = {"category_id": 5, "bbox": [100, 200, 50, 80]}
        result = coco_to_yolo_label(annotation, image_width=1000, image_height=800)
        # x_center = (100 + 25) / 1000 = 0.125
        # y_center = (200 + 40) / 800 = 0.3
        # w = 50 / 1000 = 0.05
        # h = 80 / 800 = 0.1
        assert result == "5 0.125000 0.300000 0.050000 0.100000"

    def test_clamps_to_unit_range(self):
        """Bbox extending beyond image edges should be clamped."""
        annotation = {"category_id": 0, "bbox": [950, 750, 100, 100]}
        result = coco_to_yolo_label(annotation, image_width=1000, image_height=800)
        parts = result.split()
        assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])
```

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/test_data_utils.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add training/data_utils.py tests/test_data_utils.py
git commit -m "Add data utilities: product mapping, train/val split, COCO-to-YOLO conversion"
```

---

## Task 3: YOLO Dataset Preparation Script

**Files:**
- Create: `training/prepare_yolo_dataset.py`

- [ ] **Step 1: Write the preparation script**

```python
# training/prepare_yolo_dataset.py
# NOTE: This is a TRAINING-ONLY script. It runs locally/on GCP, NOT in the sandbox.
# shutil is used here for file copying — it is blocked in sandbox submission code.
"""Generate YOLO-format dataset from COCO annotations.

Creates:
  data/yolo_dataset/images/{train,val}/  — symlinked images
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

    # Build image_id → image info lookup
    img_lookup = {img["id"]: img for img in images}

    # Build image_id → annotations lookup
    img_to_anns: dict[int, list] = {}
    for ann in anns:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    # Create train/val split
    train_ids, val_ids = create_train_val_split(images, anns, val_ratio=0.2, seed=42)

    for split, image_ids in [("train", train_ids), ("val", val_ids)]:
        img_dir = YOLO_DIR / "images" / split
        lbl_dir = YOLO_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_id in image_ids:
            img_info = img_lookup[img_id]
            src = COCO_DIR / "images" / img_info["file_name"]

            # Copy image (use .jpg extension consistently)
            dst_name = Path(img_info["file_name"]).stem + ".jpg"
            dst = img_dir / dst_name
            if not dst.exists():
                shutil.copy2(src, dst)

            # Write YOLO labels
            label_file = lbl_dir / (Path(img_info["file_name"]).stem + ".txt")
            img_anns = img_to_anns.get(img_id, [])
            lines = []
            for ann in img_anns:
                line = coco_to_yolo_label(ann, img_info["width"], img_info["height"])
                lines.append(line)
            label_file.write_text("\n".join(lines) + "\n" if lines else "")

    # Write dataset.yaml
    # Build category names list
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
```

- [ ] **Step 2: Run the script to generate dataset**

Run: `python training/prepare_yolo_dataset.py`
Expected: Prints train/val counts, creates `data/yolo_dataset/` and `dataset.yaml`

- [ ] **Step 3: Verify output**

Run: `ls data/yolo_dataset/images/train/ | wc -l && ls data/yolo_dataset/labels/train/ | wc -l`
Expected: ~198 images and ~198 label files

- [ ] **Step 4: Commit**

```bash
git add training/prepare_yolo_dataset.py
git commit -m "Add YOLO dataset preparation script"
```

---

## Task 4: Crop Extraction Script

**Files:**
- Create: `training/prepare_crops.py`

- [ ] **Step 1: Write the crop extraction script**

```python
# training/prepare_crops.py
"""Extract product crops from training images for classifier training.

Creates:
  data/crops/{train,val}/{category_id}/crop_{annotation_id}.jpg

Also copies and augments product reference images into the same structure.

Usage:
  python training/prepare_crops.py
"""

import json
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

    img_cache = {}
    count = 0

    for ann in anns:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        split = "val" if img_id in val_id_set else "train"

        out_dir = CROPS_DIR / split / str(cat_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"crop_{ann['id']}.jpg"
        if out_path.exists():
            continue

        # Load image (cached per image_id)
        if img_id not in img_cache:
            img_info = img_lookup[img_id]
            img_path = COCO_DIR / "images" / img_info["file_name"]
            img_cache = {img_id: Image.open(img_path).convert("RGB")}

        img = img_cache[img_id]
        x, y, w, h = ann["bbox"]

        # Crop with bounds checking
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img.width, int(x + w))
        y2 = min(img.height, int(y + h))

        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        crop = img.crop((x1, y1, x2, y2)).resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
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

    # Print summary
    for split in ("train", "val"):
        split_dir = CROPS_DIR / split
        if split_dir.exists():
            cats = list(split_dir.iterdir())
            total = sum(len(list(c.iterdir())) for c in cats if c.is_dir())
            print(f"  {split}: {total} crops across {len(cats)} categories")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run to verify**

Run: `python training/prepare_crops.py`
Expected: Prints crop counts for train and val splits

- [ ] **Step 3: Commit**

```bash
git add training/prepare_crops.py
git commit -m "Add crop extraction script for classifier training data"
```

---

## Task 5: Submission Inference Utilities

**Files:**
- Create: `submission/utils.py`
- Create: `tests/test_submission_utils.py`

This is the core inference logic that runs in the sandbox. Must avoid all blocked imports.

- [ ] **Step 1: Write tests for tiling logic**

```python
# tests/test_submission_utils.py
import numpy as np
import pytest
from submission.utils import (
    compute_tiles,
    map_tile_boxes_to_image,
    classify_detections,
    compute_final_score,
)


class TestComputeTiles:
    def test_small_image_no_tiling(self):
        """Images <= tile_size should produce a single full-image tile."""
        tiles = compute_tiles(800, 600, tile_size=1280, overlap=0.2)
        assert len(tiles) == 1
        assert tiles[0] == (0, 0, 800, 600)

    def test_large_image_produces_tiles(self):
        """Images > tile_size should be sliced with overlap."""
        tiles = compute_tiles(3000, 2000, tile_size=1280, overlap=0.2)
        assert len(tiles) > 1
        # Each tile should not exceed tile_size
        for x, y, w, h in tiles:
            assert w <= 1280
            assert h <= 1280

    def test_tiles_cover_full_image(self):
        """Union of tiles should cover the entire image."""
        img_w, img_h = 3000, 2000
        tiles = compute_tiles(img_w, img_h, tile_size=1280, overlap=0.2)

        # Check every pixel is covered by at least one tile
        covered = np.zeros((img_h, img_w), dtype=bool)
        for x, y, w, h in tiles:
            covered[y:y+h, x:x+w] = True
        assert covered.all()


class TestMapTileBoxes:
    def test_offset_applied(self):
        """Boxes from tiles should be offset to image coordinates."""
        tile_boxes = np.array([[10.0, 20.0, 50.0, 60.0]])  # x1, y1, x2, y2
        tile_offset = (100, 200)  # tile starts at (100, 200) in image
        result = map_tile_boxes_to_image(tile_boxes, tile_offset)
        expected = np.array([[110.0, 220.0, 150.0, 260.0]])
        np.testing.assert_array_equal(result, expected)


class TestClassifyDetections:
    def test_high_confidence_classifier_wins(self):
        """When classifier confidence > threshold, use classifier category."""
        result = classify_detections(
            yolo_category=5,
            yolo_class_conf=0.3,
            classifier_probs=np.array([0.0] * 5 + [0.1] + [0.0] * 3 + [0.8] + [0.0] * 348),
            reference_embeddings=np.zeros((357, 1408)),
            crop_embedding=np.zeros(1408),
            classifier_threshold=0.7,
            reference_threshold=0.8,
        )
        assert result == 9  # classifier top-1

    def test_yolo_fallback(self):
        """When both classifier and reference are low confidence, use YOLO category."""
        result = classify_detections(
            yolo_category=42,
            yolo_class_conf=0.5,
            classifier_probs=np.ones(357) / 357,  # uniform = low confidence
            reference_embeddings=np.random.randn(357, 1408).astype(np.float32),
            crop_embedding=np.random.randn(1408).astype(np.float32),
            classifier_threshold=0.7,
            reference_threshold=0.8,
        )
        assert result == 42


class TestComputeFinalScore:
    def test_score_multiplication(self):
        """Final score is yolo_conf * classification_conf."""
        score = compute_final_score(yolo_conf=0.9, classification_conf=0.8)
        assert abs(score - 0.72) < 1e-6

    def test_score_clamped(self):
        """Score should be between 0 and 1."""
        score = compute_final_score(yolo_conf=1.5, classification_conf=0.8)
        assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_submission_utils.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement utils.py**

```python
# submission/utils.py
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

    Args:
        img_width: Image width in pixels.
        img_height: Image height in pixels.
        tile_size: Max tile dimension.
        overlap: Fraction of overlap between adjacent tiles.

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

    Returns:
        Final category_id.
    """
    # Signal 1: Classifier
    clf_top1 = int(np.argmax(classifier_probs))
    clf_conf = float(classifier_probs[clf_top1])

    if clf_conf > classifier_threshold:
        return clf_top1

    # Signal 2: Reference embedding nearest neighbor
    if reference_embeddings is not None and crop_embedding is not None:
        # Cosine similarity (embeddings should be L2-normalized)
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

    # Signal 3: YOLO fallback
    if yolo_class_conf > 0.15 or clf_conf > 0.15:
        return yolo_category

    # Very low confidence across all signals → unknown_product
    return 355  # Configurable via UNKNOWN_PRODUCT_ID in run.py


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
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
    return xywh


def load_image(path: Path) -> Image.Image:
    """Load image from path, convert to RGB."""
    return Image.open(path).convert("RGB")


def image_id_from_filename(filename: str) -> int:
    """Extract numeric image_id from filename like img_00042.jpg → 42."""
    stem = Path(filename).stem
    return int(stem.split("_")[-1])
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_submission_utils.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add submission/utils.py tests/test_submission_utils.py
git commit -m "Add submission inference utilities: tiling, WBF, classification logic"
```

---

## Task 6: Submission Entry Point (run.py)

**Files:**
- Create: `submission/run.py`

- [ ] **Step 1: Write run.py**

```python
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
CONFIDENCE_FLOOR = 0.05  # Skip detections below this
UNKNOWN_PRODUCT_ID = 355  # May be 356 per TASK.md — verify with first submission


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
    # Preprocess: resize, normalize, CHW, batch
    orig_w, orig_h = tile_img.size
    img_resized = tile_img.resize((input_size, input_size), Image.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: arr})

    # YOLO ONNX output: (1, num_classes+4, num_detections) — transposed
    output = outputs[0]  # (1, 4+nc, N)
    if output.ndim == 3:
        output = output[0].T  # (N, 4+nc)

    if len(output) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    # Parse: first 4 columns are xywh (center format), rest are class scores
    boxes_cxcywh = output[:, :4]
    class_scores = output[:, 4:]

    # Convert center format to xyxy
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Scale from input_size back to tile coordinates
    scale_x = orig_w / input_size
    scale_y = orig_h / input_size
    boxes_xyxy[:, [0, 2]] *= scale_x
    boxes_xyxy[:, [1, 3]] *= scale_y

    # Get best class per detection
    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    # Filter low confidence
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

    # Stack and preprocess
    batch = np.stack(crops).astype(np.float32) / 255.0
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch = (batch - mean) / std
    batch = np.transpose(batch, (0, 3, 1, 2))  # NCHW

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})

    probs = outputs[0]  # (N, num_classes)
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

    all_boxes = []  # List of (N, 4) arrays normalized to [0, 1]
    all_scores = []  # List of (N,) arrays
    all_labels = []  # List of (N,) arrays

    for tx, ty, tw, th in tiles:
        tile_img = img.crop((tx, ty, tx + tw, ty + th))
        boxes, scores, class_ids = run_yolo_on_tile(yolo_session, tile_img)

        if len(boxes) == 0:
            continue

        # Map to full image coordinates
        boxes = map_tile_boxes_to_image(boxes, (tx, ty))

        # Normalize to [0, 1] for WBF
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
        # Extract crops and classify in batches
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
                yolo_class_conf = yolo_conf  # Approximate

                # Remap classifier probs from ImageFolder index to category_id
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

                clf_conf = float(all_probs[j].max())
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

    # Locate model files (same directory as run.py)
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
        # Load class mapping (ImageFolder idx → category_id)
        # CRITICAL: ImageFolder sorts folder names lexicographically,
        # so index 0 != category_id 0. This mapping corrects that.
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
```

- [ ] **Step 2: Verify no blocked imports**

Run: `python -c "import ast; tree = ast.parse(open('submission/run.py').read()); imports = [n.names[0].name if isinstance(n, ast.Import) else n.module for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]; blocked = {'os','sys','subprocess','socket','ctypes','builtins','importlib','pickle','marshal','shelve','shutil','yaml','requests','urllib','http','multiprocessing','threading','signal','gc'}; violations = [i for i in imports if i and i.split('.')[0] in blocked]; print('BLOCKED:' if violations else 'CLEAN', violations)"`
Expected: `CLEAN []`

- [ ] **Step 3: Commit**

```bash
git add submission/run.py
git commit -m "Add submission entry point with two-stage inference pipeline"
```

---

## Task 7: YOLO Training Script

**Files:**
- Create: `training/train_yolo.py`

Runs on GCP GPU VM. Uses ultralytics 8.1.0.

- [ ] **Step 1: Write training script**

```python
# training/train_yolo.py
"""Fine-tune YOLOv8x on NorgesGruppen shelf data.

Runs on GCP GPU VM with ultralytics==8.1.0.

Usage:
  python training/train_yolo.py [--imgsz 1280] [--epochs 150] [--batch 4]
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--model", default="yolov8x.pt", help="Base model")
    parser.add_argument("--dataset", default="dataset.yaml")
    parser.add_argument("--name", default="yolov8x_shelf")
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.dataset,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
        # Augmentation
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        # Training
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Output
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    print(f"Training complete. Best weights: runs/detect/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add training/train_yolo.py
git commit -m "Add YOLOv8x training script for GCP"
```

---

## Task 8: Classifier Training Script

**Files:**
- Create: `training/train_classifier.py`

Runs on GCP GPU VM. Uses timm 0.9.12.

- [ ] **Step 1: Write classifier training script**

```python
# training/train_classifier.py
"""Train EfficientNet-B2 product classifier on extracted crops.

Runs on GCP GPU VM with timm==0.9.12.
Expects crops at data/crops/{train,val}/{category_id}/*.jpg

Usage:
  python training/train_classifier.py [--epochs 50] [--batch 64] [--lr 1e-4]
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


NUM_CLASSES = 357
IMG_SIZE = 224


def build_transforms(is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """Inverse frequency class weights for imbalanced dataset."""
    counts = Counter(dataset.targets)
    total = len(dataset.targets)
    weights = torch.zeros(NUM_CLASSES)
    for cls_idx, count in counts.items():
        weights[cls_idx] = total / (len(counts) * count)
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/crops")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output-dir", default="runs/classifier")
    parser.add_argument("--name", default="efficientnet_b2_shelf")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    train_dataset = datasets.ImageFolder(
        Path(args.data_dir) / "train",
        transform=build_transforms(is_train=True),
    )
    val_dataset = datasets.ImageFolder(
        Path(args.data_dir) / "val",
        transform=build_transforms(is_train=False),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model
    model = timm.create_model("efficientnet_b2", pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)

    # Loss with class weights
    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / max(val_total, 1)
        avg_loss = train_loss / train_total

        print(f"Epoch {epoch+1}/{args.epochs} — loss: {avg_loss:.4f}, "
              f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final
    torch.save(model.state_dict(), output_dir / "last.pt")

    # Save class mapping (ImageFolder uses sorted folder names)
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: int(k) for k, v in class_to_idx.items()}
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(idx_to_class, f)

    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Weights: {output_dir / 'best.pt'}")
    print(f"Class mapping: {output_dir / 'class_mapping.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add training/train_classifier.py
git commit -m "Add EfficientNet-B2 classifier training script"
```

---

## Task 9: Reference Embedding Builder

**Files:**
- Create: `training/build_reference_embeddings.py`

- [ ] **Step 1: Write embedding builder**

```python
# training/build_reference_embeddings.py
"""Pre-compute reference embeddings from product images using trained classifier.

Runs on GCP after classifier training. Produces reference_embeddings.npy.

Usage:
  python training/build_reference_embeddings.py \
    --model-weights runs/classifier/efficientnet_b2_shelf/best.pt \
    --class-mapping runs/classifier/efficientnet_b2_shelf/class_mapping.json \
    --product-images data/product_images \
    --annotations data/coco_dataset/train/annotations.json \
    --output reference_embeddings.npy
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import timm
from torchvision import transforms

from training.data_utils import (
    load_coco_annotations,
    load_product_metadata,
    build_product_category_mapping,
)

NUM_CLASSES = 357
IMG_SIZE = 224
EMBED_DIM = 1408  # EfficientNet-B2 penultimate layer


def get_feature_extractor(model: torch.nn.Module) -> torch.nn.Module:
    """Remove classification head, return feature extractor."""
    model.classifier = torch.nn.Identity()
    return model


def embed_images(
    model: torch.nn.Module,
    image_paths: list[Path],
    device: torch.device,
) -> np.ndarray:
    """Compute embeddings for a list of images.

    Returns:
        (N, EMBED_DIM) float32 array, L2-normalized.
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = []
    model.eval()

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            embedding = model(tensor).cpu().numpy().flatten()
            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
            embeddings.append(embedding)

    return np.array(embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-weights", required=True)
    parser.add_argument("--class-mapping", required=True)
    parser.add_argument("--product-images", default="data/product_images")
    parser.add_argument("--annotations", default="data/coco_dataset/train/annotations.json")
    parser.add_argument("--output", default="reference_embeddings.npy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model as feature extractor
    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(args.model_weights, map_location=device)
    model.load_state_dict(state_dict)
    model = get_feature_extractor(model)
    model = model.to(device)

    # Load class mapping (ImageFolder idx → category_id)
    with open(args.class_mapping) as f:
        idx_to_cat = json.load(f)
    idx_to_cat = {int(k): int(v) for k, v in idx_to_cat.items()}

    # Build product_code → category_id mapping
    annotations = load_coco_annotations(Path(args.annotations))
    metadata = load_product_metadata(Path(args.product_images) / "metadata.json")
    product_mapping = build_product_category_mapping(annotations, metadata)

    # Compute embeddings per category
    # Shape: (NUM_CLASSES, EMBED_DIM), zero vector for categories without reference images
    reference_embeddings = np.zeros((NUM_CLASSES, EMBED_DIM), dtype=np.float16)
    product_dir = Path(args.product_images)

    category_count = 0
    for product_code, cat_id in product_mapping.items():
        prod_path = product_dir / product_code
        if not prod_path.is_dir():
            continue

        # Prioritize main and front images (closest to shelf appearance)
        priority_names = ["main.jpg", "front.jpg"]
        image_paths = []
        for name in priority_names:
            p = prod_path / name
            if p.exists():
                image_paths.append(p)

        # Add other angles as fallback
        for p in sorted(prod_path.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p not in image_paths:
                image_paths.append(p)

        if not image_paths:
            continue

        embeddings = embed_images(model, image_paths, device)
        # Average embedding for this category
        avg_embedding = embeddings.mean(axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 1e-8:
            avg_embedding = avg_embedding / norm

        reference_embeddings[cat_id] = avg_embedding.astype(np.float16)
        category_count += 1

    np.save(args.output, reference_embeddings)
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Saved {args.output} ({size_mb:.1f} MB)")
    print(f"Embedded {category_count} categories out of {NUM_CLASSES}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add training/build_reference_embeddings.py
git commit -m "Add reference embedding builder for product classification"
```

---

## Task 10: Model Export Script

**Files:**
- Create: `training/export_models.py`

- [ ] **Step 1: Write export script**

```python
# training/export_models.py
"""Export trained models to ONNX FP16 for submission.

Usage:
  python training/export_models.py \
    --yolo-weights runs/detect/yolov8x_shelf/weights/best.pt \
    --clf-weights runs/classifier/efficientnet_b2_shelf/best.pt \
    --output-dir submission/
"""

import argparse
from pathlib import Path

import torch
import timm
from ultralytics import YOLO

NUM_CLASSES = 357
IMG_SIZE_CLF = 224


def export_yolo(weights_path: str, output_dir: Path):
    """Export YOLOv8x to ONNX FP16."""
    model = YOLO(weights_path)
    model.export(
        format="onnx",
        imgsz=1280,
        opset=17,
        half=True,
        simplify=True,
    )
    # ultralytics saves .onnx next to the .pt file
    onnx_path = Path(weights_path).with_suffix(".onnx")
    dest = output_dir / "yolo_detector.onnx"
    onnx_path.rename(dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"YOLO exported: {dest} ({size_mb:.1f} MB)")


def export_classifier(weights_path: str, output_dir: Path):
    """Export EfficientNet-B2 to ONNX FP16 with dual output (probs + embeddings)."""
    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Create a wrapper that outputs both classification logits and embeddings
    class DualOutputModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.features = base_model
            # Store original classifier
            self.classifier = base_model.classifier
            # Replace with identity to get embeddings
            base_model.classifier = torch.nn.Identity()

        def forward(self, x):
            embeddings = self.features(x)
            logits = self.classifier(embeddings)
            return logits, embeddings

    dual_model = DualOutputModel(model)
    dual_model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE_CLF, IMG_SIZE_CLF)

    dest = output_dir / "classifier.onnx"
    torch.onnx.export(
        dual_model,
        dummy_input,
        str(dest),
        opset_version=17,
        input_names=["input"],
        output_names=["logits", "embeddings"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "embeddings": {0: "batch_size"},
        },
    )

    # Convert to FP16
    import onnx
    from onnx import numpy_helper
    onnx_model = onnx.load(str(dest))
    for tensor in onnx_model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT:
            arr = numpy_helper.to_array(tensor).astype("float16")
            new_tensor = numpy_helper.from_array(arr, tensor.name)
            tensor.CopyFrom(new_tensor)
    onnx.save(onnx_model, str(dest))

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Classifier exported: {dest} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--clf-weights", required=True)
    parser.add_argument("--output-dir", default="submission")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_yolo(args.yolo_weights, output_dir)
    export_classifier(args.clf_weights, output_dir)

    # Summary
    total_size = sum(
        f.stat().st_size for f in output_dir.iterdir()
        if f.suffix in (".onnx", ".npy")
    ) / (1024 * 1024)
    print(f"\nTotal weight size: {total_size:.1f} MB / 420 MB limit")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add training/export_models.py
git commit -m "Add ONNX FP16 model export script"
```

---

## Task 11: Local Evaluation Script

**Files:**
- Create: `scripts/evaluate_local.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write test for score computation**

```python
# tests/test_evaluate.py
from scripts.evaluate_local import compute_composite_score


class TestCompositeScore:
    def test_formula(self):
        """Score = 0.7 * det_mAP + 0.3 * cls_mAP."""
        score = compute_composite_score(det_map=0.8, cls_map=0.5)
        assert abs(score - 0.71) < 1e-6

    def test_perfect_score(self):
        score = compute_composite_score(det_map=1.0, cls_map=1.0)
        assert abs(score - 1.0) < 1e-6

    def test_detection_only(self):
        """Detection-only (cls=0) gives max 0.7."""
        score = compute_composite_score(det_map=1.0, cls_map=0.0)
        assert abs(score - 0.7) < 1e-6
```

- [ ] **Step 2: Write evaluation script**

```python
# scripts/evaluate_local.py
"""Local evaluation script replicating competition scoring.

Usage:
  python scripts/evaluate_local.py \
    --predictions predictions.json \
    --ground-truth data/coco_dataset/train/annotations.json \
    --val-image-ids data/yolo_dataset/val_image_ids.json
"""

import argparse
import json
import copy
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_composite_score(det_map: float, cls_map: float) -> float:
    """Compute competition composite score."""
    return 0.7 * det_map + 0.3 * cls_map


def evaluate_detection_map(coco_gt: COCO, predictions: list[dict], image_ids: set[int] | None = None) -> float:
    """Compute detection mAP@0.5 (category-agnostic).

    All predictions and ground truth mapped to a single category for
    category-agnostic evaluation.
    """
    # Remap everything to category 0
    det_preds = []
    for p in predictions:
        det_preds.append({**p, "category_id": 0})

    # Create modified ground truth with all categories as 0
    gt_copy = copy.deepcopy(coco_gt.dataset)
    for ann in gt_copy["annotations"]:
        ann["category_id"] = 0
    gt_copy["categories"] = [{"id": 0, "name": "product", "supercategory": "product"}]

    det_coco_gt = COCO()
    det_coco_gt.dataset = gt_copy
    det_coco_gt.createIndex()

    coco_dt = det_coco_gt.loadRes(det_preds) if det_preds else det_coco_gt.loadRes([])
    coco_eval = COCOeval(det_coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = [0.5]
    if image_ids:
        coco_eval.params.imgIds = sorted(image_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # mAP@0.5


def evaluate_classification_map(coco_gt: COCO, predictions: list[dict], image_ids: set[int] | None = None) -> float:
    """Compute classification mAP@0.5 (per-category, standard COCO eval)."""
    if not predictions:
        return 0.0

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = [0.5]
    if image_ids:
        coco_eval.params.imgIds = sorted(image_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # mAP@0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--val-image-ids", help="JSON file with list of val image IDs (optional, evaluates all if omitted)")
    args = parser.parse_args()

    # Load predictions
    with open(args.predictions) as f:
        predictions = json.load(f)

    # Load ground truth
    coco_gt = COCO(args.ground_truth)

    # Filter to val images if specified
    val_ids = None
    if args.val_image_ids:
        with open(args.val_image_ids) as f:
            val_ids = set(json.load(f))
        predictions = [p for p in predictions if p["image_id"] in val_ids]

    # Evaluate (pass val_ids to filter ground truth too)
    det_map = evaluate_detection_map(coco_gt, predictions, image_ids=val_ids)
    cls_map = evaluate_classification_map(coco_gt, predictions, image_ids=val_ids)
    composite = compute_composite_score(det_map, cls_map)

    print(f"\n{'='*50}")
    print(f"Detection mAP@0.5:       {det_map:.4f}")
    print(f"Classification mAP@0.5:  {cls_map:.4f}")
    print(f"Composite Score:         {composite:.4f}")
    print(f"  (0.7 × {det_map:.4f} + 0.3 × {cls_map:.4f})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluate_local.py tests/test_evaluate.py
git commit -m "Add local evaluation script replicating competition scoring"
```

---

## Task 12: GCP VM Setup & Data Upload Scripts

**Files:**
- Create: `scripts/setup_gcp_vm.sh`
- Create: `scripts/upload_data.sh`

- [ ] **Step 1: Write VM setup script**

```bash
#!/usr/bin/env bash
# scripts/setup_gcp_vm.sh
# Provision a GPU VM on GCP for model training.
#
# Usage: bash scripts/setup_gcp_vm.sh [vm-name]

set -euo pipefail

VM_NAME="${1:-training-vm-1}"
PROJECT="ainm26osl-716"
ZONE="europe-north1-b"
MACHINE_TYPE="g2-standard-16"  # 16 vCPU, 64GB RAM, 1x L4 GPU
ACCELERATOR="type=nvidia-l4,count=1"
IMAGE_FAMILY="pytorch-2-6-cu124"
IMAGE_PROJECT="deeplearning-platform-release"
DISK_SIZE="100GB"

echo "Creating VM: $VM_NAME"
echo "  Project: $PROJECT"
echo "  Zone: $ZONE"
echo "  Machine: $MACHINE_TYPE"
echo "  GPU: L4"

gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="$ACCELERATOR" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$DISK_SIZE" \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"

echo ""
echo "VM created. Connect with:"
echo "  gcloud compute ssh $VM_NAME --project=$PROJECT --zone=$ZONE"
echo ""
echo "After connecting, run:"
echo "  pip install ultralytics==8.1.0 timm==0.9.12 onnx"
```

- [ ] **Step 2: Write data upload script**

```bash
#!/usr/bin/env bash
# scripts/upload_data.sh
# Upload training data and code to GCP VM.
#
# Usage: bash scripts/upload_data.sh [vm-name]

set -euo pipefail

VM_NAME="${1:-training-vm-1}"
PROJECT="ainm26osl-716"
ZONE="europe-north1-b"
REMOTE_DIR="~/norges-gruppen"

echo "Uploading to $VM_NAME:$REMOTE_DIR"

# Create remote directory
gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$ZONE" \
  --command="mkdir -p $REMOTE_DIR/data"

# Upload code
gcloud compute scp --recurse \
  training/ scripts/ submission/ dataset.yaml TASK.md \
  "$VM_NAME:$REMOTE_DIR/" \
  --project="$PROJECT" --zone="$ZONE"

# Upload data (large — use compressed zips)
gcloud compute scp \
  "Coco Dataset NM NGD.zip" "NM NGD Produktbilder.zip" \
  "$VM_NAME:$REMOTE_DIR/data/" \
  --project="$PROJECT" --zone="$ZONE"

echo ""
echo "Upload complete. SSH in and unzip:"
echo "  cd $REMOTE_DIR/data"
echo "  unzip 'Coco Dataset NM NGD.zip' -d coco_dataset"
echo "  unzip 'NM NGD Produktbilder.zip' -d product_images"
echo ""
echo "Then prepare dataset and train:"
echo "  cd $REMOTE_DIR"
echo "  python training/prepare_yolo_dataset.py"
echo "  python training/prepare_crops.py"
echo "  python training/train_yolo.py --epochs 150 --batch 4"
echo "  python training/train_classifier.py --epochs 50 --batch 64"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/setup_gcp_vm.sh scripts/upload_data.sh
git commit -m "Add GCP VM setup and data upload scripts"
```

---

## Task 13: Submission Packaging Script

**Files:**
- Create: `scripts/build_submission.sh`

- [ ] **Step 1: Write packaging script**

```bash
#!/usr/bin/env bash
# scripts/build_submission.sh
# Package submission.zip for competition upload.
#
# Expects model files in submission/ directory:
#   submission/run.py
#   submission/utils.py
#   submission/yolo_detector.onnx
#   submission/classifier.onnx
#   submission/reference_embeddings.npy
#
# Usage: bash scripts/build_submission.sh

set -euo pipefail

SUBMISSION_DIR="submission"
OUTPUT="submission.zip"

echo "Checking submission files..."

# Required files
for f in run.py utils.py yolo_detector.onnx; do
  if [ ! -f "$SUBMISSION_DIR/$f" ]; then
    echo "ERROR: Missing $SUBMISSION_DIR/$f"
    exit 1
  fi
done

# Optional but expected
for f in classifier.onnx reference_embeddings.npy class_mapping.json; do
  if [ ! -f "$SUBMISSION_DIR/$f" ]; then
    echo "WARNING: Missing $SUBMISSION_DIR/$f (will run in YOLO-only mode)"
  fi
done

# Check blocked imports in Python files
echo "Security check: scanning for blocked imports..."
BLOCKED="import os|import sys|import subprocess|import socket|import pickle|import yaml|import threading|import multiprocessing"
if grep -rE "$BLOCKED" "$SUBMISSION_DIR"/*.py 2>/dev/null; then
  echo "ERROR: Blocked imports found! Fix before submitting."
  exit 1
fi
echo "  No blocked imports found."

# Check total weight size
TOTAL_SIZE=$(find "$SUBMISSION_DIR" -name "*.onnx" -o -name "*.npy" -o -name "*.pt" | xargs stat -f%z 2>/dev/null | paste -sd+ | bc)
TOTAL_MB=$((TOTAL_SIZE / 1024 / 1024))
echo "  Total weight size: ${TOTAL_MB}MB / 420MB"
if [ "$TOTAL_MB" -gt 420 ]; then
  echo "ERROR: Weights exceed 420MB limit!"
  exit 1
fi

# Count files
PY_COUNT=$(find "$SUBMISSION_DIR" -name "*.py" | wc -l | tr -d ' ')
echo "  Python files: $PY_COUNT / 10"
if [ "$PY_COUNT" -gt 10 ]; then
  echo "ERROR: Too many Python files (max 10)!"
  exit 1
fi

WEIGHT_COUNT=$(find "$SUBMISSION_DIR" \( -name "*.onnx" -o -name "*.pt" -o -name "*.npy" -o -name "*.safetensors" \) | wc -l | tr -d ' ')
echo "  Weight files: $WEIGHT_COUNT / 3"
if [ "$WEIGHT_COUNT" -gt 3 ]; then
  echo "ERROR: Too many weight files (max 3)!"
  exit 1
fi

# Build zip
rm -f "$OUTPUT"
cd "$SUBMISSION_DIR"
zip -r "../$OUTPUT" . -x ".*" "__MACOSX/*" "__pycache__/*" "*.pyc"
cd ..

echo ""
echo "Created $OUTPUT"
unzip -l "$OUTPUT" | head -15
echo ""
echo "Verify run.py is at root (not in a subfolder)."
echo "Upload at: https://app.ainm.no/submit/norgesgruppen-data"
```

- [ ] **Step 2: Commit**

```bash
git add scripts/build_submission.sh
git commit -m "Add submission packaging script with validation checks"
```

---

## Task 14: Final Integration Verification

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Verify project structure**

Run: `find . -type f -not -path './data/*' -not -path './.git/*' -not -path './__pycache__/*' | sort`
Expected: All files from the project structure exist

- [ ] **Step 3: Verify no blocked imports in submission code**

Run: `python -c "import ast; [print(f) for f in ['submission/run.py','submission/utils.py'] for n in ast.walk(ast.parse(open(f).read())) if isinstance(n,(ast.Import,ast.ImportFrom)) and any(x in (getattr(n,'module','') or ','.join(a.name for a in getattr(n,'names',[]))) for x in ['os','sys','subprocess','pickle','yaml','threading'])]"`
Expected: No output (clean)

---

## Execution Order Summary

Tasks can be partially parallelized:

```
Task 1 (scaffolding)
  │
  ▼
Task 2 (data utils) ──────────────────────┐
  │                                        │
  ├──▶ Task 3 (YOLO dataset prep)         │
  │                                        │
  ├──▶ Task 4 (crop extraction)           │
  │                                        │
  ▼                                        ▼
Task 5 (submission utils) ──▶ Task 6 (run.py)
  │
  ├──▶ Task 7 (YOLO training script)
  │
  ├──▶ Task 8 (classifier training)
  │
  ├──▶ Task 9 (reference embeddings)
  │
  ├──▶ Task 10 (model export)
  │
  ▼
Task 11 (local eval)
  │
  ▼
Task 12 (GCP scripts) ──▶ Task 13 (submission packaging)
  │
  ▼
Task 14 (init files + integration)
```

After Task 14, the code is ready. Next steps:
1. Run `prepare_yolo_dataset.py` and `prepare_crops.py` locally
2. Provision GCP VM and upload
3. Train YOLO + classifier on GCP
4. Export models, build embeddings
5. Download weights, package submission
6. Evaluate locally, then submit
