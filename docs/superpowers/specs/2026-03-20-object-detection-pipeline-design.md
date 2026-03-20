# NorgesGruppen Object Detection Pipeline — Design Spec

## Overview

Two-stage object detection and classification pipeline for the NM i AI 2026 NorgesGruppen challenge. Detects and identifies grocery products on store shelves.

**Score formula**: `0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5`

**Target score**: 0.75-0.85

## Architecture

Two-stage pipeline:

1. **Stage 1 — YOLOv8x detector**: Fine-tuned on shelf images with tiled inference for high-res images. Provides bounding boxes + baseline category predictions.
2. **Stage 2 — EfficientNet-B2 classifier**: Classifies cropped detections using softmax + nearest-neighbor fallback against product reference embeddings. Solves the long-tail category problem.

Three weight files submitted:
- YOLOv8x FP16 `.onnx` (~130MB) — ONNX preferred over `.pt` to avoid sandbox `pickle` import blocking risk
- EfficientNet-B2 FP16 `.onnx` (~30MB)
- Reference embeddings `.npy` (~1MB)

Total: ~161MB of 420MB budget.

### Sandbox `.pt` Risk

The sandbox blocks `import pickle`. While `torch.load()` may still work (PyTorch handles pickle internally via C extensions), this is unverified. **Primary plan: export both models to ONNX** (opset 17, FP16, CUDAExecutionProvider). Fallback: test `.pt` loading with an early sandbox submission. ONNX is also faster on L4 GPUs.

## Data Analysis

| Metric | Value |
|---|---|
| Training images | 248 (4 store sections: Egg, Frokost, Knekkebrod, Varmedrikker) |
| Annotations | 22,731 |
| Categories | 356 in training data (IDs 0-355, with 355 = unknown_product) |
| Avg products/image | 92 (max 235) |
| Image dimensions | 481-5712px wide |
| Categories with <5 annotations | 74 |
| Categories with <10 annotations | 110 |
| Product reference images | 344 folders on disk (327 from metadata + 17 CUSTOM_XXX folders), ~1,599 total images |
| Reference-to-category mapping | 326 products match by exact name to 321 unique category IDs |

### Category ID Discrepancy

TASK.md states 357 categories (IDs 0-356, with 356 = unknown_product). The actual training data has 356 categories (IDs 0-355, with 355 = unknown_product). **We train with `nc=357` to be safe** — if the test set uses category_id 356, our model can predict it. Category 356 will have no training examples, but the reference embedding fallback can still handle it.

### Product-to-Category Mapping

The training annotations do NOT contain `product_code` or `product_name` fields (despite TASK.md suggesting they do). Product reference images are linked to categories via **exact name matching** between `metadata.json` product names and `annotations.json` category names. Edge cases:
- Category 300 has an empty string name — 5 product references map to it. Exclude from reference matching.
- 1 product (`FRIELE FROKOST PRESSKANNE 250G`) has images but no matching category. Ignore.
- 17 `CUSTOM_XXX` folders contain only `main.jpg` — investigate during implementation, include if useful.

**Key challenge**: Heavily long-tailed distribution. 74 categories have fewer than 5 training samples — a single YOLO model cannot learn these reliably. Product reference images are the key to solving classification on the tail.

## Project Structure

```
norges-gruppen-computer-vision/
├── data/                          # Training data (gitignored)
│   ├── coco_dataset/train/        # 248 shelf images + annotations.json
│   └── product_images/            # 327 product reference folders
├── training/
│   ├── train_yolo.py              # YOLOv8x fine-tuning script
│   ├── prepare_crops.py           # Extract product crops from training data
│   ├── train_classifier.py        # EfficientNet-B2 classifier training
│   ├── build_reference_embeddings.py  # Pre-compute product reference embeddings
│   └── export_models.py           # FP16 export for submission
├── submission/
│   ├── run.py                     # Entry point (inference pipeline)
│   └── utils.py                   # Tiled inference, WBF, embedding matching
├── scripts/
│   ├── setup_gcp_vm.sh            # Provision GPU VM on GCP
│   ├── upload_data.sh             # Upload training data to VM
│   ├── build_submission.sh        # Package zip for upload
│   └── evaluate_local.py          # Local mAP evaluation with train/val split
├── TASK.md
├── CLAUDE.md
└── .gitignore
```

## Training Pipeline

All training runs on GCP GPU VMs in project `ainm26osl-716`.

### Train/Val Split

80/20 stratified split on images (not annotations). 198 train images, 50 val images. Stratified by store section to ensure each section is represented in validation.

### Stage 1 — YOLOv8x Fine-Tuning

| Parameter | Value |
|---|---|
| Base model | YOLOv8x (COCO pretrained) |
| Classes | nc=357 (0-355 products + 356 unknown_product safety) |
| Image size | imgsz=1280 |
| Epochs | 100-200 with early stopping (patience=20) |
| Optimizer | SGD with cosine LR (ultralytics default) |
| Batch size | As large as VRAM allows (likely 4-8 on A100) |
| Augmentation | Mosaic, mixup, copy-paste (cranked up), HSV shifts, flips |

**Pin `ultralytics==8.1.0`** — sandbox has this exact version.

Training with `nc=357` gives detection AND baseline classification in one forward pass. Classification will be weak on rare categories but strong on the ~142 categories with 50+ annotations. Category 356 has no training data but exists as a safety net for the test set.

### Stage 2 — EfficientNet-B2 Crop Classifier

**Training data** (two sources combined):

1. **Crops from training images**: Extract each annotated bbox from training images, resize to 224×224. ~22,700 labeled crops across 356 categories.
2. **Product reference images**: 326 matched products × ~4.6 angles avg ≈ 1,500 images. Augmented heavily to bridge domain gap (random backgrounds, brightness/contrast, perspective transforms, Gaussian noise). Excludes empty-name category 300 and unmatched products.

| Parameter | Value |
|---|---|
| Base model | EfficientNet-B2 (ImageNet pretrained, timm==0.9.12) |
| Classes | 357 |
| Image size | 224×224 |
| Epochs | 50 with early stopping |
| Loss | Cross-entropy with inverse-frequency class weights |
| Augmentation | Random crop, color jitter, perspective, Gaussian noise, Mixup/CutMix |
| Optimizer | AdamW with cosine LR |

**Pin `timm==0.9.12`** — sandbox has this exact version.

### Stage 3 — Reference Embeddings

After training the classifier:

1. Remove classification head, extract penultimate layer (1408-dim for EfficientNet-B2)
2. Embed all product reference images (prioritize main + front angles — most similar to shelf appearance)
3. L2-normalize embeddings
4. Average per-product to get one embedding per category
5. Save as `.npy` (357 × 1408 × float16 ≈ ~1MB, zero vector for categories without reference images)

## Inference Pipeline

Executed via `python run.py --input /data/images --output /output/predictions.json` in sandbox (L4 GPU, 8GB RAM, 300s timeout).

### Flow

```
Input images
    │
    ▼
┌─────────────────────┐
│  Tiled Inference     │  Per image:
│  (YOLOv8x @ 1280)   │  - image <= 1280px: run YOLO once
│                      │  - image > 1280px: slice into 1280×1280 tiles
│                      │    with 20% overlap (~256px)
│                      │  - Always also run on full image resized to 1280
│                      │  - Map tile detections back to full image coords
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  WBF Merge           │  Weighted Box Fusion (ensemble_boxes.ensemble_boxes_wbf)
│                      │  - Merge overlapping detections across tiles
│                      │  - iou_thr=0.55 (slightly above eval threshold to
│                      │    avoid merging adjacent products on dense shelves)
│                      │  - Preserves confidence scores + YOLO category_ids
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Crop & Classify     │  Per detection:
│  (EfficientNet-B2)   │  - Crop from original image, resize to 224×224
│                      │  - Forward through classifier
│                      │  - Compare embedding to reference embeddings
│                      │  - Decision logic picks final category_id
└─────────┬───────────┘
          │
          ▼
    predictions.json
```

### Classification Decision Logic

Three signals per detection:

1. **YOLO category + confidence** (from Stage 1)
2. **EfficientNet softmax top-1** (from Stage 2 classification head)
3. **Reference cosine similarity** (from Stage 2 embedding vs reference embeddings)

Decision:
- EfficientNet top-1 confidence > 0.7 → use EfficientNet category
- Else reference cosine similarity > 0.8 → use nearest reference category
- Else → use YOLO category
- If all three signals have very low confidence (below 0.15) → assign `unknown_product` (category 355)

Thresholds tuned on validation set.

### Confidence Score Computation

The output `score` field (used for mAP precision-recall ranking) is computed as:
- `score = yolo_confidence × classification_confidence`
- Where `classification_confidence` is the confidence of whichever signal won (EfficientNet softmax, reference cosine sim, or YOLO class confidence)
- This down-weights detections where we're unsure about both location and identity

### Memory Management

Both YOLO (~130MB) and EfficientNet (~30MB) ONNX models fit in L4's 24GB VRAM simultaneously. However, to stay within 8GB system RAM:
- Process one image at a time
- For crop classification, batch crops in groups of 32 (not all ~92 at once)
- Use `numpy` arrays for crops, not PIL Image objects (lower memory overhead)
- If OOM occurs, fall back to YOLO-only predictions (still scores up to 0.70)

### Error Handling

If the classifier or reference embeddings fail to load:
- Log the error and fall back to YOLO-only mode
- YOLO-only predictions still score the full 70% detection component
- This is the graceful degradation path

### Time Budget

| Step | Per image | Total (~50 imgs) |
|---|---|---|
| Tiling + YOLO inference | 2-4s | 100-200s |
| WBF merge | ~10ms | ~0.5s |
| Crop + classify (~92 crops/img) | 0.5-1s | 25-50s |
| I/O + overhead | — | ~10s |
| **Total** | | **~140-260s** |

40-160s headroom within 300s timeout. If tight, skip classification on low-confidence detections (score < 0.1).

## GCP Training Setup

| Resource | Value |
|---|---|
| GCP Project | `ainm26osl-716` |
| gcloud profile | `nmiai-unlimited` |
| VM type | `g2-standard-16` (L4) or `a2-highgpu-1g` (A100) |
| Disk | 100GB SSD |
| Region | `europe-north1` |
| Image | Deep Learning VM, PyTorch 2.6 + CUDA 12.4 |

### Experiment Plan

| # | Experiment | Purpose | VM |
|---|---|---|---|
| 1 | YOLOv8x, imgsz=1280, 150 epochs | Main detection model | VM-1 |
| 2 | YOLOv8x, imgsz=1280, copy-paste cranked | Test copy-paste augmentation | VM-1 (sequential) |
| 3 | YOLOv8l, imgsz=640 | Backup for potential ensemble | VM-2 |
| 4 | EfficientNet-B2, crops + reference imgs | Main classifier | VM-2 |
| 5 | EfficientNet-B2, heavier ref augmentation | Classifier variant | VM-2 (sequential) |

Experiments on separate VMs run in parallel.

## Submission Workflow

1. Export best YOLO checkpoint as ONNX FP16 (opset 17, via ultralytics export)
2. Export best EfficientNet-B2 as ONNX FP16 (opset 17)
3. Save reference embeddings as `.npy`
4. Validate locally with `evaluate_local.py` (replicates competition scoring)
5. Package: `run.py` + `utils.py` + 2 ONNX files + 1 `.npy` → `submission.zip`
6. Upload to competition submit page
7. Max 3 submissions/day — only submit after local validation

## Sandbox Constraints Checklist

- [x] `run.py` at zip root
- [x] Max 3 weight files: 2× `.onnx` + `.npy`
- [x] Total weights < 420MB (~161MB)
- [x] No blocked imports (use `pathlib` not `os`, `json` not `yaml`)
- [x] No `pip install` — all packages pre-installed
- [x] GPU auto-detected via `torch.cuda.is_available()`
- [x] `torch.no_grad()` during all inference
- [x] Process images individually (8GB RAM limit)
- [x] Pin `ultralytics==8.1.0`, `timm==0.9.12` for training

## Local Evaluation

`evaluate_local.py` replicates the competition scoring on our val split:

1. Run `run.py` on val images to generate `predictions.json`
2. Use `pycocotools.cocoeval.COCOeval` to compute:
   - **Detection mAP@0.5**: All category_ids mapped to a single class (category-agnostic)
   - **Classification mAP@0.5**: Standard per-category COCO eval
3. Compute composite: `0.7 × det_mAP + 0.3 × cls_mAP`
4. Report per-category AP breakdown to identify weak spots

This prevents wasting the 3 daily submission slots on untested changes.

## Future Optimization Path

If time and submission slots allow:

1. **Add ensemble**: Train YOLOv8l @ 640 as second detector, WBF-merge with YOLOv8x @ 1280. Two complementary scales.
2. **Test-Time Augmentation (TTA)**: Horizontal flip + multi-scale on YOLO. Improves detection at cost of 2-3x inference time.
3. **Tune thresholds**: Grid search classification decision thresholds on validation set.
4. **ONNX for YOLO too**: If ultralytics `.pt` loading is slow, export YOLO to ONNX with CUDAExecutionProvider for faster inference.
