# Error Analysis Tool — Design Spec

**Date**: 2026-03-20
**Goal**: Build `scripts/analyze_errors.py` to identify where the submission pipeline loses the most score, with targeted visualizations of the worst failures.

## Context

- Competition score: 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5
- Current best: 0.8346 (#89/195), target: 0.92+
- 50 validation images, 357 product categories
- Deadline: Sunday 2026-03-23, 18 submissions remaining
- Local hardware: M1 Max 32GB RAM (CPU inference via onnxruntime)

## Architecture

```
Val images (50)  →  Submission pipeline (run.py)  →  predictions per image
                                                          ↓
Ground truth (annotations.json)  →  Matcher (IoU ≥ 0.5)  →  Matched pairs
                                                          ↓
                                              ┌───────────┴───────────┐
                                              ↓                       ↓
                                    Metrics engine            Visualization
                                    (JSON report)          (worst-10 PNGs)
```

The script imports `process_image()` directly from `submission/run.py` to test the exact pipeline that gets submitted. No reimplementation.

## Data Flow: Val Image ID → File Path

The COCO annotations file contains `images` entries with `file_name` fields. The script loads annotations, filters to val image IDs from `val_image_ids.json`, and builds a `{image_id: Path}` mapping to resolve file paths for `process_image()`.

## Detection-GT Matching

Predictions are sorted by confidence (highest first), then greedily matched to ground truth using IoU ≥ 0.5. Each GT box can be matched at most once. This mirrors the COCO evaluation protocol (confidence-ranked matching).

Matching is **class-agnostic for detection** (any prediction overlapping a GT box counts as detected) and **class-aware for classification** (correct only if category matches). The four buckets:

| Bucket | Meaning |
|--------|---------|
| TP-correct | IoU ≥ 0.5 AND category matches |
| TP-misclassified | IoU ≥ 0.5 BUT wrong category |
| FP | No ground truth match (spurious detection) |
| FN | Ground truth box with no matching prediction |

## Analysis Modules

### 1. Per-Image Breakdown
For each of 50 val images: TP/FP/FN counts, detection recall/precision, classification accuracy. Sorted worst-first by a proxy score: `image_score = 0.7 × (TP / (TP + FP + FN)) + 0.3 × (TP_correct / max(TP, 1))`. This is an F1-like proxy — not identical to mAP, but directionally correct for identifying problem images.

### 2. Per-Category Breakdown
For each of 357 categories: GT instance count, detection rate, classification accuracy, most common misclassification target. Sorted by `damage = n_fn + n_misclassified` (raw count of lost detections + wrong classifications). Raw count is preferred over rate because a category with 200 instances and 80% recall loses more score than one with 5 instances and 0% recall.

### 3. Pipeline Stage Attribution
Runs the scorer three ways to isolate where score comes from:
- **YOLO-only** (no classifier, use YOLO category_id directly)
- **YOLO + classifier** (full pipeline as submitted)
- **Per-model**: YOLO-x alone vs YOLO-l alone vs ensemble

This requires running inference multiple times with different model configurations. To keep runtime manageable, the per-model runs reuse cached YOLO outputs where possible.

### 4. Threshold Sensitivity
Sweeps two key parameters:
- Confidence floor: 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5
- WBF IoU threshold: 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8

**How this works with module-level globals**: The initial inference pass runs with `CONFIDENCE_FLOOR=0.001` (monkey-patched via `run.CONFIDENCE_FLOOR = 0.001` before inference) to capture nearly all raw YOLO outputs. The raw per-image outputs (boxes, scores, labels) are cached. The threshold sweep then refilters these cached outputs at each confidence floor and re-runs WBF (also with monkey-patched `run.WBF_IOU_THR`) and scoring. No re-inference needed.

Reports the optimal combo and score delta vs current settings.

### 5. Classification Signal Analysis
For each TP-misclassified detection, logs:
- YOLO's category guess + confidence
- Classifier's top-3 predictions + confidences
- Reference embedding best-match similarity
- Which signal "won" in the priority logic

This identifies whether the classifier, reference embeddings, or YOLO fallback is the primary source of misclassification.

**Note on DINOv2**: The baked reference embeddings are 1408-dim (EfficientNet). DINOv2 produces 384-dim embeddings, so the DINOv2 classification path cannot use the current baked references. The analysis script will only analyze the EfficientNet classifier + reference embedding path. DINOv2 analysis is deferred until DINOv2-specific reference embeddings are generated.

## Visualizations

All output goes to `analysis_output/` (gitignored).

### Worst 10 Images (`analysis_output/worst_images/`)
Shelf images are downscaled to max 2000px wide for visualization to keep file sizes manageable. Box overlays:
- Green solid = TP-correct
- Orange solid = TP-misclassified (label: "pred→GT")
- Red solid = FP (spurious)
- Blue dashed = FN (missed GT)

Filename: `{rank:02d}_{filename}_recall{recall:.2f}.png`

### Worst 10 Categories (`analysis_output/worst_categories/`)
Grid of misclassified crops for each category showing predicted vs ground truth labels.

### Summary Charts (`analysis_output/`)
- `score_breakdown.png` — bar chart comparing YOLO-only, YOLO+clf, and ensemble variant scores
- `threshold_sweep.png` — heatmap of composite score across confidence × WBF IoU grid
- `category_damage.png` — horizontal bar chart of top 20 categories by score lost

### Console Output
Concise text summary with key numbers. Full structured data saved to `analysis_output/report.json`.

## Implementation Notes

### Import Strategy
The script adds `submission/` to `sys.path` and imports from `run` and `utils` modules directly. This ensures we test the exact code that ships.

**Key detail**: `submission/baked_data.py` is 1.3MB of base64-encoded embeddings. Importing it takes a few seconds and ~50MB memory for decoding. This happens once at startup — acceptable.

**Module-level globals**: `run.py` defines `CONFIDENCE_FLOOR`, `WBF_IOU_THR`, etc. as module-level globals. The analysis script monkey-patches these before calling inference functions (e.g., `import run; run.CONFIDENCE_FLOOR = 0.001`). This is intentional — it lets us control thresholds without modifying the submission code.

### Runtime Estimate
M1 Max CPU: ~2-5s per YOLO 1280px inference call. Each image has ~6-10 tiles + full-image pass, ×2 for TTA, ×2 models ≈ 30-40 calls/image.
- Full ensemble pass: 50 images × ~35 calls × ~3.5s ≈ 100 min
- Per-model attribution (2 extra passes): ~50 min each
- Threshold sweep + scoring: ~5 min (refilters cached data, no re-inference)
- Visualization: ~5 min
- **Total: ~3-4 hours** with `--skip-per-model` bringing it down to ~2 hours

**Caching**: Pre-WBF tile-level outputs (per-model lists of boxes, scores, labels from `run_yolo_tta`) are saved to `analysis_output/cache/` as `.npz` files per image. This granularity is required because the threshold sweep must re-run WBF with different `iou_thr` and `skip_box_thr` values. If the script crashes or is interrupted, it resumes from cached outputs.

### CLI Interface
```bash
python scripts/analyze_errors.py \
  --submission-dir submission/ \
  --images-dir data/coco_dataset/train/images/ \
  --ground-truth data/coco_dataset/train/annotations.json \
  --val-image-ids data/yolo_dataset/val_image_ids.json \
  --output-dir analysis_output/
```

Optional flags:
- `--skip-visualization` — metrics only, no image generation
- `--skip-per-model` — skip individual model runs (saves ~1 hour)
- `--models yolo_detector.onnx` — test a specific model subset

### Scoring
The script imports `evaluate_detection_map`, `evaluate_classification_map`, and `compute_composite_score` from `scripts.evaluate_local` to ensure score consistency with the existing evaluation tool. No reimplementation of COCO mAP.

### Dependencies
- `onnxruntime` (already installed)
- `matplotlib` (for visualizations)
- `pycocotools` (already used by evaluate_local.py)
- `ensemble_boxes` (already used by submission)
- `PIL/Pillow` (already used)

No new dependencies required beyond matplotlib.

## Files Changed

| File | Change |
|------|--------|
| `scripts/analyze_errors.py` | New — main analysis script |
| `.gitignore` | Add `analysis_output/` |

## Output Structure

```
analysis_output/
├── cache/                   # Cached per-image YOLO outputs (.npz)
│   ├── img_00042.npz
│   └── ...
├── report.json              # Full structured results
├── score_breakdown.png      # Pipeline stage comparison
├── threshold_sweep.png      # Parameter sensitivity heatmap
├── category_damage.png      # Top 20 worst categories
├── worst_images/
│   ├── 01_img_00042_recall0.61.png
│   └── ...
└── worst_categories/
    ├── 01_cat_123_acc0.20.png
    └── ...
```
