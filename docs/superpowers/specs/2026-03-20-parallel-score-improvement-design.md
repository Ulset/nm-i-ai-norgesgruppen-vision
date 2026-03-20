# Parallel Score Improvement — Design Spec

**Date**: 2026-03-20
**Goal**: Improve competition score from 0.8346 to 0.92+ (rank #89 → top 10)
**Deadline**: Sunday 2026-03-23
**Score formula**: 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5

## Current State

- Best score: 0.8346 (#89/195). Top team: 0.9200.
- Detection: YOLOv8x (mAP50 ~0.73 local, higher in submission with TTA/tiling)
- Classification: EfficientNet-B2 is broken (0.49% val accuracy — class mapping bug). Reference embeddings carry this component.
- Submission: ensemble pipeline with WBF, tiled inference, horizontal-flip TTA
- Sandbox: Python 3.11, L4 GPU, onnxruntime only, 420MB/3 files, 300s timeout, max 10 Python files
- **Baseline preserved**: Archive current submission.zip as `submission_baseline_0.8346.zip` before any changes.

## Approach: Parallel Training on 6 GCP VMs

All compute on GCP. Nothing runs locally on Mac.

### VM-1: YOLO11x Detection @ 1280px

**Rationale**: YOLO11x is the latest YOLO architecture with improved C3k2 blocks and attention. Consistently outperforms YOLOv8x on COCO benchmarks.

**Ultralytics version**: `ultralytics>=8.3.0` (required for YOLO11 support)

```
model: yolo11x.pt
imgsz: 1280
batch: 2
epochs: 300
patience: 50
mosaic: 1.0
mixup: 0.2
copy_paste: 0.3
close_mosaic: 30
hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4
fliplr: 0.5, flipud: 0.0
optimizer: SGD, lr0: 0.01, weight_decay: 0.0005
warmup_epochs: 3
```

**GPU**: Use `g2-standard-48` with A100 or `a2-highgpu-1g` for faster training at 1280px. L4 may be too slow for 300 epochs at batch=2.

**Post-training**: Export to ONNX FP16. **Validate ONNX output shape matches `(1, 4+nc, N)` contract expected by run.py**. Run dummy inference to confirm.

### VM-2: YOLOv9e Detection @ 1280px

**Rationale**: YOLOv9 uses Programmable Gradient Information (PGI) and GELAN architecture — fundamentally different from YOLO11, providing ensemble diversity.

**Ultralytics version**: `ultralytics>=8.1.34` (YOLOv9e support)

Same hyperparameters as VM-1 but with `yolov9e.pt`. Same A100 GPU recommendation.

**Post-training**: Same ONNX output shape validation as VM-1.

### VM-3: YOLO11x Detection @ 640px

**Rationale**: Different input resolution captures different scale information. Faster inference allows more TTA passes within the 300s timeout.

**Ultralytics version**: `ultralytics>=8.3.0`

Same as VM-1 but: `imgsz: 640, batch: 8`. Can use L4 GPU (g2-standard-16) since 640px is much lighter.

### VM-4: Fix Classifier + Retrain EfficientNet-B2

**Rationale**: The 30% classification component is severely underperforming due to a class mapping bug. Fixing this is the single highest-ROI task.

Steps:
1. SSH into VM, inspect `data/crops/` directory structure vs COCO annotation category IDs
2. Compare ImageFolder class indices with category_id mapping in annotations
3. Fix `prepare_crops.py` to ensure directory names match category IDs
4. Re-run crop preparation
5. Retrain with improvements:
   - Focal loss (gamma=2.0) instead of weighted cross-entropy
   - Label smoothing 0.1
   - 100 epochs, patience 20
   - Same augmentation pipeline
6. Rebuild reference embeddings from fixed model
7. Bake new embeddings into `baked_data.py`

### VM-5: DINOv2 Reference Embeddings

**Rationale**: DINOv2 produces state-of-the-art visual features without fine-tuning. If it beats EfficientNet for crop → product matching, we swap it in.

**Integration path**: If DINOv2 wins, it replaces EfficientNet as the embedding source for reference matching. Classification falls back to YOLO category when reference match confidence is low. Reference embeddings regenerated at 384-dim (replacing 1408-dim). `classify_detections()` updated to use nearest-neighbor only (no softmax logits from DINOv2).

Steps:
1. Install torch + DINOv2 via torch.hub
2. Load `dinov2_vits14` (small, 22M params)
3. Extract 384-dim embeddings for all product reference images
4. Export DINOv2-small as ONNX FP16 (~44MB)
5. Test cosine similarity matching accuracy vs EfficientNet embeddings on held-out crops
6. If better: integrate into submission, update baked_data.py with 384-dim embeddings

### VM-6: Inference Optimization

**Rationale**: Threshold tuning and scoring logic improvements are free accuracy. Need a VM with the dataset + current models to run evaluate_local.py.

Changes to test:
- Lower CLASSIFIER_THRESHOLD: 0.7 → 0.5
- Lower REFERENCE_THRESHOLD: 0.8 → 0.6
- WBF_IOU_THR: test 0.45, 0.50, 0.55, 0.60
- CONFIDENCE_FLOOR: test 0.01, 0.03, 0.05, 0.10
- Multi-scale TTA: add 0.9x and 1.1x scaling
- Scoring: test `max(yolo_conf, clf_conf)` and weighted geometric mean vs current multiplicative
- Per-model WBF weights (weight by each model's mAP50)

## Code Updates Required

The following code changes are needed to support new model combinations:

1. **`submission/run.py`**: Update model loading to support configurable ONNX filenames and input sizes. Currently hardcodes `yolo_detector.onnx` (1280) and `yolo_l_detector.onnx` (640). Needs to handle any 2-model combination.
2. **`training/export_models.py`**: Add YOLO11x and YOLOv9e export paths. Add ONNX output shape validation step.
3. **`scripts/build_submission.sh`**: Update to accept configurable model filenames.
4. **Post-FP16 validation**: After every FP16 export, run local eval and compare mAP to FP32. If accuracy drops >0.5%, use mixed-precision (keep BN and final layers in FP32).

## Weight Budget

Max 3 weight files, 420MB total.

Best case (pick top 2 detectors + classifier):
| File | Size (FP16 est.) |
|------|-----------------|
| YOLO11x 1280 | ~114MB |
| YOLOv9e 1280 or YOLO11x 640 | ~115MB or ~114MB |
| Classifier (EfficientNet-B2 or DINOv2-small) | ~16MB or ~44MB |
| **Total** | **~245-273MB** |

If only one detector is significantly better, use one detector + classifier + DINOv2:
| File | Size (FP16 est.) |
|------|-----------------|
| Best YOLO | ~114MB |
| EfficientNet-B2 | ~16MB |
| DINOv2-small | ~44MB |
| **Total** | **~174MB** |

## Submission Strategy

Daily submission limit: 6/day (verified from portal screenshot showing 6/6 used).

- **Fri** (6 submissions): First submission = current baseline (safety). Rest = inference tuning from VM-6.
- **Sat** (6 submissions): New YOLO models should be done. Build ensemble, test combinations.
- **Sun** (6 submissions): Final optimization. Best ensemble + best classifier + tuned thresholds.

**Rollback**: Always keep `submission_baseline_0.8346.zip` as known-good fallback. Never use all 6 daily submissions on untested changes — always save 1-2 for the proven best.

## Success Criteria

- Score >= 0.90 (top 20)
- Stretch goal: >= 0.92 (competitive with #1)

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| YOLO11x/v9e don't improve over v8x on this data | We still have v8x+v8l ensemble as baseline |
| ONNX output format differs between YOLO versions | Validate output shape `(1, 4+nc, N)` after every export; fix run.py parsing if needed |
| Classifier fix doesn't resolve mapping issue | DINOv2 embeddings as fallback |
| Weight budget too tight for 2 detectors + classifier | FP16 export verified to fit; can drop to 1 detector |
| Training too slow on L4 at 1280px | Use A100 VMs for high-res training |
| Training doesn't finish by Saturday | patience=50 + 300 epochs gives room; can use best checkpoint |
| Sandbox timeout with 3-model ensemble | Profile inference time; drop slowest model if needed |
| FP16 accuracy degradation | Post-conversion validation; fall back to mixed-precision if needed |
| Python file count exceeds 10 | Currently at 3 files; track carefully if adding DINOv2 preprocessing |
