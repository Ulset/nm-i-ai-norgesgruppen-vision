# Training Log

## 2026-03-20 — Initial Training (VM-1)

### YOLOv8x v1 (detection)
- **VM**: training-vm-1
- **Config**: yolov8x.pt, 1280px, batch=2, 150 epochs, ultralytics 8.1.0
- **Started**: 12:01 UTC, finished 13:04 UTC
- **Result**: mAP50 **0.732**, mAP50-95 0.471
- **Weights**: `runs/detect/train/weights/best.pt` on VM-1

### EfficientNet-B2 v1 (classifier — BROKEN)
- **VM**: training-vm-1
- **Started**: 13:04 UTC, finished 13:43 UTC
- **Result**: Severe overfitting — 0.49% validation accuracy
- **Root cause**: ImageFolder class_to_idx mismatch between train/val splits
- **Weights**: `runs/classify/best.pt` on VM-1 (unusable)

---

## 2026-03-20 — Ensemble Training Round (VM-1 + VM-2)

### YOLOv8l (ensemble model, VM-1)
- **Config**: yolov8l.pt, 640px, batch=8, 200 epochs, patience=30
- **Started**: 13:59 UTC, finished ~14:28 UTC
- **Epochs**: 158 (early stopped)
- **Result**: mAP50 **0.710**, mAP50-95 0.460
- **Weights**: `runs/detect/yolov8l_640/weights/best.pt` on VM-1
- **ONNX**: `runs/detect/yolov8l_640/weights/best.onnx` (168MB)

### YOLOv8x v2 retrain (VM-1, sequential after v8l)
- **Config**: yolov8x.pt, 1280px, batch=2, 200 epochs, patience=30
- **Started**: 14:28 UTC
- **Epochs**: 31 (early stopped)
- **Result**: mAP50 **0.709**, mAP50-95 0.462
- **Weights**: `runs/detect/yolov8x_shelf_v2/weights/best.pt` on VM-1

### YOLOv8x v2 retrain (VM-2, dedicated)
- **Config**: yolov8x.pt, 1280px, batch=?, 200 epochs, patience=30
- **Started**: ~same time, finished 15:11 UTC
- **Epochs**: 53 (early stopped)
- **Result**: mAP50 **0.731**, mAP50-95 0.477
- **Weights**: `runs/detect/yolov8x_v2/weights/best.pt` on VM-2
- **ONNX**: `runs/detect/yolov8x_v2/weights/best.onnx` (262MB)

---

## 2026-03-20 — Parallel Assault Training Round

All models trained with ultralytics 8.4.24, PyTorch 2.10.0.
Augmentation: mosaic=1.0, mixup=0.2, copy_paste=0.3, close_mosaic=30.

### YOLO11x @ 1280px (VM: training-vm-1)
- **Config**: yolo11x.pt, 1280px, batch=2, 300 epochs, patience=50
- **Started**: ~17:51 UTC
- **Latest check**: epoch 102/300, mAP50 **0.727** (best seen: 0.738)
- **Speed**: ~2.5 it/s, 9.9GB VRAM
- **Weights will be at**: `runs/detect/yolo11x_1280/weights/best.pt` on VM-1

### YOLOv9e @ 1280px (VM: training-vm-2)
- **Config**: yolov9e.pt, 1280px, batch=2, 300 epochs, patience=50
- **Started**: ~17:51 UTC
- **Latest check**: epoch 88/300, mAP50 **0.736**
- **Speed**: ~1.9 it/s, 11.4GB VRAM
- **Weights will be at**: `runs/detect/yolov9e_1280/weights/best.pt` on VM-2

### YOLO11x @ 640px (VM: training-yolo11x-640)
- **Config**: yolo11x.pt, 640px, batch=8, 300 epochs, patience=50
- **Started**: ~17:51 UTC
- **Latest check**: epoch 173/300, mAP50 **0.724** — appears to have early-stopped or plateaued
- **Speed**: ~2.5 it/s, 9.7GB VRAM
- **Weights will be at**: `runs/detect/yolo11x_640/weights/best.pt` on VM

### EfficientNet-B2 v2 — FIXED (VM: training-classifier-fix)
- **Config**: efficientnet_b2, focal loss (gamma=2.0), label smoothing 0.1, 100 epochs, patience=20
- **Started**: ~18:00 UTC (restarted after class mapping fix)
- **Latest check**: epoch 50/100, val_acc **0.903** (was 0.004 before fix!)
- **Bug fix**: ImageFolder class_to_idx mismatch between train/val resolved in commit ac2d7d7
- **Weights will be at**: `runs/classifier/effnet_b2_fixed_v2/best.pt` on VM

### DINOv2 Embeddings (VM: training-dinov2-embeds)
- **Config**: dinov2_vits14 (22M params, 384-dim)
- **Status**: Reference embeddings saved (357, 384) float16 at `runs/dinov2/dinov2_reference_embeddings.npy`
- **ONNX export FAILED**: Device mismatch error (mask_token on CPU, input on CUDA)
- **Needs**: Fix ONNX export by moving all model params to same device, or export on CPU

---

## Round 2: Full-Data + Augmented Training (started ~20:30 UTC)

### YOLO11x full-data @ 1280px (VM: training-vm-1) — DONE
- **Dataset**: ALL 248 images, no val holdout
- **Completed**: 300/300 epochs, mAP50 **0.987*** (*val overlaps train)
- **ONNX exported**: 109.7MB, on VM-6 + local (/tmp/yolo_11x_full.onnx)
- **Saved to**: `runs/detect/yolo11x_full/` on VM-1

### YOLOv9e full-data @ 1280px (VM: training-vm-2)
- **Dataset**: ALL 248 images, no val holdout
- **Latest**: epoch 44/300, mAP50 **0.868**

### YOLO11x synthetic @ 1280px (VM: training-classifier-fix) — DONE
- **Dataset**: 248 real + 95 synthetic images (copy-paste from product ref images)
- **Started**: ~21:10 UTC, **early-stopped at epoch 157**
- **Final**: mAP50 **0.724**
- **ONNX exported**: 109.7MB, copied to VM-6 for benchmarking
- **Saved to**: `runs/detect/yolo11x_synthetic2/` on VM-4

### YOLO11x pseudo-labeled @ 1280px (VM: training-dinov2-embeds) — DONE
- **Dataset**: Original + 2,489 pseudo-labels (conf>0.8)
- **Early-stopped**: epoch 87/300, final mAP50 **0.721**
- **ONNX exported**: 109.7MB, copied to VM-6 for benchmarking
- **Saved to**: `runs/detect/yolo11x_pseudo2/` on VM-5

### Benchmark Results (VM-6, round 1 models)
Scored against 49-image val set with full pipeline (TTA + tiling + WBF + classifier):

| Config | Det mAP50 | Notes |
|--------|----------|-------|
| YOLO11x r1 solo | **0.734** | Best single model |
| YOLOv8l old solo | **0.722** | Old baseline |
| YOLOv9e r1 solo | **0.716** | |
| Ensemble (11x+v9e) | **0.708** | Lower than solo — WBF needs tuning |

Classification mAP scoring failed (numpy compat issue on VM-6). Detection-only scores above.
Key finding: ensemble HURTS with current WBF settings. Single YOLO11x is best.

### Model Soup (VM: training-yolo11x-640) — DONE
- Averaged 4 checkpoints (epoch80/90/100/best) from round 1 YOLO11x
- Saved soup.pt (109.4MB), needs ONNX export

### EfficientNet-B2 v2 — COMPLETED
- **Final**: val_acc **0.9047**
- ONNX exported (31.2MB), downloaded locally

---

## Best Models Summary (as of ~21:15 UTC)

| Model | mAP50 / Acc | Location | Notes |
|-------|------------|----------|-------|
| YOLO11x full-data | 0.987* | VM-1 `runs/detect/yolo11x_full/` | *inflated val, DONE, ONNX on VM-6 |
| YOLOv9e full-data | 0.988* | VM-2 `runs/detect/yolov9e_full2/` | *inflated val, DONE, ONNX on VM-6 |
| YOLO11x round1 | 0.738 | VM-1 `runs/detect/yolo11x_12802/` | ONNX exported (109.7MB) |
| YOLOv9e round1 | 0.735 | VM-2 `runs/detect/yolov9e_12802/` | ONNX exported (110.6MB) |
| YOLO11x pseudo | 0.696 | VM-5 `runs/detect/yolo11x_pseudo/` | TRAINING |
| YOLO11x synthetic | 0.449 | VM-4 `runs/detect/yolo11x_synthetic/` | TRAINING (early) |
| YOLO11x soup | ? | VM-3 `runs/detect/yolo11x_soup/` | Needs ONNX export |
| YOLO11x (640px) | 0.724 | VM-3 `runs/detect/yolo11x_640/` | Done |
| YOLOv8x v1 | 0.732 | VM-1 `runs/detect/train/` | Old baseline |
| EfficientNet-B2 v2 | 90.5% acc | VM-4 `runs/classifier/effnet_b2_fixed_v2/` | DONE, exported |
| EfficientNet-B2 v1 | 0.49% acc | VM-1 `runs/classify/` | BROKEN, do not use |
