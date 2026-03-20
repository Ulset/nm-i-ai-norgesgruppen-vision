# Improvement Plan V2: Data & Model Optimization (2026-03-20 evening)

## Context
- Training round 1 complete: YOLO11x (0.738 mAP50), YOLOv9e (0.735), classifier (90.3% acc)
- All 6 VMs now available for new work
- Key insight: 110 classes have <10 examples, 41 have just 1 — massive long-tail problem
- Dataset: 248 images, 22,731 annotations, 356 categories, ~92 ann/image

## Round 2 Actions (all in parallel)

### 1. Export Current Models to ONNX (VM-1 + VM-2)
- Must do first — need these for tomorrow's submissions
- YOLO11x best.pt → ONNX FP16 (at runs/detect/yolo11x_12802/)
- YOLOv9e best.pt → ONNX FP16 (at runs/detect/yolov9e_12802/)
- Also export classifier + rebuild baked_data.py

### 2. Model Soup — Average Checkpoints (any VM, quick)
- Average weights from multiple checkpoints (epoch 80, 90, 100)
- Also try averaging YOLO11x + YOLOv9e if architectures allow
- Expected: +0.5-1% mAP for free

### 3. Train on 100% Data (VM-1 + VM-2)
- Remove val holdout, train on all 248 images
- Use same hyperparameters that worked (already validated mAP ~0.738)
- YOLO11x@1280 on VM-1, YOLOv9e@1280 on VM-2

### 4. Copy-Paste Synthetic Data (VM-3 or VM-4)
- Cut products from 1,599 reference images
- Paste onto shelf background regions from training images
- Focus on the 74 classes with <5 examples
- Generate 50-100 synthetic images
- Retrain with augmented dataset

### 5. Pseudo-Labeling (VM-5)
- Run best YOLO model on training images
- Take predictions with confidence >0.8 as additional annotations
- Merge with ground truth
- Retrain on expanded dataset

### 6. Stratified Train/Val Split (VM-6)
- Ensure every class appears in both train and val
- Better training signal for rare classes
- Quick to implement, retrain with proper split

## VM Assignments
| VM | Task |
|----|------|
| VM-1 (training-vm-1) | Export ONNX → then train YOLO11x on 100% data |
| VM-2 (training-vm-2) | Export ONNX → then train YOLOv9e on 100% data |
| VM-3 (training-yolo11x-640) | Model soup from checkpoints |
| VM-4 (training-classifier-fix) | Generate synthetic copy-paste data → retrain YOLO |
| VM-5 (training-dinov2-embeds) | Pseudo-labeling pipeline |
| VM-6 (training-inference-tune) | Export classifier + rebuild embeddings + baked_data |

## Status (as of ~20:45 UTC)

### Completed
- ONNX exports DONE: YOLO11x (109.7MB), YOLOv9e (110.6MB), classifier (31.2MB) — all downloaded to local submission/
- Classifier reference embeddings rebuilt (321/357 categories embedded)
- Model soup: weights averaged from 4 checkpoints (109.4MB) — validation failed due to checkpoint format but weights are saved
- Pseudo-labeling DONE: found **2,489 new labels** across 107 images (11,983 rejected for overlap)
- Synthetic data DONE: **95 images** generated for 95 underrepresented classes

### Training (overnight)
- VM-1: Full-data YOLO11x @ 1280px — TRAINING (all 248 images, no val holdout)
- VM-2: Full-data YOLOv9e @ 1280px — TRAINING
- VM-4: Synthetic-augmented YOLO11x @ 1280px — TRAINING (248 + 95 synthetic images)
- VM-5: Pseudo-label YOLO11x @ 1280px — TRAINING (2,489 extra labels)

### Ready for Tomorrow's Submission
Models in submission/ directory:
- yolo_detector.onnx (YOLO11x, 109.7MB)
- yolo_v9e_detector.onnx (YOLOv9e, 110.6MB)
- yolo_l_detector.onnx (old YOLOv8l, 84MB) — may replace
- classifier.onnx (fixed EfficientNet-B2, 31.2MB)
- class_mapping.json (352 classes)
Still need: update baked_data.py with new reference embeddings
