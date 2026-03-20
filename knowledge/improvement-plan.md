# Improvement Plan: Parallel Assault (2026-03-20)

## Context
- Current best: 0.8346, rank #89/195
- Top score: 0.9200
- Deadline: Sunday 2026-03-23
- 6 submissions/day, unlimited GCP compute
- Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5

## VM Assignments

### VM-1: YOLO11x @ 1280px
- Model: yolo11x.pt (latest ultralytics)
- Resolution: 1280px, batch: 2, epochs: 300, patience: 50
- Augmentation: mosaic=1.0, mixup=0.2, copy_paste=0.3, close_mosaic=30
- Purpose: Best single-model detection accuracy

### VM-2: YOLOv9e @ 1280px
- Model: yolov9e.pt
- Resolution: 1280px, batch: 2, epochs: 300, patience: 50
- Same augmentation as VM-1
- Purpose: Ensemble diversity (different architecture)

### VM-3: YOLO11x @ 640px
- Model: yolo11x.pt
- Resolution: 640px, batch: 8, epochs: 300, patience: 50
- Same augmentation
- Purpose: Multi-scale ensemble diversity, faster inference

### VM-4: Fix Classifier + Retrain
- Debug class mapping mismatch (root cause of 0.49% val accuracy)
- Re-prepare crops with correct mapping
- Retrain EfficientNet-B2 with focal loss, label smoothing 0.1
- 100 epochs, patience 20
- Rebuild reference embeddings from fixed model

### VM-5: DINOv2 Embeddings
- Export DINOv2-small as ONNX FP16 (~44MB)
- Pre-compute reference embeddings for all 327 products
- Test as replacement for EfficientNet embeddings
- Fallback: only use if it beats fixed classifier

### VM-6: Inference Optimization
- Tune confidence thresholds (lower from 0.7/0.8 to 0.5/0.6)
- Tune WBF IoU (test 0.5, 0.55, 0.6)
- Multi-scale TTA (add 0.9x, 1.1x scale)
- Better scoring logic
- Run evaluate_local.py on VM with data
- ALL compute on GCP, nothing on local Mac

## Weight Budget (420MB max, 3 files max)
- YOLO11x FP16: ~114MB
- YOLOv9e FP16: ~115MB (or YOLOv8l FP16: ~84MB)
- Classifier FP16: ~16MB (or DINOv2-small: ~44MB)
- Estimated total: 245-273MB — within budget

## Timeline
- Thu night: Spin up all VMs, start training
- Fri: Monitor, submit best available, iterate inference on VM
- Sat: Training done, download best models, build ensemble, 6 submissions
- Sun: Final tuning, final submissions

## Key Risks
- YOLO11x/v9e might not improve over YOLOv8x on this dataset
- Classifier fix might not fully resolve the mapping issue
- DINOv2 might not beat fine-tuned EfficientNet for this domain
- Mitigation: parallel approach means any single failure doesn't block progress
