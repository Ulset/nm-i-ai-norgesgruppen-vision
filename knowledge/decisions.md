# Key Decisions

## Architecture: Two-stage detection + classification
- YOLO detects product bounding boxes on shelf images
- EfficientNet-B2 classifies each crop into 357 product categories (nc=357, IDs 0-355 + 356 safety)
- Reference embeddings used for classification matching (cosine similarity)
- Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5

## Ensemble approach
- Multiple YOLO models at different scales for ensemble diversity
- Weighted Boxes Fusion (WBF) to merge predictions at IoU 0.55
- Test-Time Augmentation: horizontal flip + merge
- Tiled inference for high-res images (1280px tiles, 20% overlap)
- Auto-discover models: run.py globs for yolo_*.onnx files
- ONNX output format auto-detected: handles both (1, 4+nc, N) and (1, N, 4+nc)
- Graceful degradation if only one model available

## Classifier Bug (RESOLVED 2026-03-20)
- **Root cause found**: ImageFolder assigns class indices independently per split.
  Train had 352 classes, val had 278 classes → 272 out of 278 val indices were wrong.
  Model predicted using train indices but val expected its own indices → ~0% accuracy.
- **Fix**: Force val_dataset to use train_dataset.class_to_idx, re-map val samples.
  Commit: ac2d7d7. After fix, val_acc jumped from 0.004 to 0.68 on epoch 1, now at 90.3%.
- **File**: training/train_classifier.py lines 94-110
- **Lesson**: Always check that train/val ImageFolder produce matching class indices

## Submission Constraints
- Max 3 weight files, 420MB total, max 10 Python files
- ONNX opset ≤ 20 (we use 17)
- Can use .pt state_dict with timm/PyTorch since they're in sandbox
- Supported formats: .pt, .pth, .onnx, .safetensors, .npy
- Cannot pip install at runtime
- YOLO11/v9 NOT in sandbox ultralytics 8.1.0 — must use ONNX export

## Weight Budget Analysis
Best case with 2 detectors + classifier:
- YOLO11x FP16 ONNX: ~114MB
- YOLOv9e FP16 ONNX: ~115MB
- Classifier FP16 ONNX: ~16MB (or .pt state_dict: ~30MB)
- Total: ~245-260MB — within 420MB budget

Alternative: load EfficientNet as .pt via timm (available in sandbox)
- Avoids ONNX FP16 conversion issues
- timm 0.9.12 available in sandbox

## Submission Results
- **Best score: 0.8346** — rank #89 of 195 teams
- Top score on leaderboard: 0.9200
- 4 submissions scored, daily quota is 6/day
- All 6 submissions used for 2026-03-20, resets at midnight UTC
- **Deadline: Sunday 2026-03-23**
- GCP compute is UNLIMITED for this tournament
- Baseline archived as `submission_baseline_0.8346.zip`

## Strategy: Parallel Assault (decided 2026-03-20 evening)
- Spin up 6 VMs training in parallel
- YOLO11x@1280 + YOLOv9e@1280 + YOLO11x@640 for detection
- Fix broken classifier (class mapping bug) — DONE
- DINOv2 for better embeddings — embeddings done, ONNX export failed
- Tune inference pipeline on VM-6

## Code Changes Made This Session
- `submission/run.py`: Auto-discover YOLO models, config.json support, DINOv2 path, ONNX format detection
- `submission/utils.py`: Added classify_detections_dinov2() function
- `training/train_classifier.py`: Added FocalLoss, label smoothing, FIXED class mapping bug
- `scripts/build_submission.sh`: Dynamic model discovery, size validation
- New scripts: setup_training_vm.sh, upload_data_to_vm.sh, train_yolo11x_1280.sh, train_yolov9e_1280.sh, train_yolo11x_640.sh, fix_and_retrain_classifier.sh, dinov2_embeddings.py

## Gotchas / Lessons Learned
- `dataset.yaml` is generated at repo root (`~/norges-gruppen/dataset.yaml`), NOT at `data/yolo_dataset/dataset.yaml`
- GCP VMs need `libgl1-mesa-glx` installed for OpenCV (headless images don't have it)
- ultralytics 8.1.0 does NOT support YOLO11/v9 — must upgrade to 8.4+ for training
- SSH to VMs sometimes flakes — use two-step (SCP script, then SSH to execute)
- gcloud `--account` flag must be passed separately, not in a variable like `$GCP`
- DINOv2 via torch.hub has mask_token on CPU even when model is on CUDA — need to handle for ONNX export
