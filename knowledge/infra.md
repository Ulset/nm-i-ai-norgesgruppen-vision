# GCP Infrastructure State (Updated 2026-03-20 ~19:10 UTC)

## Project
- **GCP Project**: ainm26osl-716
- **Region**: europe-west1-c (all GPU VMs)

## GCP Auth — IMPORTANT
- Must use `--account=devstar7161@gcplab.me --project=ainm26osl-716`
- The nmiai-unlimited gcloud config has WRONG project (skyworker-prod) — always pass explicit flags
- SCP works reliably; SSH sometimes flakes (retry or use two-step: SCP script then SSH to run it)

## VMs

### training-vm-1
- **Type**: g2-standard-16 (NVIDIA L4, 24GB VRAM)
- **Zone**: europe-west1-c
- **Status**: RUNNING — training YOLO11x@1280px (epoch ~102/300)
- **Software**: PyTorch 2.10.0, ultralytics 8.4.24, CUDA 12.8
- **On disk**:
  - `~/norges-gruppen/` — full repo + data
  - `runs/detect/train/` — YOLOv8x v1 best (mAP50 0.732)
  - `runs/detect/yolov8l_640/` — YOLOv8l (mAP50 0.710)
  - `runs/detect/yolov8x_shelf_v2/` — YOLOv8x v2 (0.709)
  - `runs/detect/yolo11x_1280/` — YOLO11x TRAINING (best ~0.738)
  - `runs/classify/` — old broken classifier
  - Full dataset (COCO + crops + product images)
- **Log**: `~/yolo11x_1280_full.log`

### training-vm-2
- **Type**: g2-standard-16 (NVIDIA L4, 24GB VRAM)
- **Zone**: europe-west1-c
- **Status**: RUNNING — training YOLOv9e@1280px (epoch ~88/300)
- **Software**: PyTorch 2.10.0, ultralytics 8.4.24
- **On disk**:
  - `runs/detect/yolov9e_1280/` — YOLOv9e TRAINING (best ~0.736)
  - `runs/detect/yolov8x_v2/` — old YOLOv8x v2 (0.731)
  - Full dataset
- **Log**: `~/yolov9e_1280_full.log`

### training-yolo11x-640
- **Type**: g2-standard-16 (NVIDIA L4, 24GB VRAM)
- **Zone**: europe-west1-c
- **Status**: RUNNING — YOLO11x@640px (epoch ~173/300, likely early-stopped)
- **Software**: PyTorch 2.10.0, ultralytics 8.4.24
- **On disk**: `runs/detect/yolo11x_640/` — mAP50 0.724
- **Log**: `~/yolo11x_640_full.log`

### training-classifier-fix
- **Type**: g2-standard-16 (NVIDIA L4, 24GB VRAM)
- **Zone**: europe-west1-c
- **Status**: RUNNING — EfficientNet-B2 v2 with FIXED class mapping (epoch ~50/100)
- **On disk**: `runs/classifier/effnet_b2_fixed_v2/` — val_acc 0.903
- **Log**: `~/classifier_v2_full.log`

### training-dinov2-embeds
- **Type**: g2-standard-16 (NVIDIA L4, 24GB VRAM)
- **Zone**: europe-west1-c
- **Status**: DONE (embeddings) / FAILED (ONNX export — device mismatch)
- **On disk**: `runs/dinov2/dinov2_reference_embeddings.npy` — (357, 384) float16
- **Log**: `~/dinov2_full.log`

### training-inference-tune
- **Type**: g2-standard-16 (NVIDIA L4, 24GB VRAM)
- **Zone**: europe-west1-c
- **Status**: IDLE — waiting for trained models
- **Has**: Full dataset, all code

### astar-solver
- **Type**: e2-small
- **Zone**: europe-north1-b
- **Status**: RUNNING
- **Purpose**: Unrelated (A* solver for different competition task)

## Data Distribution
- All VMs have full dataset via HTTP tarball from VM-1 (3GB)
- Dataset YAML at `~/norges-gruppen/dataset.yaml` (NOT in data/yolo_dataset/)
- Data layout: `data/coco_dataset/`, `data/crops/`, `data/product_images/`, `data/yolo_dataset/`

## Cost Notes
- g2-standard-16 = ~$0.98/hr per VM
- Currently running 6 GPU VMs = ~$5.88/hr
- UNLIMITED compute budget for this tournament
- Stop VMs when not training!
