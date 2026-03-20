# Parallel Score Improvement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve competition score from 0.8346 to 0.92+ by training better detection models, fixing the broken classifier, and optimizing inference — all in parallel on GCP VMs.

**Architecture:** 6 independent GCP VM workstreams: 3 training new YOLO detectors (YOLO11x@1280, YOLOv9e@1280, YOLO11x@640), 1 fixing the classifier, 1 testing DINOv2 embeddings, 1 tuning inference. All models export to ONNX FP16 for the sandbox.

**Tech Stack:** ultralytics (latest for YOLO11/v9), timm, torch, onnxruntime, DINOv2 via torch.hub, GCP g2-standard-16 VMs with L4 GPUs.

**GCP Auth:** All gcloud commands must use `--account=devstar7161@gcplab.me --project=ainm26osl-716`

---

## File Structure

### New Files
- `scripts/setup_training_vm.sh` — Updated VM provisioning for new ultralytics versions
- `scripts/train_yolo11x.sh` — Training script for VM-1 (YOLO11x@1280)
- `scripts/train_yolov9e.sh` — Training script for VM-2 (YOLOv9e@1280)
- `scripts/train_yolo11x_640.sh` — Training script for VM-3 (YOLO11x@640)
- `scripts/fix_and_retrain_classifier.sh` — Classifier fix + retrain script for VM-4
- `scripts/dinov2_embeddings.py` — DINOv2 embedding extraction + ONNX export for VM-5
- `scripts/tune_inference.sh` — Inference parameter sweep script for VM-6

### Modified Files
- `submission/run.py` — Update model loading for configurable filenames, ONNX output format handling
- `submission/utils.py` — Add DINOv2 integration path in classify_detections
- `training/export_models.py` — Add ONNX shape validation, support YOLO11/v9 exports
- `scripts/build_submission.sh` — Support configurable model filenames
- `training/train_classifier.py` — Add focal loss, label smoothing options
- `training/prepare_crops.py` — Fix class mapping bug

---

## Task 1: Archive Baseline and Prepare Infrastructure

**Files:**
- Create: `scripts/setup_training_vm.sh`

- [ ] **Step 1: Archive baseline submission**

```bash
cp submission.zip submission_baseline_0.8346.zip
```

- [ ] **Step 2: Stop existing VMs (they have stale training, save costs while we prepare)**

```bash
GCP="--account=devstar7161@gcplab.me --project=ainm26osl-716"
gcloud compute instances stop training-vm-1 --zone=europe-west1-c $GCP
gcloud compute instances stop training-vm-2 --zone=europe-west1-c $GCP
```

- [ ] **Step 3: Create updated VM provisioning script**

Create `scripts/setup_training_vm.sh` that:
- Takes VM name as argument (e.g., `yolo11x-1280`)
- Uses `g2-standard-16` with L4 GPU in `europe-west1-c`
- Uses `pytorch-2-6-cu124` deep learning image
- Installs latest `ultralytics` (not 8.1.0 — we need YOLO11/v9 support)
- Installs `timm==0.9.12`, `onnx`, `onnxsim`
- Clones the repo and uploads the dataset
- 200GB disk (extra room for multiple training runs)

```bash
#!/bin/bash
set -e
VM_NAME="${1:?Usage: setup_training_vm.sh <vm-name>}"
GCP="--account=devstar7161@gcplab.me --project=ainm26osl-716"
ZONE="europe-west1-c"

gcloud compute instances create "$VM_NAME" \
    --zone=$ZONE $GCP \
    --machine-type=g2-standard-16 \
    --accelerator=type=nvidia-l4,count=1 \
    --maintenance-policy=TERMINATE \
    --image-family=pytorch-2-6-cu124 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-balanced \
    --metadata="install-nvidia-driver=True" \
    --scopes=storage-full

echo "Waiting for VM to boot..."
sleep 30

# Install dependencies
gcloud compute ssh "$VM_NAME" --zone=$ZONE $GCP --command="
    pip install -q ultralytics timm==0.9.12 onnx onnxsim
    pip install -q opencv-python-headless
    mkdir -p ~/norges-gruppen
"

echo "VM $VM_NAME ready."
```

- [ ] **Step 4: Create a data upload helper**

The dataset needs to be on each VM. Since existing VMs already have data, we can SCP from those or re-upload. Create a small helper:

```bash
# scripts/upload_data_to_vm.sh
#!/bin/bash
VM_NAME="${1:?Usage: upload_data_to_vm.sh <vm-name>}"
GCP="--account=devstar7161@gcplab.me --project=ainm26osl-716"
ZONE="europe-west1-c"

# Upload repo code
gcloud compute scp --recurse --zone=$ZONE $GCP \
    training/ scripts/ submission/ CLAUDE.md \
    "$VM_NAME":~/norges-gruppen/

# Copy data from training-vm-1 (already has the dataset)
echo "Data must be copied from existing VM or re-uploaded. See knowledge/infra.md for data locations."
```

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_training_vm.sh scripts/upload_data_to_vm.sh
git commit -m "Add updated VM provisioning scripts for YOLO11/v9 training"
```

---

## Task 2: Create YOLO Training Scripts for VMs 1-3

**Files:**
- Create: `scripts/train_yolo11x_1280.sh`
- Create: `scripts/train_yolov9e_1280.sh`
- Create: `scripts/train_yolo11x_640.sh`

- [ ] **Step 1: Create YOLO11x 1280px training script**

```bash
#!/bin/bash
# scripts/train_yolo11x_1280.sh — Run on VM-1
set -e
export PATH=$HOME/.local/bin:$PATH
cd ~/norges-gruppen

echo "=== YOLO11x 1280px training started at $(date) ===" >> ~/training_status.log

python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11x.pt')
model.train(
    data='data/yolo_dataset/dataset.yaml',
    imgsz=1280,
    epochs=300,
    batch=2,
    patience=50,
    name='yolo11x_1280',
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.3,
    close_mosaic=30,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    fliplr=0.5, flipud=0.0,
    optimizer='SGD',
    lr0=0.01, lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    save=True, save_period=10,
    val=True, plots=True,
)
" 2>&1 | tee yolo11x_1280_train.log

echo "=== YOLO11x 1280px training finished at $(date) ===" >> ~/training_status.log

# Export to ONNX
python3 -c "
from ultralytics import YOLO
import numpy as np
model = YOLO('runs/detect/yolo11x_1280/weights/best.pt')
model.export(format='onnx', imgsz=1280, opset=17, simplify=True, half=True)
print('ONNX export complete')

# Validate output shape
import onnxruntime as ort
sess = ort.InferenceSession('runs/detect/yolo11x_1280/weights/best.onnx')
inp = np.random.randn(1, 3, 1280, 1280).astype(np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: inp})
print(f'Output shape: {out[0].shape}')
assert len(out[0].shape) == 3, f'Expected 3D output, got {out[0].shape}'
print(f'Output format: (1, {out[0].shape[1]}, {out[0].shape[2]})')
" 2>&1 | tee yolo11x_1280_export.log

echo "=== YOLO11x 1280px export finished at $(date) ===" >> ~/training_status.log
```

- [ ] **Step 2: Create YOLOv9e 1280px training script**

Same structure, but with `yolov9e.pt` and `name='yolov9e_1280'`.

```bash
#!/bin/bash
# scripts/train_yolov9e_1280.sh — Run on VM-2
set -e
export PATH=$HOME/.local/bin:$PATH
cd ~/norges-gruppen

echo "=== YOLOv9e 1280px training started at $(date) ===" >> ~/training_status.log

python3 -c "
from ultralytics import YOLO
model = YOLO('yolov9e.pt')
model.train(
    data='data/yolo_dataset/dataset.yaml',
    imgsz=1280,
    epochs=300,
    batch=2,
    patience=50,
    name='yolov9e_1280',
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.3,
    close_mosaic=30,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    fliplr=0.5, flipud=0.0,
    optimizer='SGD',
    lr0=0.01, lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    save=True, save_period=10,
    val=True, plots=True,
)
" 2>&1 | tee yolov9e_1280_train.log

echo "=== YOLOv9e 1280px training finished at $(date) ===" >> ~/training_status.log

# Export to ONNX + validate
python3 -c "
from ultralytics import YOLO
import numpy as np
model = YOLO('runs/detect/yolov9e_1280/weights/best.pt')
model.export(format='onnx', imgsz=1280, opset=17, simplify=True, half=True)

import onnxruntime as ort
sess = ort.InferenceSession('runs/detect/yolov9e_1280/weights/best.onnx')
inp = np.random.randn(1, 3, 1280, 1280).astype(np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: inp})
print(f'Output shape: {out[0].shape}')
assert len(out[0].shape) == 3, f'Expected 3D output, got {out[0].shape}'
" 2>&1 | tee yolov9e_1280_export.log

echo "=== YOLOv9e 1280px export finished at $(date) ===" >> ~/training_status.log
```

- [ ] **Step 3: Create YOLO11x 640px training script**

```bash
#!/bin/bash
# scripts/train_yolo11x_640.sh — Run on VM-3
set -e
export PATH=$HOME/.local/bin:$PATH
cd ~/norges-gruppen

echo "=== YOLO11x 640px training started at $(date) ===" >> ~/training_status.log

python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11x.pt')
model.train(
    data='data/yolo_dataset/dataset.yaml',
    imgsz=640,
    epochs=300,
    batch=8,
    patience=50,
    name='yolo11x_640',
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.3,
    close_mosaic=30,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    fliplr=0.5, flipud=0.0,
    optimizer='SGD',
    lr0=0.01, lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    save=True, save_period=10,
    val=True, plots=True,
)
" 2>&1 | tee yolo11x_640_train.log

echo "=== YOLO11x 640px training finished at $(date) ===" >> ~/training_status.log

# Export to ONNX + validate
python3 -c "
from ultralytics import YOLO
import numpy as np
model = YOLO('runs/detect/yolo11x_640/weights/best.pt')
model.export(format='onnx', imgsz=640, opset=17, simplify=True, half=True)

import onnxruntime as ort
sess = ort.InferenceSession('runs/detect/yolo11x_640/weights/best.onnx')
inp = np.random.randn(1, 3, 640, 640).astype(np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: inp})
print(f'Output shape: {out[0].shape}')
assert len(out[0].shape) == 3, f'Expected 3D output, got {out[0].shape}'
" 2>&1 | tee yolo11x_640_export.log

echo "=== YOLO11x 640px export finished at $(date) ===" >> ~/training_status.log
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_yolo11x_1280.sh scripts/train_yolov9e_1280.sh scripts/train_yolo11x_640.sh
git commit -m "Add YOLO11x and YOLOv9e training scripts for parallel VM training"
```

---

## Task 3: Create Classifier Fix and DINOv2 Scripts

**Files:**
- Create: `scripts/fix_and_retrain_classifier.sh`
- Create: `scripts/dinov2_embeddings.py`

- [ ] **Step 1: Create classifier fix + retrain script**

This script will run on VM-4. It:
1. Inspects the crop directory structure to find the mapping bug
2. Compares COCO category_ids with ImageFolder indices
3. Fixes prepare_crops.py if needed
4. Re-prepares crops
5. Retrains with focal loss

```bash
#!/bin/bash
# scripts/fix_and_retrain_classifier.sh — Run on VM-4
set -e
export PATH=$HOME/.local/bin:$PATH
cd ~/norges-gruppen

echo "=== Classifier debug started at $(date) ===" >> ~/training_status.log

# Step 1: Debug class mapping
python3 -c "
import json
from pathlib import Path

# Load COCO annotations
with open('data/coco_dataset/train/annotations.json') as f:
    coco = json.load(f)

# Get all category IDs from annotations
cat_ids = sorted(set(c['id'] for c in coco['categories']))
print(f'COCO category IDs: {len(cat_ids)} categories')
print(f'ID range: {min(cat_ids)} to {max(cat_ids)}')
print(f'First 10: {cat_ids[:10]}')

# Check crop directory structure
crop_dir = Path('data/crops/train')
if crop_dir.exists():
    dirs = sorted([int(d.name) for d in crop_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    print(f'Crop directories: {len(dirs)} dirs')
    print(f'Dir range: {min(dirs)} to {max(dirs)}')
    print(f'First 10: {dirs[:10]}')

    # Check for mismatches
    coco_set = set(cat_ids)
    dir_set = set(dirs)
    missing_dirs = coco_set - dir_set
    extra_dirs = dir_set - coco_set
    if missing_dirs:
        print(f'WARNING: {len(missing_dirs)} COCO categories missing from crops: {sorted(missing_dirs)[:10]}...')
    if extra_dirs:
        print(f'WARNING: {len(extra_dirs)} crop dirs not in COCO: {sorted(extra_dirs)[:10]}...')
else:
    print('No crop directory found - need to run prepare_crops.py first')

# Check class_mapping.json if it exists
mapping_path = Path('runs/classifier')
for p in mapping_path.rglob('class_mapping.json'):
    with open(p) as f:
        mapping = json.load(f)
    print(f'class_mapping.json at {p}: {len(mapping)} entries')
    # ImageFolder assigns indices alphabetically
    # Check if mapping correctly maps ImageFolder index -> category_id
    print(f'First 5 entries: {dict(list(mapping.items())[:5])}')
" 2>&1 | tee classifier_debug.log

# Step 2: Re-prepare crops (the script will be updated locally first if bug is found)
echo "=== Re-preparing crops at $(date) ===" >> ~/training_status.log
python3 -m training.prepare_crops 2>&1 | tee prepare_crops.log

# Step 3: Retrain classifier
echo "=== Classifier retrain started at $(date) ===" >> ~/training_status.log
python3 -m training.train_classifier \
    --data-dir data/crops \
    --epochs 100 \
    --batch 64 \
    --lr 1e-4 \
    --patience 20 \
    --name effnet_b2_fixed \
    2>&1 | tee classifier_retrain.log

echo "=== Classifier retrain finished at $(date) ===" >> ~/training_status.log

# Step 4: Build reference embeddings
echo "=== Building embeddings at $(date) ===" >> ~/training_status.log
python3 -m training.build_reference_embeddings \
    --model-weights runs/classifier/effnet_b2_fixed/best.pt \
    --class-mapping runs/classifier/effnet_b2_fixed/class_mapping.json \
    --output runs/classifier/effnet_b2_fixed/reference_embeddings.npy \
    2>&1 | tee embeddings.log

echo "=== Classifier pipeline complete at $(date) ===" >> ~/training_status.log
```

- [ ] **Step 2: Create DINOv2 embeddings script**

```python
#!/usr/bin/env python3
"""scripts/dinov2_embeddings.py — Run on VM-5
Extract DINOv2 embeddings for product reference images and export model to ONNX."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def load_dinov2(model_name: str = "dinov2_vits14") -> nn.Module:
    """Load DINOv2 model from torch.hub."""
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    return model


def get_transform() -> transforms.Compose:
    """DINOv2 preprocessing: resize to 224, normalize with ImageNet stats."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_embeddings(
    model: nn.Module, image_dir: Path, transform: transforms.Compose, device: torch.device
) -> dict[str, np.ndarray]:
    """Extract embeddings for all images in a directory structure.

    Expects: image_dir/{category_id}/{image_files}
    Returns: {category_id: mean_embedding}
    """
    embeddings = {}

    for cat_dir in sorted(image_dir.iterdir()):
        if not cat_dir.is_dir():
            continue

        cat_id = cat_dir.name
        cat_embeddings = []

        for img_path in sorted(cat_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model(tensor)  # (1, 384) for vits14

            cat_embeddings.append(emb.cpu().numpy()[0])

        if cat_embeddings:
            mean_emb = np.mean(cat_embeddings, axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
            embeddings[cat_id] = mean_emb.astype(np.float16)

    return embeddings


def export_onnx(model: nn.Module, output_path: Path, device: torch.device):
    """Export DINOv2 to ONNX with FP16 weights."""
    dummy = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["input"],
        output_names=["embeddings"],
        dynamic_axes={"input": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=17,
    )

    # Convert to FP16
    import onnx
    from onnx import numpy_helper

    onnx_model = onnx.load(str(output_path))
    for tensor in onnx_model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT:
            arr = numpy_helper.to_array(tensor).astype(np.float16)
            new_tensor = numpy_helper.from_array(arr, tensor.name)
            new_tensor.data_type = onnx.TensorProto.FLOAT16
            tensor.CopyFrom(new_tensor)

    fp16_path = output_path.with_name(output_path.stem + "_fp16.onnx")
    onnx.save(onnx_model, str(fp16_path))

    # Validate
    import onnxruntime as ort
    sess = ort.InferenceSession(str(fp16_path))
    inp = np.random.randn(1, 3, 224, 224).astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: inp})
    print(f"DINOv2 ONNX output shape: {out[0].shape}")
    print(f"FP16 model size: {fp16_path.stat().st_size / 1024 / 1024:.1f}MB")

    return fp16_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-images", type=Path, default=Path("data/product_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dinov2"))
    parser.add_argument("--model", default="dinov2_vits14")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model}...")
    model = load_dinov2(args.model).to(device)
    transform = get_transform()

    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, args.product_images, transform, device)

    # Build embedding matrix (357 classes, 384 dims for vits14)
    num_classes = 357
    emb_dim = 384
    emb_matrix = np.zeros((num_classes, emb_dim), dtype=np.float16)

    for cat_id_str, emb in embeddings.items():
        cat_id = int(cat_id_str)
        if cat_id < num_classes:
            emb_matrix[cat_id] = emb

    np.save(args.output_dir / "dinov2_reference_embeddings.npy", emb_matrix)
    print(f"Saved reference embeddings: {emb_matrix.shape}")

    print("Exporting ONNX...")
    fp16_path = export_onnx(model, args.output_dir / "dinov2.onnx", device)
    print(f"DINOv2 ONNX exported to {fp16_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add scripts/fix_and_retrain_classifier.sh scripts/dinov2_embeddings.py
git commit -m "Add classifier fix and DINOv2 embedding extraction scripts"
```

---

## Task 4: Update Submission Code for New Models

**Files:**
- Modify: `submission/run.py`
- Modify: `submission/utils.py`
- Modify: `training/export_models.py`
- Modify: `scripts/build_submission.sh`

- [ ] **Step 1: Update run.py model loading to be configurable**

In `submission/run.py`, update the model loading section to detect available ONNX files dynamically rather than hardcoding filenames. The key change: scan for `*.onnx` files matching known patterns, and auto-detect input size from ONNX model metadata.

Changes to run.py:
- Replace hardcoded `yolo_detector.onnx` / `yolo_l_detector.onnx` with a discovery pattern
- Add ONNX output shape detection (handle both `(1, 4+nc, N)` and `(1, N, 4+nc)` formats)
- Keep `classifier.onnx` as-is (or `dinov2_fp16.onnx` for DINOv2 path)

Specific code changes:

```python
# Replace lines ~350-370 in run.py with:
# Auto-discover YOLO models
yolo_models = []
for onnx_file in sorted(script_dir.glob("yolo_*.onnx")):
    sess = ort.InferenceSession(str(onnx_file), providers=providers)
    input_shape = sess.get_inputs()[0].shape  # e.g., [1, 3, 1280, 1280]
    imgsz = input_shape[2] if isinstance(input_shape[2], int) else 1280
    yolo_models.append({"session": sess, "path": onnx_file, "imgsz": imgsz})
    print(f"Loaded YOLO: {onnx_file.name} @ {imgsz}px")
```

- [ ] **Step 2: Add ONNX output format handling**

YOLO11 and YOLOv9 may output `(1, N, 4+nc)` instead of `(1, 4+nc, N)`. Add auto-detection:

```python
# In run_yolo_on_tile() function, after getting raw output:
raw = output[0]  # shape: (1, X, Y)
if raw.shape[1] < raw.shape[2]:
    # Format: (1, 4+nc, N) — standard YOLOv8
    predictions = raw[0].T  # (N, 4+nc)
else:
    # Format: (1, N, 4+nc) — some YOLO11/v9 exports
    predictions = raw[0]  # (N, 4+nc)
```

- [ ] **Step 3: Add DINOv2 classification path in utils.py**

Add a new function for DINOv2-based classification:

```python
def classify_detections_dinov2(
    crop_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_threshold: float = 0.6,
    unknown_product_id: int = 355,
) -> tuple[int, float]:
    """Classify using DINOv2 cosine similarity only (no softmax logits)."""
    similarities = reference_embeddings @ crop_embedding
    best_idx = int(np.argmax(similarities))
    best_sim = float(similarities[best_idx])

    if best_sim >= reference_threshold:
        return best_idx, best_sim
    return unknown_product_id, best_sim
```

- [ ] **Step 4: Update build_submission.sh**

Update to include all ONNX files matching `yolo_*.onnx` pattern and either `classifier.onnx` or `dinov2_fp16.onnx`:

```bash
# Replace hardcoded file list with glob
for f in submission/yolo_*.onnx submission/classifier.onnx submission/dinov2_fp16.onnx; do
    [ -f "$f" ] && FILES="$FILES $f"
done
```

- [ ] **Step 5: Update export_models.py with ONNX validation**

Add shape validation after every export:

```python
def validate_onnx(onnx_path: Path, imgsz: int):
    """Validate ONNX model output shape matches expected format."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    inp = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: inp})
    shape = out[0].shape
    print(f"ONNX validation: {onnx_path.name} -> shape {shape}")
    assert len(shape) == 3, f"Expected 3D output, got {shape}"
    # Determine format
    if shape[1] < shape[2]:
        print(f"  Format: (1, {shape[1]}, {shape[2]}) — standard (4+nc, N)")
    else:
        print(f"  Format: (1, {shape[1]}, {shape[2]}) — transposed (N, 4+nc)")
    return shape
```

- [ ] **Step 6: Commit**

```bash
git add submission/run.py submission/utils.py training/export_models.py scripts/build_submission.sh
git commit -m "Update submission pipeline for YOLO11/v9 and DINOv2 support"
```

---

## Task 5: Create Inference Tuning Script

**Files:**
- Create: `scripts/tune_inference.sh`

- [ ] **Step 1: Create parameter sweep script**

This runs on VM-6 with the dataset and current models:

```bash
#!/bin/bash
# scripts/tune_inference.sh — Run on VM-6
set -e
cd ~/norges-gruppen

echo "=== Inference tuning started at $(date) ===" >> ~/training_status.log

# Run parameter sweep
python3 -c "
import subprocess
import json

# Parameters to sweep
configs = [
    {'conf_floor': 0.05, 'clf_thr': 0.7, 'ref_thr': 0.8, 'wbf_iou': 0.55},  # baseline
    {'conf_floor': 0.03, 'clf_thr': 0.5, 'ref_thr': 0.6, 'wbf_iou': 0.55},
    {'conf_floor': 0.03, 'clf_thr': 0.5, 'ref_thr': 0.6, 'wbf_iou': 0.50},
    {'conf_floor': 0.03, 'clf_thr': 0.5, 'ref_thr': 0.6, 'wbf_iou': 0.45},
    {'conf_floor': 0.01, 'clf_thr': 0.4, 'ref_thr': 0.5, 'wbf_iou': 0.50},
    {'conf_floor': 0.05, 'clf_thr': 0.6, 'ref_thr': 0.7, 'wbf_iou': 0.50},
    {'conf_floor': 0.03, 'clf_thr': 0.5, 'ref_thr': 0.6, 'wbf_iou': 0.60},
]

results = []
for i, cfg in enumerate(configs):
    print(f'Config {i+1}/{len(configs)}: {cfg}')
    # Would need to modify run.py to accept these as env vars or CLI args
    # For now, log the plan
    results.append(cfg)

print(json.dumps(results, indent=2))
" 2>&1 | tee tuning_results.log

echo "=== Inference tuning finished at $(date) ===" >> ~/training_status.log
```

Note: The actual tuning requires making run.py accept parameters via environment variables. We'll add env var overrides to run.py:

```python
# At top of run.py, after constants:
import json as _json
CONFIDENCE_FLOOR = float(__import__('pathlib').Path('/tmp/conf_floor').read_text()) if __import__('pathlib').Path('/tmp/conf_floor').exists() else 0.05
# ... etc for each tunable parameter
```

Actually, simpler: read from a `config.json` if present:

```python
# In run.py, after constant definitions:
_config_path = script_dir / "config.json"
if _config_path.exists():
    _cfg = _json.loads(_config_path.read_text())
    CONFIDENCE_FLOOR = _cfg.get("confidence_floor", CONFIDENCE_FLOOR)
    CLASSIFIER_THRESHOLD = _cfg.get("classifier_threshold", CLASSIFIER_THRESHOLD)
    REFERENCE_THRESHOLD = _cfg.get("reference_threshold", REFERENCE_THRESHOLD)
    WBF_IOU_THR = _cfg.get("wbf_iou_thr", WBF_IOU_THR)
```

- [ ] **Step 2: Commit**

```bash
git add scripts/tune_inference.sh
git commit -m "Add inference parameter tuning script"
```

---

## Task 6: Provision VMs and Launch All Training

**Files:** None (infrastructure only)

- [ ] **Step 1: Create 4 new VMs (reuse existing 2)**

```bash
GCP="--account=devstar7161@gcplab.me --project=ainm26osl-716"
ZONE="europe-west1-c"

# VM-1 and VM-2 already exist — reuse them with new ultralytics
# Create VM-3 through VM-6
for vm in yolo11x-640 classifier-fix dinov2-embeds inference-tune; do
    gcloud compute instances create "training-$vm" \
        --zone=$ZONE $GCP \
        --machine-type=g2-standard-16 \
        --accelerator=type=nvidia-l4,count=1 \
        --maintenance-policy=TERMINATE \
        --image-family=pytorch-2-6-cu124 \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=200GB \
        --boot-disk-type=pd-balanced \
        --metadata="install-nvidia-driver=True" \
        --scopes=storage-full &
done
wait
```

- [ ] **Step 2: Upload code and data to all VMs**

For each VM, SCP the repo code and ensure dataset is available. Existing VMs (training-vm-1, training-vm-2) already have data. New VMs need data copied.

- [ ] **Step 3: Update ultralytics on existing VMs**

```bash
for vm in training-vm-1 training-vm-2; do
    gcloud compute ssh "$vm" --zone=$ZONE $GCP --command="pip install -q ultralytics --upgrade" &
done
wait
```

- [ ] **Step 4: Launch all training in parallel**

```bash
# VM-1: YOLO11x 1280
gcloud compute ssh training-vm-1 --zone=$ZONE $GCP --command="nohup bash ~/norges-gruppen/scripts/train_yolo11x_1280.sh > /dev/null 2>&1 &"

# VM-2: YOLOv9e 1280
gcloud compute ssh training-vm-2 --zone=$ZONE $GCP --command="nohup bash ~/norges-gruppen/scripts/train_yolov9e_1280.sh > /dev/null 2>&1 &"

# VM-3: YOLO11x 640
gcloud compute ssh training-yolo11x-640 --zone=$ZONE $GCP --command="nohup bash ~/norges-gruppen/scripts/train_yolo11x_640.sh > /dev/null 2>&1 &"

# VM-4: Classifier fix
gcloud compute ssh training-classifier-fix --zone=$ZONE $GCP --command="nohup bash ~/norges-gruppen/scripts/fix_and_retrain_classifier.sh > /dev/null 2>&1 &"

# VM-5: DINOv2
gcloud compute ssh training-dinov2-embeds --zone=$ZONE $GCP --command="nohup python3 ~/norges-gruppen/scripts/dinov2_embeddings.py > ~/dinov2.log 2>&1 &"

# VM-6: Inference tuning (after uploading current models)
gcloud compute ssh training-inference-tune --zone=$ZONE $GCP --command="nohup bash ~/norges-gruppen/scripts/tune_inference.sh > /dev/null 2>&1 &"
```

- [ ] **Step 5: Verify all training started**

```bash
for vm in training-vm-1 training-vm-2 training-yolo11x-640 training-classifier-fix training-dinov2-embeds training-inference-tune; do
    echo "=== $vm ==="
    gcloud compute ssh "$vm" --zone=$ZONE $GCP --command="cat ~/training_status.log 2>/dev/null; nvidia-smi | grep 'python\|MiB' | head -3" 2>/dev/null
done
```

---

## Task 7: Monitor and Collect Results (Ongoing)

- [ ] **Step 1: Check training progress periodically**

```bash
# Quick status check script
GCP="--account=devstar7161@gcplab.me --project=ainm26osl-716"
ZONE="europe-west1-c"

for vm in training-vm-1 training-vm-2 training-yolo11x-640 training-classifier-fix; do
    echo "=== $vm ==="
    gcloud compute ssh "$vm" --zone=$ZONE $GCP --command="
        tail -1 ~/training_status.log 2>/dev/null
        tail -3 ~/norges-gruppen/*.log 2>/dev/null | grep -E 'mAP|epoch|Epoch'
    " 2>/dev/null
done
```

- [ ] **Step 2: When training completes, download best models**

```bash
# Download from each VM
GCP="--account=devstar7161@gcplab.me --project=ainm26osl-716"
ZONE="europe-west1-c"

# YOLO11x 1280 best
gcloud compute scp $GCP --zone=$ZONE \
    training-vm-1:~/norges-gruppen/runs/detect/yolo11x_1280/weights/best.onnx \
    submission/yolo_detector.onnx

# YOLOv9e or YOLO11x 640 (pick the better one)
gcloud compute scp $GCP --zone=$ZONE \
    training-vm-2:~/norges-gruppen/runs/detect/yolov9e_1280/weights/best.onnx \
    submission/yolo_l_detector.onnx

# Fixed classifier
gcloud compute scp $GCP --zone=$ZONE \
    training-classifier-fix:~/norges-gruppen/runs/classifier/effnet_b2_fixed/best.pt \
    ./classifier_fixed.pt
# Then export to ONNX locally or on VM
```

- [ ] **Step 3: Build submission and test locally**

```bash
bash scripts/build_submission.sh
# Verify zip contents and size
unzip -l submission.zip
du -sh submission.zip
```

- [ ] **Step 4: Update knowledge base with results**

Update `knowledge/training-log.md` with all new training results.
Update `knowledge/infra.md` with new VM states.
