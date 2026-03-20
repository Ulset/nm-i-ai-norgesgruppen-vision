#!/usr/bin/env bash
# Fix class mapping issues and retrain the EfficientNet-B2 classifier with focal loss.
# Intended to run on a GCP VM with GPU.
set -euo pipefail

LOG=~/training_status.log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

cd "$(dirname "$0")/.."
log "Working directory: $(pwd)"

# ── Step 1: Debug class mapping ──────────────────────────────────────────────
log "=== Step 1: Debugging class mapping ==="

python3 -c "
import json
from pathlib import Path

# Load COCO annotations
ann_path = Path('data/coco_dataset/train/annotations.json')
with open(ann_path) as f:
    coco = json.load(f)

cat_ids = sorted([c['id'] for c in coco['categories']])
print(f'COCO categories: {len(cat_ids)} (min={min(cat_ids)}, max={max(cat_ids)})')
print(f'First 20 IDs: {cat_ids[:20]}')

# Check crop directory structure
crops_dir = Path('data/crops/train')
if crops_dir.exists():
    crop_dirs = sorted([int(d.name) for d in crops_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    print(f'Crop directories: {len(crop_dirs)} (min={min(crop_dirs)}, max={max(crop_dirs)})')
    print(f'First 20 dirs: {crop_dirs[:20]}')

    # Compare
    coco_set = set(cat_ids)
    crop_set = set(crop_dirs)
    missing_crops = coco_set - crop_set
    extra_crops = crop_set - coco_set
    if missing_crops:
        print(f'COCO IDs missing from crops ({len(missing_crops)}): {sorted(missing_crops)[:20]}...')
    if extra_crops:
        print(f'Crop dirs not in COCO ({len(extra_crops)}): {sorted(extra_crops)[:20]}...')
    if not missing_crops and not extra_crops:
        print('All COCO category IDs match crop directories.')
else:
    print('WARNING: data/crops/train does not exist yet')

# Check class_mapping.json if it exists
for mapping in Path('runs/classifier').rglob('class_mapping.json'):
    with open(mapping) as f:
        cm = json.load(f)
    print(f'class_mapping {mapping}: {len(cm)} entries, sample: {dict(list(cm.items())[:5])}')
"
log "Class mapping debug complete"

# ── Step 2: Re-prepare crops ─────────────────────────────────────────────────
log "=== Step 2: Re-preparing crops ==="
python3 -m training.prepare_crops
log "Crop preparation complete"

# ── Step 3: Retrain classifier with focal loss ───────────────────────────────
log "=== Step 3: Retraining classifier with focal loss ==="
python3 -m training.train_classifier \
    --data-dir data/crops \
    --epochs 100 \
    --batch 64 \
    --lr 1e-4 \
    --patience 20 \
    --name effnet_b2_fixed \
    --focal-loss \
    --label-smoothing 0.1
log "Classifier training complete"

# ── Step 4: Build reference embeddings ───────────────────────────────────────
log "=== Step 4: Building reference embeddings ==="
python3 -m training.build_reference_embeddings \
    --model-weights runs/classifier/effnet_b2_fixed/best.pt \
    --class-mapping runs/classifier/effnet_b2_fixed/class_mapping.json \
    --output runs/classifier/effnet_b2_fixed/reference_embeddings.npy
log "Reference embeddings built"

log "=== All steps complete ==="
