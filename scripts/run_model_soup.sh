#!/bin/bash
set -e
export PATH=$HOME/.local/bin:$PATH
cd ~/norges-gruppen

echo "=== Model soup started at $(date) ===" >> ~/training_status.log

# Configuration
TRAIN_DIR="runs/detect/yolo11x_1280"
SOUP_DIR="runs/detect/yolo11x_soup/weights"
CHECKPOINTS=(
    "${TRAIN_DIR}/weights/epoch80.pt"
    "${TRAIN_DIR}/weights/epoch90.pt"
    "${TRAIN_DIR}/weights/epoch100.pt"
    "${TRAIN_DIR}/weights/best.pt"
)

# Verify checkpoints exist
echo "Checking checkpoints..."
for ckpt in "${CHECKPOINTS[@]}"; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: Checkpoint not found: $ckpt"
        echo "Available checkpoints:"
        ls -la "${TRAIN_DIR}/weights/"*.pt 2>/dev/null || echo "  (none)"
        exit 1
    fi
    echo "  Found: $ckpt ($(du -h "$ckpt" | cut -f1))"
done

# Step 1: Create model soup
echo ""
echo "=== Step 1: Averaging checkpoints ==="
mkdir -p "$SOUP_DIR"

python3 scripts/model_soup.py \
    --checkpoints "${CHECKPOINTS[@]}" \
    --output "${SOUP_DIR}/soup.pt" \
    --model yolo11x \
    2>&1 | tee yolo11x_soup.log

# Step 2: Run validation on val set
echo ""
echo "=== Step 2: Validating soup on val set ==="
python3 -c "
from ultralytics import YOLO
model = YOLO('${SOUP_DIR}/soup.pt')
metrics = model.val(
    data='data/yolo_dataset/dataset.yaml',
    imgsz=1280,
    batch=4,
    split='val',
    save_json=True,
    name='yolo11x_soup_val',
)
print(f'mAP50:    {metrics.box.map50:.4f}')
print(f'mAP50-95: {metrics.box.map:.4f}')
" 2>&1 | tee -a yolo11x_soup.log

# Step 3: Export to ONNX if validation passed
echo ""
echo "=== Step 3: Exporting soup to FP16 ONNX ==="
python3 scripts/export_fp16_safe.py \
    --weights "${SOUP_DIR}/soup.pt" \
    --imgsz 1280 \
    --output "${SOUP_DIR}/soup_fp16.onnx" \
    2>&1 | tee -a yolo11x_soup.log

echo ""
echo "=== Model soup complete at $(date) ===" >> ~/training_status.log
echo "Results:"
ls -lh "${SOUP_DIR}/"
echo "Done!"
