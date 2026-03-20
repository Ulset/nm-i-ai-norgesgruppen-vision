#!/bin/bash
set -e
export PATH=$HOME/.local/bin:$PATH
cd ~/norges-gruppen

echo "=== Pseudo-labeling started at $(date) ===" >> ~/training_status.log

# Step 1: Generate pseudo-labels from best YOLO11x model
python3 scripts/pseudo_label.py \
    --model runs/detect/yolo11x_1280/weights/best.pt \
    --images data/coco_dataset/train/images \
    --existing-labels data/yolo_dataset/labels/train \
    --output-dir data/yolo_dataset_pseudo \
    --confidence 0.8 \
    --iou-threshold 0.5 \
    --imgsz 1280 \
    2>&1 | tee pseudo_label.log

echo "=== Pseudo-labeling finished at $(date) ===" >> ~/training_status.log

# Step 2: Train YOLO11x on pseudo-labeled dataset
echo "=== YOLO11x pseudo-label training started at $(date) ===" >> ~/training_status.log

python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11x.pt')
model.train(
    data='data/yolo_dataset_pseudo/dataset_pseudo.yaml',
    imgsz=1280,
    epochs=300,
    batch=2,
    patience=50,
    name='yolo11x_1280_pseudo',
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
" 2>&1 | tee yolo11x_1280_pseudo_train.log

echo "=== YOLO11x pseudo-label training finished at $(date) ===" >> ~/training_status.log

# Step 3: Export to ONNX and validate
python3 -c "
from ultralytics import YOLO
import numpy as np
model = YOLO('runs/detect/yolo11x_1280_pseudo/weights/best.pt')
model.export(format='onnx', imgsz=1280, opset=17, simplify=True, half=True)

import onnxruntime as ort
sess = ort.InferenceSession('runs/detect/yolo11x_1280_pseudo/weights/best.onnx')
inp = np.random.randn(1, 3, 1280, 1280).astype(np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: inp})
shape = out[0].shape
print(f'Output shape: {shape}')
assert 361 in shape, f'Expected 361 (4+357) in shape dimensions, got {shape}'
print('ONNX validation passed!')
" 2>&1 | tee yolo11x_1280_pseudo_export.log

echo "=== YOLO11x pseudo-label export complete at $(date) ===" >> ~/training_status.log
