#!/bin/bash
set -e
export PATH=$HOME/.local/bin:$PATH
cd ~/norges-gruppen

echo "=== Starting YOLOv8l training (ensemble model) at $(date) ===" >> ~/training_status.log
python3 -m training.train_yolo --model yolov8l.pt --epochs 200 --batch 8 --imgsz 640 --name yolov8l_640 --patience 30 >> yolo_l_train.log 2>&1
echo "=== YOLOv8l training finished at $(date) ===" >> ~/training_status.log

echo "=== Starting YOLOv8x clean retrain at $(date) ===" >> ~/training_status.log
python3 -m training.train_yolo --model yolov8x.pt --epochs 200 --batch 2 --imgsz 1280 --name yolov8x_shelf_v2 --patience 30 >> yolo_x_v2_train.log 2>&1
echo "=== YOLOv8x clean retrain finished at $(date) ===" >> ~/training_status.log

echo "=== ALL TRAINING DONE at $(date) ===" >> ~/training_status.log
