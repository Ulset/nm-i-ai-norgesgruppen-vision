"""Fine-tune YOLOv8x on NorgesGruppen shelf data.

Runs on GCP GPU VM with ultralytics==8.1.0.

Usage:
  python -m training.train_yolo [--imgsz 1280] [--epochs 150] [--batch 4]
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--model", default="yolov8x.pt", help="Base model")
    parser.add_argument("--dataset", default="dataset.yaml")
    parser.add_argument("--name", default="yolov8x_shelf")
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.dataset,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    print(f"Training complete. Best weights: runs/detect/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
