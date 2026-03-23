# NorgesGruppen Shelf Detective

Grocery product detection for the [NM i AI 2026](https://ainm.no) competition. Given a photo of a store shelf, find every product and figure out what it is.

356 different products, some with as few as *one* training example. A cereal box that only shows up once in the dataset still needs to be recognized.

## How it works

Two-stage pipeline — first find the products, then classify them.

```mermaid
flowchart TD
    A[Shelf Image] --> B{Image > 1280px?}
    B -->|Yes| C[Slice into overlapping tiles]
    B -->|No| D[Run as-is]
    C --> E[YOLO on each tile]
    D --> E
    E --> F[Horizontal flip TTA]
    F --> G[Weighted Box Fusion]
    G --> H[Crop each detection]
    H --> I[EfficientNet-B2 classifier]
    I --> J{Confident?}
    J -->|> 0.7| K[Use classifier prediction]
    J -->|No| L{Reference match?}
    L -->|> 0.8 similarity| M[Use nearest product match]
    L -->|No| N[Fall back to YOLO prediction]
    K --> O[predictions.json]
    M --> O
    N --> O

    style A fill:#4a9eff,color:#fff
    style G fill:#ff6b6b,color:#fff
    style O fill:#51cf66,color:#fff
```

**Stage 1 — Detection:** YOLO detects products using tiled inference (1280px tiles, 20% overlap) to handle huge shelf images (up to 5712px). Horizontal flip TTA doubles detections, and Weighted Box Fusion merges everything. The pipeline auto-discovers all `yolo_*.onnx` files in the submission, so multiple YOLO models can ensemble together.

**Stage 2 — Classification:** Each detected product gets cropped, resized to 224px, and run through an EfficientNet-B2 classifier (90.5% val accuracy). For low-confidence predictions, reference embeddings from product catalog photos provide a cosine-similarity fallback. If neither is confident, the YOLO class prediction is used as-is.

## What we tried

We threw a lot at the wall over the competition weekend:

| Approach | Result | Verdict |
|---|---|---|
| YOLOv8x @ 1280px (baseline) | mAP50 0.732 | Solid starting point |
| YOLO11x @ 1280px | mAP50 0.738 | Best single model |
| YOLOv9e @ 1280px | mAP50 0.736 | Close second |
| YOLO11x @ 640px | mAP50 0.724 | Faster but weaker |
| Multi-model ensemble (WBF) | mAP50 0.708 | Hurt — WBF needed tuning |
| Full-data training (no val holdout) | mAP50 0.987* | *Inflated, but good for submission |
| Pseudo-labeling | mAP50 0.721 | Didn't help |
| Synthetic data (copy-paste augmentation) | mAP50 0.724 | Marginal |
| Model soup (checkpoint averaging) | Not benchmarked | Exported but untested |
| DINOv2 embeddings | Extracted ok | ONNX export failed (device mismatch) |

The classifier was the biggest win — fixing a class mapping bug took it from **0.49% to 90.5% accuracy**, which was basically free points on the 30% classification component.

## Architecture decisions

- **Two-stage over end-to-end**: The dataset is brutally long-tailed (most common product: 422 annotations, 74 products with <5). YOLO finds boxes well but can't classify rare products. The classifier + reference embeddings handle the tail.
- **ONNX for everything**: The sandbox blocks `pickle`, so PyTorch checkpoints can't be loaded directly for YOLO. ONNX also runs faster via `onnxruntime-gpu`.
- **YOLO11/v9 trained with ultralytics 8.4.24**: The sandbox has ultralytics 8.1.0 which doesn't support these architectures, but their ONNX exports work fine — raw tensor math is architecture-agnostic.
- **Baked data**: Reference embeddings and class mappings are embedded directly in `baked_data.py` as base64-encoded numpy arrays (avoids needing extra files that count against the weight limit).

## Project structure

```
├── training/                     # Runs on GCP GPU VMs
│   ├── train_yolo.py             # YOLO fine-tuning (supports v8x/11x/v9e)
│   ├── train_classifier.py       # EfficientNet-B2 with focal loss
│   ├── prepare_yolo_dataset.py   # COCO → YOLO format conversion
│   ├── prepare_crops.py          # Extract product crops for classifier
│   ├── build_reference_embeddings.py  # Compute reference embeddings
│   ├── export_models.py          # ONNX FP16 export
│   └── data_utils.py             # Shared data loading utilities
├── submission/                   # What gets zipped and uploaded
│   ├── run.py                    # Entry point — ensemble + TTA + two-stage
│   ├── utils.py                  # Tiling, WBF, classification logic
│   └── baked_data.py             # Embedded reference embeddings + class mapping
├── scripts/
│   ├── evaluate_local.py         # Local mAP scoring (pycocotools)
│   ├── build_submission.sh       # Package & validate the zip
│   ├── setup_training_vm.sh      # Provision GCP L4 GPU VM
│   ├── upload_data_to_vm.sh      # Push data to VM
│   ├── train_yolo11x_1280.sh     # YOLO11x training config
│   ├── train_yolov9e_1280.sh     # YOLOv9e training config
│   ├── train_yolo11x_640.sh      # YOLO11x 640px config
│   ├── fix_and_retrain_classifier.sh  # Classifier with fixed class mapping
│   ├── dinov2_embeddings.py      # DINOv2 feature extraction
│   ├── pseudo_label.py           # Pseudo-labeling pipeline
│   ├── generate_synthetic_data.py # Copy-paste augmentation
│   ├── model_soup.py             # Checkpoint averaging
│   ├── export_fp16_safe.py       # Safe FP16 ONNX conversion
│   └── prepare_full_dataset.py   # Full-data (no holdout) prep
└── tests/                        # Unit tests
```

## Training data

| What | Size |
|---|---|
| Shelf images | 248 (Egg, Frokost, Knekkebrod, Varmedrikker sections) |
| Annotations | 22,731 bounding boxes |
| Product categories | 356 (plus `unknown_product`) |
| Products per image | 92 avg, up to 235 |
| Product reference photos | 327 products x ~5 angles each |
| Image resolution | 481px to 5712px wide |

## Quickstart

**Prepare data:**
```bash
python3 -m training.prepare_yolo_dataset   # COCO → YOLO format
python3 -m training.prepare_crops          # Extract classifier training crops
```

**Train on GCP:**
```bash
bash scripts/setup_training_vm.sh          # Provision L4 GPU VM
bash scripts/upload_data_to_vm.sh          # Push data to VM

# On the VM:
python3 -m training.train_yolo --model yolo11x.pt --imgsz 1280 --epochs 300 --batch 2
python3 -m training.train_classifier --epochs 100 --batch 64
python3 -m training.build_reference_embeddings \
    --model-weights runs/classifier/best.pt \
    --class-mapping runs/classifier/class_mapping.json
python3 -m training.export_models \
    --yolo-weights runs/detect/best.pt \
    --clf-weights runs/classifier/best.pt
```

**Evaluate & submit:**
```bash
python3 -m scripts.evaluate_local \
    --predictions predictions.json \
    --ground-truth data/coco_dataset/train/annotations.json
bash scripts/build_submission.sh  # Validates size/imports/counts, then zips
```

## Sandbox constraints

The submission runs in a locked-down Docker container:

- **GPU:** NVIDIA L4 (24 GB VRAM)
- **RAM:** 8 GB
- **Timeout:** 300 seconds for the entire test set
- **Network:** None — fully offline
- **Blocked imports:** `os`, `sys`, `subprocess`, `pickle`, `yaml`, `threading`, `multiprocessing`
- **Weight limit:** 420 MB total, max 3 weight files, max 10 Python files

We use ONNX for both models and `pathlib` everywhere instead of `os`.

## Tech stack

| Component | Version | Why |
|---|---|---|
| ultralytics | 8.1.0 (sandbox) / 8.4.24 (training) | YOLO training and export |
| timm | 0.9.12 | EfficientNet-B2 backbone |
| onnxruntime-gpu | 1.20.0 | ONNX inference with CUDA |
| ensemble-boxes | 1.0.9 | Weighted Box Fusion |
| pycocotools | 2.0.7 | mAP evaluation |

## Infrastructure

Training ran on GCP (`europe-north1`) with up to 6 parallel L4 GPU VMs, each running different model variants simultaneously. Unlimited compute budget courtesy of the competition organizers.

## Competition

Part of [NM i AI 2026](https://ainm.no) — the Norwegian AI Championship. NorgesGruppen Data challenge (object detection track).
