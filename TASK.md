# NorgesGruppen Object Detection — NM i AI 2026

## Goal

Detect and classify grocery products on store shelves. Submit a `.zip` with model code — it runs in a sandboxed Docker container on competition servers with GPU.

**Score formula:** `0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5`

- **Detection (70%)**: Did you find the products? (IoU >= 0.5, category ignored)
- **Classification (30%)**: Did you identify the right product? (IoU >= 0.5 AND correct category_id)
- Detection-only submissions (all `category_id: 0`) can score up to **0.70**
- Score range: 0.0 to 1.0

## Training Data

Download from the competition **Submit** page (login required).

### COCO Dataset (`NM_NGD_coco_dataset.zip`, ~864 MB)
- 248 shelf images from Norwegian grocery stores
- ~22,700 bounding box annotations (COCO format)
- 357 product categories (IDs 0-355 = products, 356 = `unknown_product`)
- 4 store sections: Egg, Frokost, Knekkebrod, Varmedrikker

### Product Reference Images (`NM_NGD_product_images.zip`, ~60 MB)
- 327 products with multi-angle photos (main, front, back, left, right, top, bottom)
- Organized by barcode: `{product_code}/main.jpg`, etc.
- Includes `metadata.json` with product names and annotation counts

### Annotation Format (`annotations.json`)

```json
{
  "images": [
    {"id": 1, "file_name": "img_00001.jpg", "width": 2000, "height": 1500}
  ],
  "categories": [
    {"id": 0, "name": "VESTLANDSLEFSA TØRRE 10STK 360G", "supercategory": "product"},
    {"id": 356, "name": "unknown_product", "supercategory": "product"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 42,
      "bbox": [141, 49, 169, 152],
      "area": 25688,
      "iscrowd": 0,
      "product_code": "8445291513365",
      "product_name": "NESCAFE VANILLA LATTE 136G NESTLE",
      "corrected": true
    }
  ]
}
```

- `bbox` = `[x, y, width, height]` in pixels (COCO format)
- `product_code` = barcode
- `corrected` = manually verified annotation

## Submission Format

### Zip Structure

```
submission.zip
├── run.py              # Required: entry point
├── model.onnx          # Optional: model weights (.pt, .onnx, .safetensors, .npy)
└── utils.py            # Optional: helper code
```

`run.py` **must be at the zip root** — not inside a subfolder. Most common submission error.

### Zip Limits

| Limit | Value |
|---|---|
| Max zip size (uncompressed) | 420 MB |
| Max files | 1000 |
| Max Python files | 10 |
| Max weight files (.pt, .pth, .onnx, .safetensors, .npy) | 3 |
| Max weight size total | 420 MB |
| Allowed file types | .py, .json, .yaml, .yml, .cfg, .pt, .pth, .onnx, .safetensors, .npy |

### run.py Contract

Executed as:
```bash
python run.py --input /data/images --output /output/predictions.json
```

**Input**: `/data/images/` contains JPEG shelf images (`img_XXXXX.jpg`)

**Output**: JSON array to `--output` path:

```json
[
  {
    "image_id": 42,
    "category_id": 0,
    "bbox": [120.5, 45.0, 80.0, 110.0],
    "score": 0.923
  }
]
```

| Field | Type | Description |
|---|---|---|
| `image_id` | int | From filename: `img_00042.jpg` -> `42` |
| `category_id` | int | Product category (0-355), or 356 for unknown |
| `bbox` | [x, y, w, h] | COCO format bounding box in pixels |
| `score` | float | Confidence (0-1) |

### Creating the Zip

```bash
cd my_submission/
zip -r ../submission.zip . -x ".*" "__MACOSX/*"
```

Verify: `unzip -l submission.zip | head -10` — should show `run.py` directly.

## Sandbox Environment

| Resource | Limit |
|---|---|
| Python | 3.11 |
| CPU | 4 vCPU |
| Memory | 8 GB |
| GPU | NVIDIA L4 (24 GB VRAM) |
| CUDA | 12.4 |
| Network | **None** (fully offline) |
| Timeout | **300 seconds** |

### Pre-installed Packages

PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, ultralytics 8.1.0, onnxruntime-gpu 1.20.0, opencv-python-headless 4.9.0.80, albumentations 1.3.1, Pillow 10.2.0, numpy 1.26.4, scipy 1.12.0, scikit-learn 1.4.0, pycocotools 2.0.7, ensemble-boxes 1.0.9, timm 0.9.12, supervision 0.18.0, safetensors 0.4.2.

**Cannot `pip install` at runtime.**

### Security Restrictions — Blocked Imports

`os`, `sys`, `subprocess`, `socket`, `ctypes`, `builtins`, `importlib`, `pickle`, `marshal`, `shelve`, `shutil`, `yaml`, `requests`, `urllib`, `http.client`, `multiprocessing`, `threading`, `signal`, `gc`

Blocked calls: `eval()`, `exec()`, `compile()`, `__import__()`, `getattr()` with dangerous names.

Also blocked: ELF/Mach-O/PE binaries, symlinks, path traversal.

**Use `pathlib` instead of `os`. Use `json` instead of `yaml`.**

## Model Strategy

### Available Frameworks (pre-installed, pin these versions for training)

| Framework | Models | Version to pin |
|---|---|---|
| ultralytics 8.1.0 | YOLOv8n/s/m/l/x, YOLOv5u, RT-DETR-l/x | `ultralytics==8.1.0` |
| torchvision 0.21.0 | Faster R-CNN, RetinaNet, SSD, FCOS | `torchvision==0.21.0` |
| timm 0.9.12 | ResNet, EfficientNet, ViT, Swin, ConvNeXt (backbones) | `timm==0.9.12` |

### NOT in sandbox (require ONNX export or custom code)

YOLOv9, YOLOv10, YOLO11, RF-DETR, Detectron2, MMDetection, HuggingFace Transformers

### Recommended Weight Formats

| Format | When to use |
|---|---|
| `.onnx` | Universal — any framework, fast inference |
| `.pt` (ultralytics 8.1.0) | Simple YOLOv8/RT-DETR workflow |
| `.pt` (state_dict + model class) | Custom architectures |
| `.safetensors` | Safe loading, no pickle |

### Version Pitfalls

| Risk | Fix |
|---|---|
| ultralytics 8.2+ weights on 8.1.0 | Pin `ultralytics==8.1.0` or export ONNX |
| torch 2.7+ full model save on 2.6.0 | Use `torch.save(model.state_dict())` |
| timm 1.0+ weights on 0.9.12 | Pin `timm==0.9.12` or export ONNX |
| ONNX opset > 20 | Export with `opset_version=17` |

### Key Tips

- GPU always available — larger models (YOLOv8m/l/x) feasible within 300s
- FP16 recommended: smaller weights, faster L4 inference
- `torch.no_grad()` during inference
- Process images individually to stay within 8 GB memory
- Fine-tune on competition data with `nc=357` for correct category_ids
- Pretrained COCO models output COCO class IDs (0-79), NOT product IDs — detection-only

## Submission Limits

| Limit | Value |
|---|---|
| In-flight | 2 per team |
| Per day | 3 per team |
| Infra failure freebies | 2/day (don't count) |

Resets at midnight UTC. You can **select any submission for final evaluation** (not just highest-scoring).

## Common Errors

| Error | Fix |
|---|---|
| `run.py not found at zip root` | Zip contents, not folder |
| `Disallowed file type: __MACOSX/...` | Use terminal zip with `-x ".*" "__MACOSX/*"` |
| `Disallowed file type: .bin` | Rename `.bin` -> `.pt` or convert to `.safetensors` |
| `Security scan found violations` | Remove os/subprocess/socket imports, use pathlib |
| `No predictions.json in output` | Write to `--output` path |
| `Timed out after 300s` | Use GPU (`model.to("cuda")`), or smaller model |
| `Exit code 137` | OOM — reduce batch size or use FP16 |
| `Exit code 139` | Segfault — version mismatch, re-export or use ONNX |
| `ModuleNotFoundError` | Package not in sandbox — export to ONNX |
