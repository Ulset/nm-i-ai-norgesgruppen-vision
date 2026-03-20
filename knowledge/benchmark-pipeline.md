# Benchmark Pipeline

## Location
VM-6 (`training-inference-tune`), europe-west1-c

## What's On VM-6
- All round 1 ONNX models in `~/models/`:
  - `yolo_11x_r1.onnx` (YOLO11x round 1, 110MB)
  - `yolo_v9e_r1.onnx` (YOLOv9e round 1, 111MB)
  - `yolo_v8l_old.onnx` (old YOLOv8l, 84MB)
  - `classifier.onnx` (fixed EfficientNet-B2, 31MB)
- Full dataset (val: 49 images, ground truth annotations)
- Benchmark script at `~/norges-gruppen/scripts/benchmark_models.sh`
- Submission code (run.py, utils.py, baked_data.py)

## How to Benchmark

### 1. Export ONNX on training VM
```bash
# On the training VM where model finished:
cd ~/norges-gruppen
python3 -c "
from ultralytics import YOLO
model = YOLO('runs/detect/<run_name>/weights/best.pt')
model.export(format='onnx', imgsz=1280, opset=17, simplify=True, half=True)
"
```

### 2. Copy ONNX to VM-6
```bash
ZONE="europe-west1-c"
ACCT="--account=devstar7161@gcplab.me"
PROJ="--project=ainm26osl-716"

# From training VM to VM-6 (via local machine or HTTP server)
gcloud compute scp --zone=$ZONE $ACCT $PROJ \
    <training-vm>:~/norges-gruppen/runs/detect/<run>/weights/best.onnx \
    training-inference-tune:~/models/yolo_<name>.onnx
```

### 3. Run benchmark
```bash
gcloud compute ssh training-inference-tune --zone=$ZONE $ACCT $PROJ \
    --command="cd ~/norges-gruppen && bash ~/run_benchmark.sh"
```

### 4. Read results
```bash
gcloud compute ssh training-inference-tune --zone=$ZONE $ACCT $PROJ \
    --command="cat ~/benchmark_results/summary.txt"
```

## How It Works
- For each `yolo_*.onnx` in `~/models/`, creates a temp submission dir
- Runs the full inference pipeline (run.py) with TTA + tiling + WBF + classifier
- Scores against val ground truth using evaluate_local.py
- Reports: detection_mAP50, classification_mAP50, composite score (0.7*det + 0.3*cls)
- Also tests ensemble (all YOLO models together)

## Performance
- ~10-15 min per model config (49 images, 1280px, TTA + tiling)
- 6-8 configs = ~1.5-2 hours total
- Could add --fast flag to disable TTA for quicker iteration

## Round 2 Models to Benchmark (when done)
- `yolo_11x_full.onnx` — from VM-1 (full-data YOLO11x)
- `yolo_v9e_full.onnx` — from VM-2 (full-data YOLOv9e)
- `yolo_11x_synthetic.onnx` — from VM-4 (synthetic-augmented)
- `yolo_11x_pseudo.onnx` — from VM-5 (pseudo-labeled)
- `yolo_11x_soup.onnx` — from VM-3 (model soup, needs ONNX export)
