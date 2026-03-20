#!/bin/bash
# Benchmark all available YOLO models against validation set.
# Runs on VM-6 (training-inference-tune) which has the dataset.
#
# Usage: bash scripts/benchmark_models.sh
#
# For each ONNX model found in ~/models/, runs the full inference pipeline
# and scores it against the validation ground truth.

set -e
cd ~/norges-gruppen

MODELS_DIR=~/models
RESULTS_DIR=~/benchmark_results
VAL_IMAGES=data/yolo_dataset/images/val
GT_ANNOTATIONS=data/coco_dataset/train/annotations.json
VAL_IDS=data/yolo_dataset/val_image_ids.json

mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "  Model Benchmark Suite"
echo "  $(date)"
echo "============================================"
echo ""

# Check prerequisites
if [ ! -d "$VAL_IMAGES" ]; then
    echo "ERROR: Val images not found at $VAL_IMAGES"
    exit 1
fi

if [ ! -f "$GT_ANNOTATIONS" ]; then
    echo "ERROR: Ground truth not found at $GT_ANNOTATIONS"
    exit 1
fi

# List available models
echo "Available YOLO models:"
ls -lh "$MODELS_DIR"/yolo_*.onnx 2>/dev/null || echo "  (none found)"
echo ""
echo "Available classifiers:"
ls -lh "$MODELS_DIR"/classifier*.onnx 2>/dev/null || echo "  (none found)"
echo ""

# Find the classifier (use the first one found)
CLASSIFIER=$(ls "$MODELS_DIR"/classifier*.onnx 2>/dev/null | head -1)
if [ -n "$CLASSIFIER" ]; then
    echo "Using classifier: $CLASSIFIER"
else
    echo "No classifier found — will run YOLO-only mode"
fi

echo ""
echo "============================================"
echo "  Running benchmarks..."
echo "============================================"
echo ""

# Score summary file
SUMMARY="$RESULTS_DIR/summary.txt"
echo "Model | Det_mAP50 | Cls_mAP50 | Composite" > "$SUMMARY"
echo "------|-----------|-----------|----------" >> "$SUMMARY"

# Test each YOLO model individually
for model in "$MODELS_DIR"/yolo_*.onnx; do
    [ -f "$model" ] || continue
    model_name=$(basename "$model" .onnx)
    echo "--- Testing: $model_name ---"

    # Create temp submission dir with just this model
    tmpdir=$(mktemp -d)
    cp submission/run.py submission/utils.py submission/baked_data.py "$tmpdir/"
    cp "$model" "$tmpdir/yolo_detector.onnx"
    [ -n "$CLASSIFIER" ] && cp "$CLASSIFIER" "$tmpdir/classifier.onnx"

    # Run inference
    pred_file="$RESULTS_DIR/${model_name}_predictions.json"
    python3 "$tmpdir/run.py" --input "$VAL_IMAGES" --output "$pred_file" 2>&1 | tail -5

    # Score
    if [ -f "$pred_file" ]; then
        score=$(python3 -c "
import json, sys
sys.path.insert(0, 'scripts')
from evaluate_local import *

gt = COCO('$GT_ANNOTATIONS')
with open('$pred_file') as f:
    preds = json.load(f)

val_ids = None
if Path('$VAL_IDS').exists():
    val_ids = set(json.load(open('$VAL_IDS')))

det = evaluate_detection_map(gt, preds, val_ids)
cls = evaluate_classification_map(gt, preds, val_ids)
comp = compute_composite_score(det, cls)
print(f'{det:.4f} {cls:.4f} {comp:.4f}')
" 2>&1)
        echo "  Score: $score"
        echo "$model_name | $score" >> "$SUMMARY"
    else
        echo "  FAILED — no predictions generated"
        echo "$model_name | FAILED" >> "$SUMMARY"
    fi

    rm -rf "$tmpdir"
    echo ""
done

# Test ensemble (all YOLO models together)
yolo_count=$(ls "$MODELS_DIR"/yolo_*.onnx 2>/dev/null | wc -l)
if [ "$yolo_count" -gt 1 ]; then
    echo "--- Testing: ENSEMBLE (all models) ---"

    tmpdir=$(mktemp -d)
    cp submission/run.py submission/utils.py submission/baked_data.py "$tmpdir/"
    cp "$MODELS_DIR"/yolo_*.onnx "$tmpdir/"
    [ -n "$CLASSIFIER" ] && cp "$CLASSIFIER" "$tmpdir/classifier.onnx"

    pred_file="$RESULTS_DIR/ensemble_predictions.json"
    python3 "$tmpdir/run.py" --input "$VAL_IMAGES" --output "$pred_file" 2>&1 | tail -5

    if [ -f "$pred_file" ]; then
        score=$(python3 -c "
import json, sys
sys.path.insert(0, 'scripts')
from evaluate_local import *

gt = COCO('$GT_ANNOTATIONS')
with open('$pred_file') as f:
    preds = json.load(f)

val_ids = None
if Path('$VAL_IDS').exists():
    val_ids = set(json.load(open('$VAL_IDS')))

det = evaluate_detection_map(gt, preds, val_ids)
cls = evaluate_classification_map(gt, preds, val_ids)
comp = compute_composite_score(det, cls)
print(f'{det:.4f} {cls:.4f} {comp:.4f}')
" 2>&1)
        echo "  Score: $score"
        echo "ENSEMBLE | $score" >> "$SUMMARY"
    fi

    rm -rf "$tmpdir"
    echo ""
fi

echo "============================================"
echo "  RESULTS SUMMARY"
echo "============================================"
cat "$SUMMARY"
echo ""
echo "Results saved to $RESULTS_DIR/"
