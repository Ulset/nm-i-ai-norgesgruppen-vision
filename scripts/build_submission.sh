#!/usr/bin/env bash
# Package submission.zip for competition upload.
# Usage: bash scripts/build_submission.sh

set -euo pipefail

SUBMISSION_DIR="submission"
OUTPUT="submission.zip"

echo "Checking submission files..."

# Required: run.py and utils.py
for f in run.py utils.py baked_data.py; do
  if [ ! -f "$SUBMISSION_DIR/$f" ]; then
    echo "ERROR: Missing $SUBMISSION_DIR/$f"
    exit 1
  fi
done

# Need at least one YOLO model
YOLO_COUNT=$(find "$SUBMISSION_DIR" -name "yolo_*.onnx" | wc -l | tr -d ' ')
if [ "$YOLO_COUNT" -eq 0 ]; then
  echo "ERROR: No YOLO model found (expected yolo_*.onnx)"
  exit 1
fi
echo "  Found $YOLO_COUNT YOLO model(s)"

# Optional classifier/DINOv2
for f in classifier.onnx dinov2_fp16.onnx; do
  if [ -f "$SUBMISSION_DIR/$f" ]; then
    echo "  Found $f"
  fi
done

echo "Security check: scanning for blocked imports..."
BLOCKED="import os|import sys|import subprocess|import socket|import pickle|import yaml|import threading|import multiprocessing"
if grep -rE "$BLOCKED" "$SUBMISSION_DIR"/*.py 2>/dev/null; then
  echo "ERROR: Blocked imports found! Fix before submitting."
  exit 1
fi
echo "  No blocked imports found."

PY_COUNT=$(find "$SUBMISSION_DIR" -name "*.py" | wc -l | tr -d ' ')
echo "  Python files: $PY_COUNT / 10"
if [ "$PY_COUNT" -gt 10 ]; then
  echo "ERROR: Too many Python files (max 10)!"
  exit 1
fi

WEIGHT_COUNT=$(find "$SUBMISSION_DIR" \( -name "*.onnx" -o -name "*.pt" -o -name "*.npy" -o -name "*.safetensors" \) | wc -l | tr -d ' ')
echo "  Weight files: $WEIGHT_COUNT / 3"
if [ "$WEIGHT_COUNT" -gt 3 ]; then
  echo "ERROR: Too many weight files (max 3)!"
  exit 1
fi

# Check total size
TOTAL_SIZE=$(du -sm "$SUBMISSION_DIR" | cut -f1)
echo "  Total submission size: ${TOTAL_SIZE}MB / 420MB"
if [ "$TOTAL_SIZE" -gt 420 ]; then
  echo "ERROR: Submission too large (${TOTAL_SIZE}MB > 420MB)!"
  exit 1
fi

rm -f "$OUTPUT"
cd "$SUBMISSION_DIR"
zip -r "../$OUTPUT" . -x ".*" "__MACOSX/*" "__pycache__/*" "*.pyc"
cd ..

echo ""
echo "Created $OUTPUT"
unzip -l "$OUTPUT" | head -15
echo ""
echo "Verify run.py is at root (not in a subfolder)."
echo "Upload at: https://app.ainm.no/submit/norgesgruppen-data"
