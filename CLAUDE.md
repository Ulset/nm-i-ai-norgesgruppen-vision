# NorgesGruppen Object Detection — NM i AI 2026

## Competition
- **Task**: Detect and classify grocery products on store shelves
- **Score**: 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5
- **Submission**: .zip with run.py + model weights, runs in sandboxed Docker (L4 GPU, 300s timeout)
- **Submission URL**: https://app.ainm.no/submit/norgesgruppen-data

## GCP Infrastructure
- **Project**: `ainm26osl-716` (unlimited compute)
- **User**: `devstar7161@gcplab.me`
- **Service account**: `59268370848-compute@developer.gserviceaccount.com`
- **gcloud profile**: `nmiai-unlimited`
- **Region**: `europe-north1`
- **Compute**: UNLIMITED — use as many VMs/GPUs as needed

## Key Constraints
- Sandbox: Python 3.11, L4 GPU (24GB VRAM), 8GB RAM, 300s timeout, NO network
- Blocked imports: os, sys, subprocess, pickle, yaml, threading, multiprocessing
- Use `pathlib` instead of `os`, `json` instead of `yaml`
- Max 3 weight files, 420MB total, 10 Python files
- Pin `ultralytics==8.1.0`, `timm==0.9.12` for training (match sandbox versions)
- Train with nc=357 (IDs 0-355 + 356 safety)
- 6 submissions/day max — validate locally first

## Running
```bash
# Prepare YOLO dataset from COCO annotations
python training/prepare_yolo_dataset.py

# Extract product crops for classifier training
python training/prepare_crops.py

# Local evaluation
python scripts/evaluate_local.py --predictions predictions.json --ground-truth data/coco_dataset/train/annotations.json

# Build submission zip
bash scripts/build_submission.sh
```

## Git Commits
- **NEVER add Co-Authored-By lines** to commit messages
