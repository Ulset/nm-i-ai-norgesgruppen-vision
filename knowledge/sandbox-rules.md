# Sandbox Environment Rules (from app.ainm.no/docs)

## Pre-installed Packages (exact versions)
- PyTorch 2.6.0+cu124
- torchvision 0.21.0+cu124
- ultralytics 8.1.0
- onnxruntime-gpu 1.20.0
- opencv-python-headless 4.9.0.80
- albumentations 1.3.1
- Pillow 10.2.0
- numpy 1.26.4
- scipy 1.12.0
- scikit-learn 1.4.0
- pycocotools 2.0.7
- ensemble-boxes 1.0.9
- timm 0.9.12
- supervision 0.18.0
- safetensors 0.4.2

## Cannot pip install at runtime

## Supported Model Architectures (native)
- YOLOv8n/s/m/l/x, YOLOv5u, RT-DETR-l/x (via ultralytics)
- Faster R-CNN, RetinaNet, SSD, FCOS, Mask R-CNN (via torchvision)
- ResNet, EfficientNet, ViT, Swin, ConvNeXt (via timm)

## NOT Available (require ONNX or custom code)
- YOLOv9, YOLOv10, YOLO11, RF-DETR
- Detectron2, MMDetection, HuggingFace Transformers

**Workaround**: Export to ONNX or include model code with state_dict weights.

## ONNX Requirements
- opset version ≤ 20 (we use 17, so fine)
- Runs with onnxruntime-gpu 1.20.0
- Providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]

## Blocked Imports
os, sys, subprocess, socket, ctypes, builtins, importlib,
pickle, marshal, shelve, shutil, yaml, requests, urllib,
http.client, multiprocessing, threading, signal, gc, code, codeop, pty

## Blocked Functions
eval(), exec(), compile(), __import__(), getattr() with dangerous names

## Alternative: PyTorch state_dict Loading
Since PyTorch 2.6.0 and timm 0.9.12 are available in sandbox,
we could load EfficientNet-B2 as a .pt state_dict directly instead
of ONNX. This avoids FP16 conversion issues.
Example: timm.create_model("efficientnet_b2", num_classes=357) + model.load_state_dict(...)

## File Limits
- Max 3 weight files (.onnx, .pt, .pth, .npy, .safetensors)
- Max 10 Python files
- Max 420MB total
- Weight formats: .pt, .pth, .onnx, .safetensors, .npy

## Hardware
- L4 GPU (24GB VRAM)
- 8GB RAM
- 300 second timeout
- No network access
