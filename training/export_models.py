"""Export trained models to ONNX FP16 for submission.

Supports ensemble export (multiple YOLO models).
Bakes reference embeddings + class mapping into a single .npy file.

Usage:
  python -m training.export_models \
    --yolo-weights runs/detect/yolov8x_shelf/weights/best.pt \
    --yolo-l-weights runs/detect/yolov8l_640/weights/best.pt \
    --clf-weights runs/classifier/efficientnet_b2_shelf/best.pt \
    --class-mapping runs/classifier/efficientnet_b2_shelf/class_mapping.json \
    --reference-embeddings submission/reference_embeddings.npy \
    --output-dir submission/
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
# PyTorch 2.6+ defaults weights_only=True which breaks ultralytics 8.1.0 model loading.
_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, **{**kwargs, "weights_only": False})

import timm
from ultralytics import YOLO

NUM_CLASSES = 357
IMG_SIZE_CLF = 224


def export_yolo(weights_path: str, output_path: Path, imgsz: int = 1280):
    """Export YOLO to ONNX. FP16 conversion done on GPU, falls back to FP32 on CPU."""
    model = YOLO(weights_path)

    # Try GPU export for FP16, fall back to CPU FP32
    device = "0" if torch.cuda.is_available() else "cpu"
    use_half = torch.cuda.is_available()

    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=17,
        half=use_half,
        simplify=True,
        device=device,
    )

    onnx_src = Path(weights_path).with_suffix(".onnx")

    # If exported as FP32 (CPU), convert to FP16 manually
    if not use_half:
        import onnx
        from onnx import numpy_helper
        onnx_model = onnx.load(str(onnx_src))
        for tensor in onnx_model.graph.initializer:
            if tensor.data_type == onnx.TensorProto.FLOAT:
                arr = numpy_helper.to_array(tensor).astype("float16")
                new_tensor = numpy_helper.from_array(arr, tensor.name)
                tensor.CopyFrom(new_tensor)
        onnx.save(onnx_model, str(onnx_src))

    shutil.move(str(onnx_src), str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"YOLO exported: {output_path} ({size_mb:.1f} MB)")


def export_classifier(weights_path: str, output_dir: Path):
    """Export EfficientNet-B2 to ONNX FP16 with dual output (logits + embeddings)."""
    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    class DualOutputModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.features = base_model
            self.classifier = base_model.classifier
            base_model.classifier = torch.nn.Identity()

        def forward(self, x):
            embeddings = self.features(x)
            logits = self.classifier(embeddings)
            return logits, embeddings

    dual_model = DualOutputModel(model)
    dual_model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE_CLF, IMG_SIZE_CLF)

    dest = output_dir / "classifier.onnx"
    torch.onnx.export(
        dual_model,
        dummy_input,
        str(dest),
        opset_version=17,
        input_names=["input"],
        output_names=["logits", "embeddings"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "embeddings": {0: "batch_size"},
        },
    )

    # Convert to FP16
    import onnx
    from onnx import numpy_helper
    onnx_model = onnx.load(str(dest))
    for tensor in onnx_model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT:
            arr = numpy_helper.to_array(tensor).astype("float16")
            new_tensor = numpy_helper.from_array(arr, tensor.name)
            tensor.CopyFrom(new_tensor)
    onnx.save(onnx_model, str(dest))

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Classifier exported: {dest} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-weights", required=True, help="YOLOv8x best.pt")
    parser.add_argument("--yolo-l-weights", default=None, help="YOLOv8l best.pt (optional ensemble)")
    parser.add_argument("--clf-weights", required=True, help="EfficientNet-B2 best.pt")
    parser.add_argument("--class-mapping", default=None, help="class_mapping.json path")
    parser.add_argument("--reference-embeddings", default=None, help="reference_embeddings.npy path")
    parser.add_argument("--output-dir", default="submission")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export YOLOv8x
    export_yolo(args.yolo_weights, output_dir / "yolo_detector.onnx", imgsz=1280)

    # Export YOLOv8l (if available)
    if args.yolo_l_weights:
        export_yolo(args.yolo_l_weights, output_dir / "yolo_l_detector.onnx", imgsz=640)

    # Export classifier
    export_classifier(args.clf_weights, output_dir)

    # Copy class_mapping.json and reference_embeddings.npy
    if args.class_mapping and Path(args.class_mapping).exists():
        shutil.copy2(args.class_mapping, output_dir / "class_mapping.json")
    if args.reference_embeddings and Path(args.reference_embeddings).exists():
        shutil.copy2(args.reference_embeddings, output_dir / "reference_embeddings.npy")

    # Summary
    weight_extensions = (".onnx", ".npy", ".pt", ".pth", ".safetensors")
    weight_files = [f for f in output_dir.iterdir() if f.suffix in weight_extensions]
    total_size = sum(f.stat().st_size for f in weight_files) / (1024 * 1024)
    print(f"\nWeight files: {len(weight_files)} / 3 limit")
    for f in sorted(weight_files):
        print(f"  {f.name}: {f.stat().st_size / (1024*1024):.1f} MB")
    print(f"Total weight size: {total_size:.1f} MB / 420 MB limit")

    if len(weight_files) > 3:
        print("WARNING: More than 3 weight files! Need to consolidate.")
    if total_size > 420:
        print("WARNING: Total weight size exceeds 420 MB limit!")


if __name__ == "__main__":
    main()
