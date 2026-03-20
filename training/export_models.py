"""Export trained models to ONNX FP16 for submission.

Usage:
  python -m training.export_models \
    --yolo-weights runs/detect/yolov8x_shelf/weights/best.pt \
    --clf-weights runs/classifier/efficientnet_b2_shelf/best.pt \
    --output-dir submission/
"""

import argparse
from pathlib import Path

import torch
# PyTorch 2.6+ defaults weights_only=True which breaks ultralytics 8.1.0 model loading.
_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, **{**kwargs, "weights_only": False})

import timm
from ultralytics import YOLO

NUM_CLASSES = 357
IMG_SIZE_CLF = 224


def export_yolo(weights_path: str, output_dir: Path):
    model = YOLO(weights_path)
    model.export(
        format="onnx",
        imgsz=1280,
        opset=17,
        half=True,
        simplify=True,
    )
    onnx_path = Path(weights_path).with_suffix(".onnx")
    dest = output_dir / "yolo_detector.onnx"
    onnx_path.rename(dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"YOLO exported: {dest} ({size_mb:.1f} MB)")


def export_classifier(weights_path: str, output_dir: Path):
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
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--clf-weights", required=True)
    parser.add_argument("--output-dir", default="submission")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_yolo(args.yolo_weights, output_dir)
    export_classifier(args.clf_weights, output_dir)

    total_size = sum(
        f.stat().st_size for f in output_dir.iterdir()
        if f.suffix in (".onnx", ".npy")
    ) / (1024 * 1024)
    print(f"\nTotal weight size: {total_size:.1f} MB / 420 MB limit")


if __name__ == "__main__":
    main()
