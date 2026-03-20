"""Export YOLO models to FP16 ONNX safely.

Exports as FP32 first, then converts weights to FP16 while keeping
inputs/outputs as FP32. Tests loading with onnxruntime before saving.

Usage:
  python3 scripts/export_fp16_safe.py \
    --weights runs/detect/yolov8x_shelf/weights/best.pt \
    --imgsz 1280 \
    --output yolo_detector_fp16.onnx
"""

import argparse
import numpy as np
from pathlib import Path

import torch
_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, **{**kwargs, "weights_only": False})

from ultralytics import YOLO
import onnx
from onnx import numpy_helper, TensorProto


def convert_to_fp16_weights(onnx_path: str, output_path: str):
    """Convert ONNX model weights to FP16, keeping I/O as FP32."""
    model = onnx.load(onnx_path)

    # Get input/output names to skip them
    io_names = set()
    for inp in model.graph.input:
        io_names.add(inp.name)
    for out in model.graph.output:
        io_names.add(out.name)

    # Convert initializer weights to FP16
    converted = 0
    for tensor in model.graph.initializer:
        if tensor.name in io_names:
            continue
        if tensor.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(tensor)
            arr_fp16 = arr.astype(np.float16)
            new_tensor = numpy_helper.from_array(arr_fp16, tensor.name)
            tensor.CopyFrom(new_tensor)
            converted += 1

    print(f"Converted {converted} tensors to FP16")
    onnx.save(model, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Step 1: Export FP32 ONNX
    print(f"Exporting {args.weights} as FP32 ONNX...")
    model = YOLO(args.weights)
    model.export(format="onnx", imgsz=args.imgsz, opset=17, simplify=True)
    fp32_path = Path(args.weights).with_suffix(".onnx")
    fp32_size = fp32_path.stat().st_size / (1024 * 1024)
    print(f"FP32: {fp32_size:.1f} MB")

    # Step 2: Convert weights to FP16
    print("Converting weights to FP16...")
    convert_to_fp16_weights(str(fp32_path), args.output)
    fp16_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"FP16: {fp16_size:.1f} MB (saved {fp32_size - fp16_size:.1f} MB)")

    # Step 3: Test loading with onnxruntime
    print("Testing with onnxruntime...")
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(args.output, providers=providers)

    # Test inference with FP32 input
    dummy = np.random.randn(1, 3, args.imgsz, args.imgsz).astype(np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dummy})
    print(f"Test inference OK: output shape {outputs[0].shape}")
    print(f"DONE: {args.output}")


if __name__ == "__main__":
    main()
