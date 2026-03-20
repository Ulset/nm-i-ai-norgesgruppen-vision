#!/usr/bin/env python3
"""Extract DINOv2 reference embeddings and export the backbone to ONNX (FP16).

For each product category directory, loads all reference images, extracts
embeddings with DINOv2, computes an L2-normalised mean embedding, and saves
a (357, 384) float16 matrix.  Also exports the DINOv2 model to ONNX and
converts to FP16.

Usage:
  python scripts/dinov2_embeddings.py \
      [--product-images data/product_images] \
      [--output-dir runs/dinov2] \
      [--model dinov2_vits14]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

NUM_CLASSES = 357
EMBED_DIM = 384  # dinov2_vits14 output dimension
IMG_SIZE = 224


def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_dinov2(model_name: str, device: torch.device) -> torch.nn.Module:
    """Load DINOv2 model from torch.hub."""
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    image_dir: Path,
    transform: transforms.Compose,
    device: torch.device,
) -> np.ndarray:
    """Extract and L2-normalise the mean embedding for images in a directory."""
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not image_paths:
        return None

    embeddings = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        emb = model(tensor)  # (1, EMBED_DIM)
        embeddings.append(emb.cpu().numpy())

    mean_emb = np.concatenate(embeddings, axis=0).mean(axis=0)
    # L2 normalise
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb /= norm
    return mean_emb


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    device: torch.device,
) -> None:
    """Export DINOv2 to ONNX and convert weights to FP16."""
    import onnx
    from onnx import numpy_helper

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    onnx_fp32_path = output_path.with_suffix(".fp32.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_fp32_path),
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "embeddings": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"ONNX FP32 exported: {onnx_fp32_path} "
          f"({onnx_fp32_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Convert to FP16
    onnx_model = onnx.load(str(onnx_fp32_path))
    for tensor in onnx_model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT:
            arr = numpy_helper.to_array(tensor).astype(np.float16)
            new_tensor = numpy_helper.from_array(arr, tensor.name)
            tensor.CopyFrom(new_tensor)
            tensor.data_type = onnx.TensorProto.FLOAT16

    onnx.save(onnx_model, str(output_path))
    print(f"ONNX FP16 saved: {output_path} "
          f"({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Clean up FP32 intermediate
    onnx_fp32_path.unlink()


def validate_onnx(onnx_path: Path) -> None:
    """Run dummy inference through the exported ONNX model."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    outputs = session.run(None, {"images": dummy})
    print(f"ONNX validation — output shape: {outputs[0].shape}, "
          f"file size: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 reference embeddings and export ONNX model")
    parser.add_argument("--product-images", default="data/product_images",
                        help="Root directory of product reference images")
    parser.add_argument("--output-dir", default="runs/dinov2",
                        help="Output directory for embeddings and ONNX model")
    parser.add_argument("--model", default="dinov2_vits14",
                        help="DINOv2 model variant (default: dinov2_vits14)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    product_dir = Path(args.product_images)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading {args.model}...")
    model = load_dinov2(args.model, device)
    transform = get_transform()

    # Build embedding matrix
    embedding_matrix = np.zeros((NUM_CLASSES, EMBED_DIM), dtype=np.float16)
    found_classes = 0

    category_dirs = sorted(
        d for d in product_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    print(f"Found {len(category_dirs)} category directories")

    for cat_dir in category_dirs:
        cat_id = int(cat_dir.name)
        if cat_id >= NUM_CLASSES:
            print(f"  Skipping category {cat_id} (>= {NUM_CLASSES})")
            continue

        emb = extract_embeddings(model, cat_dir, transform, device)
        if emb is not None:
            embedding_matrix[cat_id] = emb.astype(np.float16)
            found_classes += 1

    emb_path = output_dir / "dinov2_reference_embeddings.npy"
    np.save(str(emb_path), embedding_matrix)
    print(f"Saved embedding matrix: {emb_path} — "
          f"shape={embedding_matrix.shape}, dtype={embedding_matrix.dtype}, "
          f"classes with embeddings: {found_classes}/{NUM_CLASSES}")

    # Export ONNX (FP16)
    onnx_path = output_dir / "dinov2_vits14.onnx"
    print("Exporting DINOv2 to ONNX (FP16)...")
    export_onnx(model, onnx_path, device)

    # Validate
    print("Validating ONNX export...")
    validate_onnx(onnx_path)

    print("Done.")


if __name__ == "__main__":
    main()
