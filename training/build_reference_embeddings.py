"""Pre-compute reference embeddings from product images using trained classifier.

Usage:
  python -m training.build_reference_embeddings \
    --model-weights runs/classifier/efficientnet_b2_shelf/best.pt \
    --class-mapping runs/classifier/efficientnet_b2_shelf/class_mapping.json \
    --product-images data/product_images \
    --annotations data/coco_dataset/train/annotations.json \
    --output reference_embeddings.npy
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import timm
from torchvision import transforms

from training.data_utils import (
    load_coco_annotations,
    load_product_metadata,
    build_product_category_mapping,
)

NUM_CLASSES = 357
IMG_SIZE = 224
EMBED_DIM = 1408  # EfficientNet-B2 penultimate layer


def get_feature_extractor(model: torch.nn.Module) -> torch.nn.Module:
    model.classifier = torch.nn.Identity()
    return model


def embed_images(
    model: torch.nn.Module,
    image_paths: list[Path],
    device: torch.device,
) -> np.ndarray:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = []
    model.eval()

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            embedding = model(tensor).cpu().numpy().flatten()
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
            embeddings.append(embedding)

    return np.array(embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-weights", required=True)
    parser.add_argument("--class-mapping", required=True)
    parser.add_argument("--product-images", default="data/product_images")
    parser.add_argument("--annotations", default="data/coco_dataset/train/annotations.json")
    parser.add_argument("--output", default="reference_embeddings.npy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(args.model_weights, map_location=device)
    model.load_state_dict(state_dict)
    model = get_feature_extractor(model)
    model = model.to(device)

    with open(args.class_mapping) as f:
        idx_to_cat = json.load(f)
    idx_to_cat = {int(k): int(v) for k, v in idx_to_cat.items()}

    annotations = load_coco_annotations(Path(args.annotations))
    metadata = load_product_metadata(Path(args.product_images) / "metadata.json")
    product_mapping = build_product_category_mapping(annotations, metadata)

    reference_embeddings = np.zeros((NUM_CLASSES, EMBED_DIM), dtype=np.float16)
    product_dir = Path(args.product_images)

    category_count = 0
    for product_code, cat_id in product_mapping.items():
        prod_path = product_dir / product_code
        if not prod_path.is_dir():
            continue

        priority_names = ["main.jpg", "front.jpg"]
        image_paths = []
        for name in priority_names:
            p = prod_path / name
            if p.exists():
                image_paths.append(p)

        for p in sorted(prod_path.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p not in image_paths:
                image_paths.append(p)

        if not image_paths:
            continue

        embeddings = embed_images(model, image_paths, device)
        avg_embedding = embeddings.mean(axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 1e-8:
            avg_embedding = avg_embedding / norm

        reference_embeddings[cat_id] = avg_embedding.astype(np.float16)
        category_count += 1

    np.save(args.output, reference_embeddings)
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Saved {args.output} ({size_mb:.1f} MB)")
    print(f"Embedded {category_count} categories out of {NUM_CLASSES}")


if __name__ == "__main__":
    main()
