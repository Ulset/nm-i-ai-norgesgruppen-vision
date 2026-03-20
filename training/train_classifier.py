"""Train EfficientNet-B2 product classifier on extracted crops.

Runs on GCP GPU VM with timm==0.9.12.
Expects crops at data/crops/{train,val}/{category_id}/*.jpg

Usage:
  python -m training.train_classifier [--epochs 50] [--batch 64] [--lr 1e-4]
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


NUM_CLASSES = 357
IMG_SIZE = 224


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance in classification tasks.

    Reduces the loss contribution from easy examples and focuses training
    on hard negatives.
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def build_transforms(is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = Counter(dataset.targets)
    total = len(dataset.targets)
    weights = torch.zeros(NUM_CLASSES)
    for cls_idx, count in counts.items():
        weights[cls_idx] = total / (len(counts) * count)
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/crops")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output-dir", default="runs/classifier")
    parser.add_argument("--name", default="efficientnet_b2_shelf")
    parser.add_argument("--focal-loss", action="store_true",
                        help="Use focal loss instead of cross-entropy")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor (default: 0.0)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.ImageFolder(
        Path(args.data_dir) / "train",
        transform=build_transforms(is_train=True),
    )

    # CRITICAL: Force val to use the same class_to_idx as train.
    # ImageFolder assigns indices based on which directories exist,
    # and val has fewer classes than train (278 vs 352), causing
    # completely different index assignments.
    val_dataset = datasets.ImageFolder(
        Path(args.data_dir) / "val",
        transform=build_transforms(is_train=False),
    )
    val_dataset.class_to_idx = train_dataset.class_to_idx
    # Re-map val samples to use train's indices
    val_samples = []
    for path, _ in val_dataset.samples:
        cls_name = Path(path).parent.name
        if cls_name in train_dataset.class_to_idx:
            val_samples.append((path, train_dataset.class_to_idx[cls_name]))
    val_dataset.samples = val_samples
    val_dataset.targets = [s[1] for s in val_samples]

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model = timm.create_model("efficientnet_b2", pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)

    class_weights = compute_class_weights(train_dataset).to(device)
    if args.focal_loss:
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=2.0,
            label_smoothing=args.label_smoothing,
        )
        print(f"Using FocalLoss (gamma=2.0, label_smoothing={args.label_smoothing})")
    elif args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )
        print(f"Using CrossEntropyLoss (label_smoothing={args.label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / max(val_total, 1)
        avg_loss = train_loss / train_total

        print(f"Epoch {epoch+1}/{args.epochs} — loss: {avg_loss:.4f}, "
              f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    torch.save(model.state_dict(), output_dir / "last.pt")

    # Save class mapping (ImageFolder idx → category_id)
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: int(k) for k, v in class_to_idx.items()}
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(idx_to_class, f)

    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Weights: {output_dir / 'best.pt'}")
    print(f"Class mapping: {output_dir / 'class_mapping.json'}")


if __name__ == "__main__":
    main()
