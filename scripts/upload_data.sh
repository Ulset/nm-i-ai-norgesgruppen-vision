#!/usr/bin/env bash
# Upload training data and code to GCP VM.
# Usage: bash scripts/upload_data.sh [vm-name]

set -euo pipefail

VM_NAME="${1:-training-vm-1}"
PROJECT="ainm26osl-716"
ZONE="europe-north1-b"
REMOTE_DIR="~/norges-gruppen"

echo "Uploading to $VM_NAME:$REMOTE_DIR"

gcloud compute ssh "$VM_NAME" --project="$PROJECT" --zone="$ZONE" \
  --command="mkdir -p $REMOTE_DIR/data"

gcloud compute scp --recurse \
  training/ scripts/ submission/ dataset.yaml TASK.md \
  "$VM_NAME:$REMOTE_DIR/" \
  --project="$PROJECT" --zone="$ZONE"

gcloud compute scp \
  "Coco Dataset NM NGD.zip" "NM NGD Produktbilder.zip" \
  "$VM_NAME:$REMOTE_DIR/data/" \
  --project="$PROJECT" --zone="$ZONE"

echo ""
echo "Upload complete. SSH in and unzip:"
echo "  cd $REMOTE_DIR/data"
echo "  unzip 'Coco Dataset NM NGD.zip' -d coco_dataset"
echo "  unzip 'NM NGD Produktbilder.zip' -d product_images"
echo ""
echo "Then prepare dataset and train:"
echo "  cd $REMOTE_DIR"
echo "  python -m training.prepare_yolo_dataset"
echo "  python -m training.prepare_crops"
echo "  python -m training.train_yolo --epochs 150 --batch 4"
echo "  python -m training.train_classifier --epochs 50 --batch 64"
