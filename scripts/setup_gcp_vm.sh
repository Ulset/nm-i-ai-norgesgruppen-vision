#!/usr/bin/env bash
# Provision a GPU VM on GCP for model training.
# Usage: bash scripts/setup_gcp_vm.sh [vm-name]

set -euo pipefail

VM_NAME="${1:-training-vm-1}"
PROJECT="ainm26osl-716"
ZONE="europe-north1-b"
MACHINE_TYPE="g2-standard-16"
ACCELERATOR="type=nvidia-l4,count=1"
IMAGE_FAMILY="pytorch-2-6-cu124"
IMAGE_PROJECT="deeplearning-platform-release"
DISK_SIZE="100GB"

echo "Creating VM: $VM_NAME"
echo "  Project: $PROJECT"
echo "  Zone: $ZONE"
echo "  Machine: $MACHINE_TYPE"
echo "  GPU: L4"

gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="$ACCELERATOR" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$DISK_SIZE" \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"

echo ""
echo "VM created. Connect with:"
echo "  gcloud compute ssh $VM_NAME --project=$PROJECT --zone=$ZONE"
echo ""
echo "After connecting, run:"
echo "  pip install ultralytics==8.1.0 timm==0.9.12 onnx"
