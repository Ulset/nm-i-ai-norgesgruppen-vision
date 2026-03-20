#!/usr/bin/env bash
# Provision a GPU training VM on GCP with all dependencies pre-installed.
# Usage: bash scripts/setup_training_vm.sh <vm-name>

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <vm-name>"
  exit 1
fi

VM_NAME="$1"
ACCOUNT="devstar7161@gcplab.me"
PROJECT="ainm26osl-716"
ZONE="europe-west1-c"
MACHINE_TYPE="g2-standard-16"
ACCELERATOR="type=nvidia-l4,count=1"
IMAGE_FAMILY="pytorch-2-6-cu124"
IMAGE_PROJECT="deeplearning-platform-release"
DISK_SIZE="200GB"
DISK_TYPE="pd-balanced"

echo "Creating VM: $VM_NAME"
echo "  Account: $ACCOUNT"
echo "  Project: $PROJECT"
echo "  Zone:    $ZONE"
echo "  Machine: $MACHINE_TYPE"
echo "  GPU:     L4"
echo "  Disk:    $DISK_SIZE ($DISK_TYPE)"

gcloud compute instances create "$VM_NAME" \
  --account="$ACCOUNT" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="$ACCELERATOR" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$DISK_SIZE" \
  --boot-disk-type="$DISK_TYPE" \
  --maintenance-policy=TERMINATE \
  --scopes=storage-full \
  --metadata="install-nvidia-driver=True"

echo ""
echo "VM created. Waiting 30s for boot..."
sleep 30

echo "Installing Python packages on VM..."
gcloud compute ssh "$VM_NAME" \
  --account="$ACCOUNT" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --command="pip install ultralytics timm==0.9.12 onnx onnxsim opencv-python-headless"

echo "Creating project directory on VM..."
gcloud compute ssh "$VM_NAME" \
  --account="$ACCOUNT" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --command="mkdir -p ~/norges-gruppen"

echo ""
echo "VM $VM_NAME is ready. Connect with:"
echo "  gcloud compute ssh $VM_NAME --account=$ACCOUNT --project=$PROJECT --zone=$ZONE"
