#!/usr/bin/env bash
# Upload training code and project files to a GCP VM.
# Usage: bash scripts/upload_data_to_vm.sh <vm-name>

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <vm-name>"
  exit 1
fi

VM_NAME="$1"
ACCOUNT="devstar7161@gcplab.me"
PROJECT="ainm26osl-716"
ZONE="europe-west1-c"
REMOTE_DIR="~/norges-gruppen"

echo "Uploading project files to $VM_NAME:$REMOTE_DIR"

gcloud compute scp --recurse \
  training/ scripts/ submission/ CLAUDE.md \
  "$VM_NAME:$REMOTE_DIR/" \
  --account="$ACCOUNT" \
  --project="$PROJECT" \
  --zone="$ZONE"

echo ""
echo "Upload complete."
echo ""
echo "NOTE: The data/ directory must be copied separately from an existing VM"
echo "that already has the dataset. For example:"
echo "  gcloud compute scp --recurse <source-vm>:~/norges-gruppen/data/ $VM_NAME:~/norges-gruppen/data/ \\"
echo "    --account=$ACCOUNT --project=$PROJECT --zone=$ZONE"
