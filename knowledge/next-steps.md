# Next Steps (as of 2026-03-20 ~19:15 UTC)

## Waiting On
1. YOLO11x@1280 training to finish (epoch ~102/300, VM-1)
2. YOLOv9e@1280 training to finish (epoch ~88/300, VM-2)
3. EfficientNet-B2 v2 training to finish (epoch ~50/100, VM-4)

## TODO When Training Finishes
1. **Export models to ONNX FP16** on each VM
   - Validate output shape has 361 dimension (4+357)
   - Validate opset ≤ 20
   - Check file sizes fit within 420MB budget
2. **Download best models** to local machine
   - `gcloud compute scp` from each VM
3. **Rebuild reference embeddings** from fixed classifier
   - Run build_reference_embeddings.py with new model
   - Bake into baked_data.py
4. **Fix DINOv2 ONNX export** (device mismatch error)
   - Or decide to skip DINOv2 if classifier is good enough at 90%+ accuracy
5. **Build submission package**
   - Pick best 2 detectors + classifier (3 weight files)
   - Run build_submission.sh
   - Verify size < 420MB
6. **Test locally** with evaluate_local.py before submitting
7. **Submit** (6 attempts per day, resets midnight UTC)

## Inference Tuning (VM-6 — not yet started)
- Lower CLASSIFIER_THRESHOLD from 0.7 → 0.5
- Lower REFERENCE_THRESHOLD from 0.8 → 0.6
- Test WBF_IOU_THR values: 0.45, 0.50, 0.55, 0.60
- Test CONFIDENCE_FLOOR values: 0.01, 0.03, 0.05
- config.json support already added to run.py

## Stop VMs When Done
All 6 GPU VMs cost ~$5.88/hr total. Stop them once training/export is complete:
```bash
ACCT="--account=devstar7161@gcplab.me"
PROJ="--project=ainm26osl-716"
ZONE="europe-west1-c"
for vm in training-vm-1 training-vm-2 training-yolo11x-640 training-classifier-fix training-dinov2-embeds training-inference-tune; do
    gcloud compute instances stop "$vm" --zone=$ZONE $ACCT $PROJ &
done
wait
```
