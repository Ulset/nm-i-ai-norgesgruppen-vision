# Speed Problem Analysis (2026-03-21)

## The Core Issue
We can't run our best models together within the 300s sandbox timeout.

## Timing Breakdown (per image, 4032x3024 shelf photos)

### Old baseline pipeline (5.5s/img, WORKED, scored 0.8346)
- YOLOv8x@1280 ONNX (FP32, 137MB) + tiling + TTA
- YOLOv8l@640 ONNX (FP32, 88MB) ensemble
- Old FP16 classifier ONNX (16MB) — broken (0.49% val acc) but fast
- Old baked embeddings (from broken classifier)
- The FP16 classifier ran fast AND produced embeddings for reference matching

### Our new pipeline attempts
| Config | Time/img | Score | Issue |
|--------|---------|-------|-------|
| Old pipeline + new YOLO + old clf + new embeddings | 3.5s | 0.8514 | Embedding space mismatch (old clf ≠ new embeddings) |
| New run.py + ONNX clf (FP32) + TTA | 10s | high | Too slow |
| New run.py + PyTorch clf + TTA | 17s | 0.71 | Even slower + no embeddings |
| Old pipeline + new YOLO, no clf | 3.3s | ~0.78 | No classification boost |

## Why the classifier is slow
1. PIL crop + resize (LANCZOS) for each of 500-1000 detections = 1.3s
2. Numpy batch preprocessing (stack + normalize) = 0.9s
3. Classifier inference (FP32 ONNX, 31MB) = 2.2s for 700 crops
4. Total classifier overhead: ~4-5s per image

The OLD classifier (FP16 ONNX, 16MB) was much faster because:
- Half the model size = faster inference
- FP16 operations = faster on GPU
- But it was accuracy-broken (0.49% val acc)

## Why PyTorch classifier didn't help
- PIL crop+resize loop is still the bottleneck (Python, not GPU)
- PyTorch model.forward() is fast, but the data prep dominates
- The PyTorch model only outputs logits (no embeddings), so reference matching doesn't work

## The Embedding Problem
- Old classifier ONNX had dual outputs: logits + 1408-dim embeddings
- Reference embeddings in baked_data.py must match the classifier's embedding space
- Using old clf embeddings + new baked embeddings = mismatched spaces = garbage matching
- Using old clf embeddings + old baked embeddings = matched but classifier is broken

## What Actually Worked Best Locally: 0.8514
Old pipeline (run.py + utils.py) + new YOLOv9e full-data + old FP16 classifier + NEW baked embeddings
- Speed: 3.5s/img
- BUT: on competition test set scored only 0.8336 (embedding mismatch hurt)

## Status as of 2026-03-21 night
- FP32 dual-output classifier ONNX exported (31.3MB) on training-classifier-fix VM
  - Matching reference embeddings built (352/357 categories)
  - BUT: FP32 = too slow (~13s/img)
- FP16 dual-output ONNX failed (mixed type error in Conv nodes)

## Ready for tomorrow morning
- **Safe submission zip** built: old pipeline + old FP16 classifier + old baked_data + new YOLOv9e full-data
  - This has matched embedding spaces (old clf + old embeddings)
  - Previous attempt failed because we used NEW embeddings with OLD classifier
  - Predicted score: should beat 0.8346 thanks to better YOLO

## Solutions to Try Tomorrow
1. **Export fixed classifier as FP16 ONNX with dual outputs (logits + embeddings)**
   - Need to wrap timm model to output both
   - Then convert to FP16 properly (using ultralytics-style half=True export)
   - This gives: fast inference + correct embeddings + matched embedding space

2. **Use old baked embeddings with old classifier** (both from broken model)
   - Speeds matched, but classifier accuracy is bad
   - Score ceiling limited by broken classifier

3. **Skip classifier entirely, optimize YOLO-only**
   - 3.3s/img, can add TTA + tiling
   - Score ~0.78 (no classification boost)

4. **Optimize crop extraction**
   - Replace PIL crop+resize loop with numpy array slicing + cv2.resize (batch)
   - Or limit to top-K detections only

5. **Train a smaller/faster classifier**
   - EfficientNet-B0 instead of B2
   - MobileNetV3 — much faster inference
