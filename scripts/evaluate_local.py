"""Local evaluation script replicating competition scoring.

Usage:
  python -m scripts.evaluate_local \
    --predictions predictions.json \
    --ground-truth data/coco_dataset/train/annotations.json \
    --val-image-ids data/yolo_dataset/val_image_ids.json
"""

import argparse
import json
import copy
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_composite_score(det_map: float, cls_map: float) -> float:
    return 0.7 * det_map + 0.3 * cls_map


def evaluate_detection_map(coco_gt: COCO, predictions: list[dict], image_ids: set[int] | None = None) -> float:
    det_preds = []
    for p in predictions:
        det_preds.append({**p, "category_id": 0})

    gt_copy = copy.deepcopy(coco_gt.dataset)
    for ann in gt_copy["annotations"]:
        ann["category_id"] = 0
    gt_copy["categories"] = [{"id": 0, "name": "product", "supercategory": "product"}]

    det_coco_gt = COCO()
    det_coco_gt.dataset = gt_copy
    det_coco_gt.createIndex()

    coco_dt = det_coco_gt.loadRes(det_preds) if det_preds else det_coco_gt.loadRes([])
    coco_eval = COCOeval(det_coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = [0.5]
    if image_ids:
        coco_eval.params.imgIds = sorted(image_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]


def evaluate_classification_map(coco_gt: COCO, predictions: list[dict], image_ids: set[int] | None = None) -> float:
    if not predictions:
        return 0.0

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = [0.5]
    if image_ids:
        coco_eval.params.imgIds = sorted(image_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--val-image-ids", help="JSON file with list of val image IDs")
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)

    coco_gt = COCO(args.ground_truth)

    val_ids = None
    if args.val_image_ids:
        with open(args.val_image_ids) as f:
            val_ids = set(json.load(f))
        predictions = [p for p in predictions if p["image_id"] in val_ids]

    det_map = evaluate_detection_map(coco_gt, predictions, image_ids=val_ids)
    cls_map = evaluate_classification_map(coco_gt, predictions, image_ids=val_ids)
    composite = compute_composite_score(det_map, cls_map)

    print(f"\n{'='*50}")
    print(f"Detection mAP@0.5:       {det_map:.4f}")
    print(f"Classification mAP@0.5:  {cls_map:.4f}")
    print(f"Composite Score:         {composite:.4f}")
    print(f"  (0.7 x {det_map:.4f} + 0.3 x {cls_map:.4f})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
