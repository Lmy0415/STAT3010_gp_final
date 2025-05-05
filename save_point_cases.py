#!/usr/bin/env python3
"""
Save top-N & bottom-N single-point cases into a pickle
Requires:
  - coco/val2017/...
  - coco/annotations/instances_val2017.json
  - point_iou_records.json  (由 experiment_point.py 生成)
Output:
  point_cases.pkl   dict(keys: top, bottom) → list of dict
"""
import os, json, pickle, cv2, numpy as np
from pycocotools.coco import COCO
from segment_anything import sam_model_registry, SamPredictor

ROOT      = os.path.dirname(__file__)
COCO_DIR  = os.path.join(ROOT, "coco")
VAL_DIR   = os.path.join(COCO_DIR, "val2017")
ANN_FILE  = os.path.join(COCO_DIR, "annotations", "instances_val2017.json")
CKPT      = os.path.join(ROOT, "checkpoints", "sam_vit_l_0b3195.pth")
DEVICE    = "cuda"

# ----- Load IoU records -----
records = json.load(open("point_iou_records.json"))
records = sorted(records, key=lambda r: r["iou"])   # ascending

TOP_K    = 5
BOTTOM_K = 5
case_ids = [*records[:BOTTOM_K], *records[-TOP_K:]]

# ----- Init helpers -----
coco       = COCO(ANN_FILE)
predictor  = SamPredictor(
    sam_model_registry["vit_l"](checkpoint=CKPT).to(DEVICE)
)
def load_rgb(path, max_dim=1024):
    bgr = cv2.imread(path)
    h, w = bgr.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(w*s), int(h*s)))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

cases = {"top": [], "bottom": []}
for rec, key in zip(
        [*records[-TOP_K:], *records[:BOTTOM_K]],
        ["top"] * TOP_K + ["bottom"] * BOTTOM_K):
    img_info = coco.loadImgs(rec["image_id"])[0]
    img_path = os.path.join(VAL_DIR, img_info["file_name"])
    rgb = load_rgb(img_path)
    ann = coco.loadAnns(rec["ann_id"])[0]
    gt_mask = coco.annToMask(ann).astype(bool)

    # regenerate pred mask (optional, but ensures same shape)
    h, w = rgb.shape[:2]
    point = np.array([[w//2, h//2]])  # dummy; we won't use
    predictor.set_image(rgb)
    # NOTE: you can skip re-predict and just store IoU; this is for viz
    pred_mask = np.zeros_like(gt_mask)

    cases[key].append(dict(
        img_path = img_path,
        gt_mask  = gt_mask,
        pred_mask = pred_mask,
        iou      = rec["iou"]
    ))

with open("point_cases.pkl", "wb") as f:
    pickle.dump(cases, f)
print("Saved point_cases.pkl  (top & bottom %d cases)" % TOP_K)
