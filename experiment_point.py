#!/usr/bin/env python3
# experiment_point.py
# Single-point Zero-shot segmentation on COCO val2017
import os, json, cv2, torch, numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from segment_anything import sam_model_registry, SamPredictor

# -------- Paths --------------------------------------------------
ROOT       = os.path.dirname(__file__)           # /home/segment-anything
COCO_DIR   = os.path.join(ROOT, "coco")
VAL_IMGDIR = os.path.join(COCO_DIR, "val2017")
ANN_FILE   = os.path.join(COCO_DIR, "annotations",
                          "instances_val2017.json")
CKPT   = os.path.join(ROOT, "checkpoints", "sam_vit_l_0b3195.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------------------------

def load_rgb(path, max_dim=1024):
    bgr = cv2.imread(path)
    h, w = bgr.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(w*s), int(h*s)),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

coco = COCO(ANN_FILE)
img_ids = coco.getImgIds()

predictor = SamPredictor(
    sam_model_registry["vit_l"](checkpoint=CKPT).to(DEVICE)
)

records, ious = [], []
for img_id in tqdm(img_ids, desc="Single-point"):
    info = coco.loadImgs(img_id)[0]
    rgb  = load_rgb(os.path.join(VAL_IMGDIR, info["file_name"]))
    predictor.set_image(rgb)

    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
        gt = coco.annToMask(ann).astype(bool)
        if not gt.any(): continue
        dist = cv2.distanceTransform(gt.astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        masks, scores, _ = predictor.predict(
            point_coords = np.array([[x, y], [x2, y2], [bx, by]])
            point_labels = np.array([1, 1, 0])
            multimask_output=True
        )
        k   = int(np.argmax(scores))
        pm  = masks[k]
        iou = np.logical_and(pm, gt).sum() / np.logical_or(pm, gt).sum()
        ious.append(iou)
        records.append({"image_id": img_id, "ann_id": ann["id"], "iou": iou})

miou      = float(np.mean(ious))
recall50  = float(np.mean([i >= .5 for i in ious]))
print(f"mIoU={miou:.3f}   Recall@0.5={recall50:.3f}   masks={len(ious)}")

with open("point_iou_records.json", "w") as f:
    json.dump(records, f)
print("Saved point_iou_records.json")
