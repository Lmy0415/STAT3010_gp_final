#!/usr/bin/env python3
"""
plot_point_curve.py
Compute Mean IoU / Recall@0.5 for different numbers of foreground points
and save two line charts into Exp_outputs/
"""
import os, json, time, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from segment_anything import sam_model_registry, SamPredictor
import cv2, torch
from utils_plot import savefig                 # ← 已统一输出目录

# ------------------------ config ------------------------
ROOT       = os.path.dirname(__file__)
VAL_DIR    = os.path.join(ROOT, "coco", "val2017")
ANN_FILE   = os.path.join(ROOT, "coco", "annotations", "instances_val2017.json")
CKPT       = os.path.join(ROOT, "checkpoints", "sam_vit_l_0b3195.pth")
DEVICE     = "cuda"

POINT_COUNTS = [1, 2, 3, 5, 9]      # 需要的点数，修改即可
MAX_DIM      = 1024                 # 图像最长边 resize
# --------------------------------------------------------

# helper
def load_img_rgb(path, max_dim=MAX_DIM):
    bgr = cv2.imread(path); h,w = bgr.shape[:2]
    scale = min(1.0, max_dim/max(h,w))
    if scale < 1.0:
        bgr = cv2.resize(bgr,(int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), scale

def fg_points_from_mask(mask, n):
    """返回 mask 内 n 个均匀采样点 (x,y)"""
    ys, xs = np.where(mask)
    idx = np.linspace(0, len(xs)-1, n, dtype=int)
    return np.stack([xs[idx], ys[idx]], axis=1)

# init
coco       = COCO(ANN_FILE)
img_ids    = coco.getImgIds()                 # 全部 5000 张
sam_model  = sam_model_registry["vit_l"](checkpoint=CKPT).to(DEVICE)
predictor  = SamPredictor(sam_model)

mean_ious  = []; recall50s = []

for n_pts in POINT_COUNTS:
    ious = []
    loop = tqdm(img_ids, desc=f"{n_pts}-pt eval")
    for img_id in loop:
        info   = coco.loadImgs(img_id)[0]
        img_p  = os.path.join(VAL_DIR, info['file_name'])
        rgb, s = load_img_rgb(img_p)
        predictor.set_image(rgb)

        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            gt = coco.annToMask(ann).astype(bool)
            if s < 1.0:
                gt = cv2.resize(gt.astype(np.uint8),
                                (rgb.shape[1], rgb.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            if not gt.any(): continue

            pts  = fg_points_from_mask(gt, n_pts)
            labs = np.ones(n_pts, dtype=int)
            masks, scores, _ = predictor.predict(
                point_coords=pts,
                point_labels=labs,
                multimask_output=True
            )
            best = int(np.argmax(scores)); pred = masks[best]

            inter = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            if union:
                ious.append(inter/union)
        loop.set_postfix(iou=np.mean(ious) if ious else 0)

    mean_ious.append(np.mean(ious))
    recall50s.append(np.mean(np.array(ious) >= .5))
    print(f"{n_pts}-pt  mIoU={mean_ious[-1]:.3f}  R50={recall50s[-1]:.3f}  Samples={len(ious)}")

# ---------------------- plot ----------------------
# mIoU
plt.figure(figsize=(5,4))
plt.plot(POINT_COUNTS, mean_ious, marker='o')
plt.xlabel("Number of Points"); plt.ylabel("Mean IoU")
plt.title("Mean IoU vs Number of Points")
plt.ylim(0,1); savefig("mean_iou_vs_points.png")

# Recall@0.5
plt.figure(figsize=(5,4))
plt.plot(POINT_COUNTS, recall50s, marker='o', color="tab:orange")
plt.xlabel("Number of Points"); plt.ylabel("Recall @ IoU≥0.5")
plt.title("Recall@0.5 vs Number of Points")
plt.ylim(0,1); savefig("recall50_vs_points.png")

print("[Done] 曲线已保存到 Exp_outputs/")
