#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COCO val2017  (前 500 张)：
  点数 = [1,2,3,5,9] → 统计
    • 普通  Top-1 mask   mIoU / Recall@0.5
    • Oracle(3 masks里与GT最接近)  mIoU / Recall@0.5
  结果图保存到 Exp_outputs/
"""
import os, cv2, json, numpy as np, torch, tqdm, matplotlib.pyplot as plt
from pycocotools.coco import COCO
from segment_anything import SamPredictor, sam_model_registry

# ---------- 路径 ----------
ROOT      = "/home/segment-anything"
DATA      = os.path.join(ROOT, "coco")                       # <-- 您的 coco 在这里
VAL_IMG   = os.path.join(DATA, "val2017")
ANN       = os.path.join(DATA, "annotations/instances_val2017.json")
CKPT      = os.path.join(ROOT, "checkpoints/sam_vit_l_0b3195.pth")  # <-- 权重
OUT_DIR   = os.path.join(ROOT, "Exp_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 初始化 ----------
device     = "cuda" if torch.cuda.is_available() else "cpu"
sam        = sam_model_registry["vit_l"](checkpoint=CKPT).to(device)
predictor  = SamPredictor(sam)
coco       = COCO(ANN)

img_ids     = coco.getImgIds()[:500]             # 抽前 500 张快速跑
POINT_LIST  = [1, 2, 3, 5, 9]
metrics = {k: [] for k in
           ["mIoU", "oracle_mIoU", "recall50", "oracle_recall50"]}

def load_rgb(path, max_dim=1024):
    bgr = cv2.imread(path);  h, w = bgr.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), s

for pnum in POINT_LIST:
    ious, ious_oracle = [], []
    for img_id in tqdm.tqdm(img_ids, desc=f"{pnum} pts"):
        info   = coco.loadImgs(img_id)[0]
        rgb, s = load_rgb(os.path.join(VAL_IMG, info["file_name"]))
        predictor.set_image(rgb)

        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            gt = coco.annToMask(ann).astype(bool)
            if s < 1.0:   # resize GT to predictor size
                gt = cv2.resize(gt.astype(np.uint8),
                                (rgb.shape[1], rgb.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            if not gt.any():
                continue

            # ---------- 采样前景点 ----------
            ys, xs = np.where(gt)
            k      = min(pnum, len(xs))                 # 至多 len(xs) 个
            # 若像素点不足，则允许 replace=True，避免 ValueError
            replace_flag = len(xs) < pnum
            idx    = np.random.choice(len(xs), size=k, replace=replace_flag)
            points = np.stack([xs[idx], ys[idx]], axis=1)
            labels = np.ones(k, dtype=int)

            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )

            # Top-1 结果
            best_idx     = int(np.argmax(scores))
            best_mask    = masks[best_idx]
            best_iou     = np.logical_and(best_mask, gt).sum() / np.logical_or(best_mask, gt).sum()
            ious.append(best_iou)

            # Oracle 3-mask 结果
            iou_all      = [(np.logical_and(m, gt).sum() / np.logical_or(m, gt).sum()) for m in masks]
            ious_oracle.append(max(iou_all))

    # 统计
    metrics["mIoU"].append(np.mean(ious))
    metrics["oracle_mIoU"].append(np.mean(ious_oracle))
    metrics["recall50"].append(np.mean(np.array(ious)         >= 0.5))
    metrics["oracle_recall50"].append(np.mean(np.array(ious_oracle) >= 0.5))

# ---------- 画图 ----------
plt.figure()
plt.plot(POINT_LIST, metrics["mIoU"],          "o-", label="Top-1 mask")
plt.plot(POINT_LIST, metrics["oracle_mIoU"],   "o--",label="Oracle mask")
plt.xlabel("# Points"); plt.ylabel("Mean IoU"); plt.ylim(0,1)
plt.title("Mean IoU vs #Points (500 COCO imgs)"); plt.grid(True); plt.legend()
plt.savefig(os.path.join(OUT_DIR, "mean_iou_vs_points_oracle.png"), dpi=150)
plt.figure()
plt.plot(POINT_LIST, metrics["recall50"],          "o-", label="Top-1 mask")
plt.plot(POINT_LIST, metrics["oracle_recall50"],   "o--",label="Oracle mask")
plt.xlabel("# Points"); plt.ylabel("Recall @ IoU≥0.5"); plt.ylim(0,1)
plt.title("Recall@0.5 vs #Points (500 COCO imgs)"); plt.grid(True); plt.legend()
plt.savefig(os.path.join(OUT_DIR, "recall50_vs_points_oracle.png"), dpi=150)

print("[Done] 结果已写入 Exp_outputs/")
