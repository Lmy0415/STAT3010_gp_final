#!/usr/bin/env python3
"""
plot_point_curve_by_size_sample500.py
– 仅随机抽 500 张 val2017
– 评估 small / medium / large 三条曲线
– 结果存 Exp_outputs/
"""
import os, random, numpy as np, matplotlib.pyplot as plt, cv2, torch
from tqdm import tqdm
from pycocotools.coco import COCO
from segment_anything import sam_model_registry, SamPredictor
from utils_plot import savefig

# ---------- 配置 ----------
ROOT      = os.path.dirname(__file__)
VAL_DIR   = os.path.join(ROOT, "coco", "val2017")
ANN_FILE  = os.path.join(ROOT, "coco", "annotations",
                         "instances_val2017.json")
CKPT      = os.path.join(ROOT, "checkpoints", "sam_vit_l_0b3195.pth")
DEVICE        = "cuda"

POINT_COUNTS  = [1, 3, 5, 9]
MAX_DIM       = 640          # 与之前保持一致
SAMPLE_N      = 500          # ← 只抽 500 张
RAND_SEED     = 42           # 复现实验
# ---------------------------

random.seed(RAND_SEED)

def load_rgb(path):
    bgr = cv2.imread(path); h,w = bgr.shape[:2]
    s = min(1.0, MAX_DIM/max(h,w))
    if s < 1.0:
        bgr = cv2.resize(bgr,(int(w*s),int(h*s)),interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), s

def fg_points(mask, n):
    ys,xs = np.where(mask)
    idx   = np.linspace(0,len(xs)-1,n,dtype=int)
    return np.stack([xs[idx], ys[idx]], 1)

def size_cat(area, img_h, img_w):
    rel = area / (img_h*img_w)
    if rel < .01:   return "small"
    if rel < .10:   return "medium"
    return "large"

# ---- COCO & SAM ----
coco = COCO(ANN_FILE)
all_ids = coco.getImgIds()
img_ids = random.sample(all_ids, SAMPLE_N)     # 抽 500 张

sam = sam_model_registry["vit_l"](checkpoint=CKPT).to(DEVICE)
predictor = SamPredictor(sam)

stats = {sz:{n:[] for n in POINT_COUNTS} for sz in ("small","medium","large")}

for n in POINT_COUNTS:
    for img_id in tqdm(img_ids, desc=f"{n}-pt"):
        info = coco.loadImgs(img_id)[0]
        rgb, s = load_rgb(os.path.join(VAL_DIR, info["file_name"]))
        predictor.set_image(rgb)

        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            gt = coco.annToMask(ann).astype(bool)
            if s < 1.0:
                gt = cv2.resize(gt.astype(np.uint8),
                                (rgb.shape[1], rgb.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            if not gt.any(): continue

            pts  = fg_points(gt, n); labs = np.ones(n,int)
            masks,scores,_ = predictor.predict(
                point_coords=pts, point_labels=labs, multimask_output=True)
            pred = masks[int(np.argmax(scores))]
            iou  = np.logical_and(pred,gt).sum() / np.logical_or(pred,gt).sum()

            sz = size_cat(ann['area'], rgb.shape[0], rgb.shape[1])
            stats[sz][n].append(iou)

# ---- 统计 & 绘图 ----
mean_iou = {sz:[np.mean(stats[sz][n]) for n in POINT_COUNTS] for sz in stats}
recall50 = {sz:[np.mean(np.array(stats[sz][n])>=.5) for n in POINT_COUNTS]
            for sz in stats}
colors = {"small":"tab:blue", "medium":"tab:orange", "large":"tab:green"}

plt.figure(figsize=(5,4))
for sz in stats: plt.plot(POINT_COUNTS, mean_iou[sz], "-o", label=sz, color=colors[sz])
plt.xlabel("#Points"); plt.ylabel("Mean IoU"); plt.ylim(0,1)
plt.title(f"Mean IoU vs #Points (500 imgs)")
plt.legend(); savefig("mean_iou_vs_points_by_size_500.png")

plt.figure(figsize=(5,4))
for sz in stats: plt.plot(POINT_COUNTS, recall50[sz], "-o", label=sz, color=colors[sz])
plt.xlabel("#Points"); plt.ylabel("Recall @0.5"); plt.ylim(0,1)
plt.title(f"Recall@0.5 vs #Points (500 imgs)")
plt.legend(); savefig("recall50_vs_points_by_size_500.png")

print("[Done] 结果保存到 Exp_outputs/")
