#!/usr/bin/env python3
"""
Plot AP / AR bar charts for Experiment A (GT-box → mask)
"""
import json, os, matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils_plot import savefig

ROOT      = os.path.dirname(__file__)
COCO_DIR  = os.path.join(ROOT, "coco")
ANN_FILE  = os.path.join(COCO_DIR, "annotations", "instances_val2017.json")

# 1. 载入评估结果
coco_gt = COCO(ANN_FILE)
coco_dt = coco_gt.loadRes("sam_gtbox_results.json")

ev = COCOeval(coco_gt, coco_dt, iouType="segm")
ev.evaluate(); ev.accumulate(); ev.summarize()

# ev.stats = 12-dim array（官方同名顺序）
ap_overall = {
    "AP@[.50:.95]": ev.stats[0],
    "AP@0.50":      ev.stats[1],
    "AP@0.75":      ev.stats[2],
}
ap_size = {
    "small":  ev.stats[3],
    "medium": ev.stats[4],
    "large":  ev.stats[5],
}
ar_overall = {
    "AR@1":   ev.stats[6],
    "AR@10":  ev.stats[7],
    "AR@100": ev.stats[8],
}
ar_size = {
    "small":  ev.stats[9],
    "medium": ev.stats[10],
    "large":  ev.stats[11],
}

# 2. 绘图
plt.figure(figsize=(6,4))
plt.bar(ap_overall.keys(), ap_overall.values())
plt.title("Zero-shot Box→Mask AP")
plt.ylim(0,1); plt.ylabel("AP")
plt.tight_layout(); savefig("ap_overall.png")

plt.figure(figsize=(5,4))
plt.bar(ap_size.keys(), ap_size.values())
plt.title("AP by Object Size"); plt.ylim(0,1)
plt.tight_layout(); savefig("ap_by_size.png")

plt.figure(figsize=(6,4))
plt.bar(ar_overall.keys(), ar_overall.values())
plt.title("Zero-shot Box→Mask AR"); plt.ylim(0,1)
plt.tight_layout(); savefig("ar_overall.png")

plt.figure(figsize=(5,4))
plt.bar(ar_size.keys(), ar_size.values())
plt.title("AR by Object Size"); plt.ylim(0,1)
plt.tight_layout(); savefig("ar_by_size.png")

print("图表已保存到项目根目录：ap_overall.png / ap_by_size.png / ar_overall.png / ar_by_size.png")
