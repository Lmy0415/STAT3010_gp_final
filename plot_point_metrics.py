#!/usr/bin/env python3
"""
Plot mIoU / Recall@0.5 bar chart & IoU histogram for Experiment B
"""
import json, numpy as np, matplotlib.pyplot as plt, os
from utils_plot import savefig

ious = [r["iou"] for r in json.load(open("point_iou_records.json"))]
mean_iou = np.mean(ious)
recall50 = np.mean(np.array(ious) >= 0.5)

# --- 柱状图 ---
plt.figure(figsize=(4,4))
plt.bar(["Mean IoU", "Recall@0.5"], [mean_iou, recall50], color=["C0","C1"])
plt.ylim(0,1); plt.ylabel("Value"); plt.title("Single-Point Zero-Shot")
for i,v in enumerate([mean_iou, recall50]):
    plt.text(i, v+0.02, f"{v:.3f}", ha="center")
plt.tight_layout(); savefig("point_metrics.png")

# --- IoU 直方图 ---
plt.figure(figsize=(6,4))
plt.hist(ious, bins=20, edgecolor="k")
plt.title(f"IoU distribution ({len(ious)} masks)")
plt.xlabel("IoU"); plt.ylabel("Count")
plt.tight_layout(); savefig("iou_hist.png")

print(f"mIoU={mean_iou:.3f}  Recall@0.5={recall50:.3f}")
print("图表已保存到：point_metrics.png / iou_hist.png")
