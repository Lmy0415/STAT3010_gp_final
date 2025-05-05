#!/usr/bin/env python3
# plot_refined_metrics.py
# 读取 sam_gtbox_refine.json 的 COCOeval 结果并生成四张图到 Exp_outputs/

import os, json
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ———— 配置 ————
ROOT        = os.path.dirname(__file__)
COCO_DIR    = os.path.join(ROOT, "coco")
ANN_FILE    = os.path.join(COCO_DIR, "annotations", "instances_val2017.json")
REFINE_JSON = os.path.join(ROOT, "sam_gtbox_refine.json")
OUT_DIR     = os.path.join(ROOT, "Exp_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 初始化 COCO ground-truth 与预测
coco   = COCO(ANN_FILE)
coco_dt = coco.loadRes(REFINE_JSON)

# 运行 COCOeval
evaluator = COCOeval(coco, coco_dt, iouType="segm")
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()  # 这个步骤会填充 evaluator.stats

stats = evaluator.stats  # numpy 数组，长度 12

# 提取指标
ap_all   = stats[0]  # AP @[.50:.95]
ap50     = stats[1]  # AP @.50
ap75     = stats[2]  # AP @.75
ap_sizes = {"small": stats[3], "medium": stats[4], "large": stats[5]}
ar_all   = {"AR@1": stats[6], "AR@10": stats[7], "AR@100": stats[8]}
ar_sizes = {"small": stats[9], "medium": stats[10], "large": stats[11]}

# 1) 整体 AP
plt.figure(figsize=(6,4))
plt.bar(['AP@[.50:.95]','AP@.50','AP@.75'], [ap_all, ap50, ap75])
plt.title("Refined Box→Mask AP (COCO)")
plt.ylabel("Average Precision")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ap_overall_refined.png"))
plt.close()

# 2) 按尺寸划分的 AP
plt.figure(figsize=(6,4))
plt.bar(list(ap_sizes.keys()), list(ap_sizes.values()))
plt.title("Refined Box→Mask AP by Size")
plt.ylabel("Average Precision")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ap_by_size_refined.png"))
plt.close()

# 3) 整体 AR
plt.figure(figsize=(6,4))
plt.bar(list(ar_all.keys()), list(ar_all.values()))
plt.title("Refined Box→Mask AR (COCO)")
plt.ylabel("Average Recall")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ar_overall_refined.png"))
plt.close()

# 4) 按尺寸划分的 AR
plt.figure(figsize=(6,4))
plt.bar(list(ar_sizes.keys()), list(ar_sizes.values()))
plt.title("Refined Box→Mask AR by Size")
plt.ylabel("Average Recall")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ar_by_size_refined.png"))
plt.close()

print(f"✅ Plots saved under {OUT_DIR}")
