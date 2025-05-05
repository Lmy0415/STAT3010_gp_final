#!/usr/bin/env python3
# experiment_box_refine.py
# “自适应”GT-box→mask（二轮box提示）评价 on COCO val2017

import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from segment_anything import sam_model_registry, SamPredictor

# ———— 配置 ————
ROOT       = os.path.dirname(__file__)
COCO_DIR   = os.path.join(ROOT, "coco")
VAL_IMGDIR = os.path.join(COCO_DIR, "val2017")
ANN_FILE   = os.path.join(COCO_DIR, "annotations", "instances_val2017.json")
CKPT       = os.path.join(ROOT, "checkpoints", "sam_vit_l_0b3195.pth")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
OUT_JSON   = "sam_gtbox_refine.json"

def encode_rle(mask):
    """将二值 mask 编码为 COCO RLE 格式"""
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

# 初始化 COCO 与 SAM predictor
coco      = COCO(ANN_FILE)
img_ids   = coco.getImgIds()
predictor = SamPredictor(
    sam_model_registry["vit_l"](checkpoint=CKPT).to(DEVICE)
)

results = []

for img_id in tqdm(img_ids, desc="Refined Box → Mask"):
    info    = coco.loadImgs(img_id)[0]
    img_bgr = cv2.imread(os.path.join(VAL_IMGDIR, info["file_name"]))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
        # 1) 首次用 GT 框 预测
        x, y, w, h = ann["bbox"]
        box1 = np.array([[x, y, x + w, y + h]], dtype=float)
        masks1, scores1, _ = predictor.predict(
            box=box1,
            multimask_output=True
        )
        k1    = int(np.argmax(scores1))
        mask1 = masks1[k1].astype(np.uint8)

        # 2) 从 mask1 提取外轮廓并计算最小外接矩形
        cnts, _ = cv2.findContours(
            mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(cnts) == 0:
            # 如果没有找到任何轮廓，退回到原始 GT 框
            rx, ry, rw, rh = x, y, w, h
        else:
            all_pts = np.vstack(cnts)
            hull    = cv2.convexHull(all_pts)
            rx, ry, rw, rh = cv2.boundingRect(hull)

        # 3) 用“收紧”后的新框 再次预测
        box2 = np.array([[rx, ry, rx + rw, ry + rh]], dtype=float)
        masks2, scores2, _ = predictor.predict(
            box=box2,
            multimask_output=True
        )
        k2    = int(np.argmax(scores2))
        mask2 = masks2[k2]

        results.append({
            "image_id":     img_id,
            "category_id":  ann["category_id"],
            "segmentation": encode_rle(mask2),
            "score":        float(scores2[k2])
        })

# 写出结果并跑 COCOeval
with open(OUT_JSON, "w") as f:
    json.dump(results, f)

coco_dt   = coco.loadRes(OUT_JSON)
evaluator = COCOeval(coco, coco_dt, iouType="segm")
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()
