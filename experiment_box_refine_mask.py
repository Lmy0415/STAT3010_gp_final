#!/usr/bin/env python3
# experiment_box_refine_mask.py
# 基于初轮mask做密集提示的GT-box→mask（二轮mask提示）评价 on COCO val2017

import os, json, cv2, torch, numpy as np
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
OUT_JSON   = "sam_gtbox_refine_mask.json"

def encode_rle(mask):
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

# 加载COCO
coco    = COCO(ANN_FILE)
img_ids = coco.getImgIds()

# 初始化SAM Predictor
predictor = SamPredictor(
    sam_model_registry["vit_l"](checkpoint=CKPT).to(DEVICE)
)

results = []
for img_id in tqdm(img_ids, desc="Mask-Refine Box → Mask"):
    info    = coco.loadImgs(img_id)[0]
    img_bgr = cv2.imread(os.path.join(VAL_IMGDIR, info["file_name"]))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
        # — 第一步：用GT框得到mask1
        x,y,w,h = ann["bbox"]
        box     = np.array([[x,y,x+w,y+h]], dtype=float)
        masks1, scores1, _ = predictor.predict(
            box=box,
            multimask_output=True
        )
        k1    = int(np.argmax(scores1))
        mask1 = masks1[k1]  # bool or 0/1 mask, shape=H×W

        # — 第二步：用mask1做dense prompt，细化分割
        mask_input = mask1[None, :, :]  # shape (1, H, W)
        masks2, scores2, _ = predictor.predict(
            mask_input=mask_input,
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

# 写出JSON并评估
with open(OUT_JSON, "w") as f:
    json.dump(results, f)
coco_dt   = coco.loadRes(OUT_JSON)
evaluator = COCOeval(coco, coco_dt, iouType="segm")
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()
