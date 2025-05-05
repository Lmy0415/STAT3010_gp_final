#!/usr/bin/env python3
# experiment_box_refine_dense.py
# “自适应”GT-box→mask（二轮 box + dense-mask 提示）评价 on COCO val2017

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
OUT_JSON   = "sam_gtbox_refine_dense.json"

def encode_rle(mask: np.ndarray) -> dict:
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def main():
    coco      = COCO(ANN_FILE)
    img_ids   = coco.getImgIds()
    predictor = SamPredictor(
        sam_model_registry["vit_l"](checkpoint=CKPT).to(DEVICE)
    )

    results = []
    for img_id in tqdm(img_ids, desc="Dense-Refine Box → Mask"):
        info    = coco.loadImgs(img_id)[0]
        img_bgr = cv2.imread(os.path.join(VAL_IMGDIR, info["file_name"]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            # — 第一步：用 GT 框预测
            x,y,w,h = ann["bbox"]
            box1 = np.array([[x, y, x+w, y+h]], dtype=float)
            masks1, scores1, _ = predictor.predict(
                box=box1, multimask_output=True
            )
            k1    = int(np.argmax(scores1))
            mask1 = masks1[k1]  # bool mask

            # — 生成“收紧”后的小框 box2
            cnts, _ = cv2.findContours(
                mask1.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if len(cnts)==0:
                # 如果 mask1 为空，仍 fallback 回 box1
                rx,ry,rw,rh = x,y,w,h
            else:
                all_pts = np.vstack(cnts)
                hull    = cv2.convexHull(all_pts)
                rx,ry,rw,rh = cv2.boundingRect(hull)

            box2 = np.array([[rx, ry, rx+rw, ry+rh]], dtype=float)

            # — 第二步：同时给出 box2 + mask1 作为 dense prompt
            dense_mask = mask1[np.newaxis, ...]  # (1, H, W)

            masks2, scores2, _ = predictor.predict(
                box=box2,
                mask_input=dense_mask,
                multimask_output=False   # 只要最优输出
            )
            mask2 = masks2[0]
            score2 = float(scores2[0])

            results.append({
                "image_id":     img_id,
                "category_id":  ann["category_id"],
                "segmentation": encode_rle(mask2),
                "score":        score2
            })

    # 写出结果并跑 COCOeval
    with open(OUT_JSON, "w") as f:
        json.dump(results, f)

    coco_dt   = coco.loadRes(OUT_JSON)
    evaluator = COCOeval(coco, coco_dt, iouType="segm")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

if __name__ == "__main__":
    main()
