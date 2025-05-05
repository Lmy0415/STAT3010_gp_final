#!/usr/bin/env python3
# experiment_box_refine_dense_fixed.py
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
OUT_JSON   = "sam_gtbox_refine_dense_fixed.json"

def encode_rle(mask: np.ndarray) -> dict:
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def main():
    coco    = COCO(ANN_FILE)
    img_ids = coco.getImgIds()
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
            # — 第一步：GT 框
            x,y,w,h = ann["bbox"]
            box1 = np.array([[x, y, x+w, y+h]], dtype=float)
            masks1, scores1, _ = predictor.predict(
                box=box1, multimask_output=True
            )
            idx1  = int(np.argmax(scores1))
            mask1 = masks1[idx1].astype(np.uint8)  # (H, W)

            # — 提炼出更紧的框 box2
            cnts, _ = cv2.findContours(
                mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if cnts:
                all_pts = np.vstack(cnts)
                hull    = cv2.convexHull(all_pts)
                rx,ry,rw,rh = cv2.boundingRect(hull)
            else:
                rx,ry,rw,rh = x,y,w,h

            box2 = np.array([[rx, ry, rx+rw, ry+rh]], dtype=float)

            # — 第二步：box2 + mask1 作为 dense prompt
            # 把 (H,W) -> (1,H,W)
            dense_prompt = mask1.astype(float)[None, :, :]

            masks2, scores2, _ = predictor.predict(
                box=box2,
                mask_input=dense_prompt,
                multimask_output=False
            )
            mask2, score2 = masks2[0], float(scores2[0])

            results.append({
                "image_id":     img_id,
                "category_id":  ann["category_id"],
                "segmentation": encode_rle(mask2),
                "score":        score2
            })

    # 写出并评测
    with open(OUT_JSON, "w") as f:
        json.dump(results, f)

    coco_dt = coco.loadRes(OUT_JSON)
    evalr   = COCOeval(coco, coco_dt, iouType="segm")
    evalr.evaluate(); evalr.accumulate(); evalr.summarize()

if __name__ == "__main__":
    main()
