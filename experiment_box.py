#!/usr/bin/env python3
# experiment_box.py
# Zero-shot GT-box → mask  evaluation on COCO val2017
import os, json, cv2, torch, numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from segment_anything import sam_model_registry, SamPredictor

# -------- Paths --------------------------------------------------
ROOT       = os.path.dirname(__file__)           # /home/segment-anything
COCO_DIR   = os.path.join(ROOT, "coco")
VAL_IMGDIR = os.path.join(COCO_DIR, "val2017")
ANN_FILE   = os.path.join(COCO_DIR, "annotations",
                          "instances_val2017.json")
CKPT   = os.path.join(ROOT, "checkpoints", "sam_vit_l_0b3195.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------------------------

def encode_rle(mask: np.ndarray) -> dict:
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

coco = COCO(ANN_FILE)
img_ids = coco.getImgIds()

predictor = SamPredictor(
    sam_model_registry["vit_l"](checkpoint=CKPT).to(DEVICE)
)

results = []
for img_id in tqdm(img_ids, desc="GT-box → mask"):
    info = coco.loadImgs(img_id)[0]
    rgb  = cv2.cvtColor(cv2.imread(os.path.join(VAL_IMGDIR,
                     info["file_name"])), cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
        x, y, w, h = ann["bbox"]
        box = np.array([[x, y, x+w, y+h]], dtype=np.float32)
        masks, scores, _ = predictor.predict(
            box=box, multimask_output=True)
        k = int(np.argmax(scores))
        results.append({
            "image_id":     img_id,
            "category_id":  ann["category_id"],
            "segmentation": encode_rle(masks[k]),
            "score":        float(scores[k])
        })

with open("sam_gtbox_results.json", "w") as f:
    json.dump(results, f)

# COCOeval
coco_dt = coco.loadRes("sam_gtbox_results.json")
evaluator = COCOeval(coco, coco_dt, iouType="segm")
evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
