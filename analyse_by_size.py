# analyse_by_size.py
import json, os, numpy as np, pandas as pd
from pycocotools.coco import COCO

ROOT      = os.path.dirname(__file__)
COCO_DIR  = os.path.join(ROOT, "coco")
ANN_FILE  = os.path.join(COCO_DIR, "annotations", "instances_val2017.json")

coco   = COCO(ANN_FILE)
info   = {img["id"]: img for img in coco.loadImgs(coco.getImgIds())}
recs   = json.load(open("point_iou_records.json"))

def size_cat(ann, img):
    r = ann["area"] / (img["height"] * img["width"])
    return "small" if r < .01 else ("medium" if r < .1 else "large")

rows = []
for r in recs:
    ann = coco.loadAnns(r["ann_id"])[0]
    rows.append({
        "iou": r["iou"],
        "size": size_cat(ann, info[r["image_id"]])
    })

df = pd.DataFrame(rows)
print(df.groupby("size").iou.agg(["mean", lambda x: (x>=.5).mean(), "count"])
      .rename(columns={"<lambda_0>":"recall@0.5"}))

