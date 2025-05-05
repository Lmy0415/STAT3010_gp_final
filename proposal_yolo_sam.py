#!/usr/bin/env python
# LVIS val2017 | YOLOv8 boxes → SAM masks | 计算 AR@1000
import os, cv2, json, numpy as np, tqdm, torch, ultralytics
from segment_anything import sam_model_registry, SamPredictor
from pycocotools.coco import COCO, mask as maskUtils

ROOT="/home/segment-anything"
LVIS_IMG=f"{ROOT}/coco/val2017"
LVIS_ANN=f"{ROOT}/coco/annotations/instances_val2017.json"   # 简化用 COCO val
CKPT=f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth"
OUT=f"{ROOT}/Exp_outputs/lvis_yolov8_sam_ar.json"

device="cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_l"](checkpoint=CKPT).to(device)
predictor = SamPredictor(sam)

det_model = ultralytics.YOLO("yolov8x.pt")

def encode(m):
    rle=maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
    rle['counts']=rle['counts'].decode('utf-8'); return rle

coco=COCO(LVIS_ANN); img_ids=coco.getImgIds()[:500]     # 先 500 张
res=[]
for img_id in tqdm.tqdm(img_ids):
    info=coco.loadImgs(img_id)[0]
    path=os.path.join(LVIS_IMG,info['file_name'])
    res_det = det_model(path, conf=0.2, iou=0.7, max_det=80, verbose=False)[0]
    if len(res_det.boxes)==0: continue
    rgb=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    for box in res_det.boxes.xyxy.cpu().numpy():
        masks,scores,_=predictor.predict(box=box,multimask_output=True)
        i=np.argmax(scores)
        res.append({"image_id":img_id,"category_id":1,"segmentation":encode(masks[i]),
                    "score":float(scores[i])})
json.dump(res,open(OUT,'w')); print("Dumped result json ->",OUT)
