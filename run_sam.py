#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from segment_anything import (
    sam_model_registry,
    SamPredictor,
    SamAutomaticMaskGenerator
)

def get_device():
    """Select device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_sam(model_type: str, ckpt_path: Path, device: str):
    """Load SAM model and return predictor and sam model."""
    sam = sam_model_registry[model_type](checkpoint=str(ckpt_path))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor, sam

def segment_box(predictor, img: np.ndarray, box: np.ndarray):
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=False,
    )
    return masks[0], scores[0]

def segment_point(predictor, img: np.ndarray, pts: np.ndarray, labs: np.ndarray):
    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=labs,
        box=None,
        multimask_output=False,
    )
    return masks[0], scores[0]

def segment_auto(sam, img: np.ndarray, iou_thresh: float, stab_thresh: float):
    """Generate automatic masks on the same device as sam."""
    gen = SamAutomaticMaskGenerator(
        sam,
        pred_iou_thresh=iou_thresh,
        stability_score_thresh=stab_thresh,
        box_nms_thresh=0.7,
    )
    # ensure float32 input
    masks_data = gen.generate(img.astype(np.float32))
    # sort by predicted_iou descending
    return sorted(masks_data, key=lambda x: x["predicted_iou"], reverse=True)

def save_mask(output: Path, mask: np.ndarray):
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(str(output))

def main():
    p = argparse.ArgumentParser(
        description="CLI for SAM segmentation (box, point or automatic)"
    )
    p.add_argument("--model",      choices=["vit_b","vit_l","vit_h"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input_dir",  type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("outputs"))
    p.add_argument("--mode",       choices=["box","point","auto"], default="auto")
    p.add_argument("--box",        nargs=4,  type=int,
                   help="x0 y0 x1 y1 for box mode")
    p.add_argument("--points",     nargs="+", type=int,
                   help="x1 y1 label1 x2 y2 label2 ... for point mode")
    p.add_argument("--iou_thresh", type=float, default=0.8,
                   help="IOU threshold for auto mode")
    p.add_argument("--stab_thresh",type=float, default=0.8,
                   help="stability score threshold for auto mode")
    args = p.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    predictor, sam = load_sam(args.model, args.checkpoint, device)

    img_paths = sorted(args.input_dir.glob("*.[jp][pn]g"))
    for img_path in tqdm(img_paths, desc="Processing images"):
        img = np.array(Image.open(img_path))

        if args.mode == "box":
            if not args.box:
                raise ValueError("--box required for box mode")
            box = np.array([args.box])
            mask, score = segment_box(predictor, img, box)
            out = args.output_dir / f"{img_path.stem}_box_{score:.2f}.png"

        elif args.mode == "point":
            if not args.points or len(args.points) % 3 != 0:
                raise ValueError("--points x y label ...")
            pts = np.array(args.points).reshape(-1,3)
            coords = pts[:,:2]
            labels = pts[:,2]
            mask, score = segment_point(predictor, img, coords, labels)
            out = args.output_dir / f"{img_path.stem}_pt_{score:.2f}.png"

        else:  # auto
            masks = segment_auto(sam, img, args.iou_thresh, args.stab_thresh)
            mask = masks[0]["segmentation"]
            score = masks[0]["predicted_iou"]
            out = args.output_dir / f"{img_path.stem}_auto_{score:.2f}.png"

        save_mask(out, mask)
        tqdm.write(f"{img_path.name} → {out.name}")

if __name__ == "__main__":
    main()
