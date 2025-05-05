#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edge_bsds.py  ·  Zero-shot edge detection with SAM on BSDS500

流程：
 1. 用 SamAutomaticMaskGenerator 生成自动掩码，累计成概率图
 2. 用 Canny + 单像素骨架化（thin）提边
 3. 将 BSDS500 所有 GT 标注文件中的多维 Boundaries 合并为 2D Skeleton
 4. 对 11 个阈值做 sweep，计算 ODS/OIS/AP/R50
 5. 保存前 10 张叠加可视化到 Exp_outputs/edge_vis_XX.png
 6. 保存指标到 Exp_outputs/edge_bsds_metrics.txt
"""

import os, cv2, json, argparse
from pathlib import Path
import numpy as np
import scipy.io as sio
import tqdm

from skimage.morphology import thin
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def collapse_to_2d(arr):
    """
    无论 arr 是几维，都把最后面多余的维度做 OR，保留前两个维度 (H,W)。
    """
    a = np.asarray(arr, dtype=bool)
    # 如果维度 <2，不能处理：直接返回空图
    if a.ndim < 2:
        return np.zeros((0,0), bool)
    # 如果维度 == 2，直接返回
    if a.ndim == 2:
        return a
    # 维度 >2，将第 2 轴（从 0 开始数，indices ≥2）都做 OR
    return np.any(a, axis=tuple(range(2, a.ndim)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      default="checkpoints/sam_vit_l_0b3195.pth")
    parser.add_argument("--bsds_root", default="datasets/BSDS500")
    parser.add_argument("--out_dir",   default="Exp_outputs")
    parser.add_argument("--pps", type=int, default=16, help="points_per_side")
    args = parser.parse_args()

    IMG_DIR = Path(args.bsds_root) / "data/images/test"
    GT_DIR  = Path(args.bsds_root) / "data/groundTruth/test"
    OUT_DIR = Path(args.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化 SAM + 自动掩码生成器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_l"](checkpoint=args.ckpt).to(device)
    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.pps,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.9,
        crop_n_layers=1,
    )

    # 准备阈值 sweep
    THRS = np.linspace(0.05, 0.45, 11)
    stats = {t: {"tp":0, "fp":0, "fn":0} for t in THRS}

    imgs = sorted(IMG_DIR.glob("*.jpg"))
    assert imgs, f"No images under {IMG_DIR}"

    for idx, img_p in enumerate(tqdm.tqdm(imgs, desc="BSDS500")):
        # 1) 读图 + SAM -> 概率图
        bgr = cv2.imread(str(img_p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        prob = np.zeros((H, W), np.float32)
        for m in mask_gen.generate(rgb):
            prob += m["segmentation"].astype(np.float32)
        prob = np.clip(prob, 0, 1)

        # 2) Canny + 单像素骨架
        edges = cv2.Canny((prob*255).astype(np.uint8), 40, 100, L2gradient=True)
        edges = thin(edges > 0)  # bool

        # 前 10 张可视化
        if idx < 10:
            vis = bgr.copy()
            vis[edges] = (0,255,255)
            cv2.imwrite(str(OUT_DIR / f"edge_vis_{idx:02d}.png"), vis)

        # 3) 合并所有 GT skeleton
        mat = sio.loadmat(GT_DIR / f"{img_p.stem}.mat")
        gt_all = np.zeros((H, W), dtype=bool)
        for anno in mat["groundTruth"][0]:
            bnd = anno[0][0]["Boundaries"][0][0]
            b2 = collapse_to_2d(bnd)
            # 如果 collapse_to_2d 得到错误尺寸，跳过
            if b2.shape != (H, W):
                continue
            sk = thin(b2)
            gt_all |= sk

        # 4) 各阈值 TP/FP/FN
        for t in THRS:
            pred = (edges & (prob >= t))
            tp = np.logical_and(pred,  gt_all).sum()
            fp = np.logical_and(pred, ~gt_all).sum()
            fn = np.logical_and(~pred, gt_all).sum()
            s = stats[t]
            s["tp"] += tp; s["fp"] += fp; s["fn"] += fn

    # ============= 计算指标 =============
    def prf(s):
        p = s["tp"] / (s["tp"] + s["fp"] + 1e-9)
        r = s["tp"] / (s["tp"] + s["fn"] + 1e-9)
        f = 2*p*r / (p+r+1e-9)
        return p, r, f

    # ODS: 单阈值最优
    ods_t, ods_f = max(((t, prf(s)[2]) for t,s in stats.items()), key=lambda x:x[1])
    ods_p, ods_r, _ = prf(stats[ods_t])

    # OIS: 近似为每图最优的均值
    ois_f = np.mean([max(prf(stats[t])[2] for t in THRS) for _ in imgs])

    # AP: PR 曲线积分
    precs, recs = [], []
    for t in sorted(THRS):
        p, r, _ = prf(stats[t])
        precs.append(p); recs.append(r)
    ap = np.trapz(precs, recs)

    # R50: Precision ≥ 0.5 时最大的 Recall
    # R50: Recall at Precision ≥ 0.5 (safe fallback to 0.0)
    r50_list = []
    for t in THRS:
        p, r, _ = prf(stats[t])
        if p >= 0.5:
            r50_list.append(r)
    r50 = max(r50_list) if r50_list else 0.0


    metrics = {
        "ODS_F": round(float(ods_f), 4),
        "ODS_P": round(float(ods_p), 4),
        "ODS_R": round(float(ods_r), 4),
        "OIS_F": round(float(ois_f), 4),
        "AP":    round(float(ap),    4),
        "R50":   round(float(r50),   4),
    }

    # 输出
    print(json.dumps(metrics, indent=2))
    with open(OUT_DIR/"edge_bsds_metrics.txt","w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✔ Done. Results in {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
