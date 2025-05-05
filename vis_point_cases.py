#!/usr/bin/env python3
"""
Visualize Top-5 / Bottom-5 single-point cases
— 保存为 PNG 文件 (no GUI needed)
"""
import pickle, cv2, matplotlib.pyplot as plt, os
from utils_plot import savefig

OUT_DIR = "case_plots"
os.makedirs(OUT_DIR, exist_ok=True)

cases = pickle.load(open("point_cases.pkl", "rb"))

def save_fig(case_list, title, fname):
    plt.figure(figsize=(15,6))
    for i, c in enumerate(case_list):
        img = cv2.cvtColor(cv2.imread(c["img_path"]), cv2.COLOR_BGR2RGB)
        plt.subplot(2,5,i+1)
        plt.imshow(img); plt.axis('off')
        plt.contour(c["gt_mask"],  colors='g', linewidths=1)
        plt.contour(c["pred_mask"],colors='r', linewidths=1)
        plt.title(f'IoU={c["iou"]:.2f}')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, fname)
    savefig(out_path)
    plt.close()
    print(f"Saved → {out_path}")

save_fig(cases["bottom"], "Worst 5 Cases (single-point)", "worst5.png")
save_fig(cases["top"],    "Best  5 Cases (single-point)", "best5.png")
