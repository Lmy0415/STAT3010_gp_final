"""
utils_plot.py
Small helper to save matplotlib figures into Exp_outputs/
"""
import os
import matplotlib.pyplot as plt

# 计算输出目录路径
ROOT     = os.path.dirname(__file__)                 # /home/segment-anything
OUT_DIR  = os.path.join(ROOT, "Exp_outputs")
os.makedirs(OUT_DIR, exist_ok=True)                  # 若不存在则创建

def savefig(filename: str, **plt_savefig_kwargs):
    """
    Save current matplotlib figure to Exp_outputs/filename
    Example:
        plt.plot(...)
        savefig("iou_hist.png")
    """
    path = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, **plt_savefig_kwargs)
    print(f"[utils_plot] Saved → {path}")
