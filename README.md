# STAT3010 Group Project: Segment Anything Experiments

**Interactive & Zero-Shot Object Segmentation with Meta’s SAM**

---

## Project Overview

This project is based on Meta Research’s [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything). It implements and evaluates segmentation on the COCO validation set and custom image collections using:

- **Box → Mask**: generate masks from bounding boxes  
- **Point Prompts**: segment via foreground/background points  
- **Box-Refine → Mask**: refine input boxes based on initial masks  
- **Batch Evaluation**: compute IoU, mAP, Recall and export visualizations  

---

## Dependencies

- Python 3.8+  
- PyTorch 1.13+  
- torchvision 0.14+  
- numpy  
- opencv-python  
- tqdm  
- [segment-anything](https://github.com/facebookresearch/segment-anything)

Install with:
```bash
pip install -r requirements.txt
```

## Installation

1. **Clone repository**  
   ```bash
   git clone git@github.com:Lmy0415/STAT3010_gp_final.git
   cd STAT3010_gp_final
   
2. **Create & activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3. **Download SAM checkpoints**
   ```bash
   mkdir -p checkpoints
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth \ -P checkpoints/

## Usage Examples

**Box → Mask**
   ```bash
   python vis_point_cases.py \
  --input_image data/coco/000000123456.jpg \
  --box 50 30 200 180 \
  --model_checkpoint checkpoints/sam_vit_l_0b3195.pth
  ```

**Point Prompts (5 FG + 5 BG)**
   ```bash
   python vis_point_cases.py \
  --input_image data/coco/000000123456.jpg \
  --points 100 150 300 250 50 75 350 275 \
  --labels 1 1 1 1 1 0 0 0 0 0 \
  --model_checkpoint checkpoints/sam_vit_l_0b3195.pth
  ```

**Batch COCO Evaluation**
  ```bash
  python evaluate_coco.py \
  --ann_file data/coco/annotations/instances_val2017.json \
  --model_checkpoint checkpoints/sam_vit_l_0b3195.pth \
  --output_dir Exp_outputs/results/
  ```

##Repository Structure
  ```bash
    .
  ├── checkpoints/            # SAM model checkpoints
  ├── data/                   # COCO validation images & annotations
  │   └── coco/
  ├── Exp_outputs/            # Experiment outputs & visualizations (git‑ignored)
  ├── vis_point_cases.py      # Box & point‑prompt visualization
  ├── evaluate_coco.py        # COCO batch‑evaluation script
  ├── utils_plot.py           # Plotting utilities
  ├── requirements.txt        # Python dependencies
  └── README.md               # This file
  ```

## Experimental Results (Summary)

- **Single-Click (Box)**
  - Mean IoU: **0.587**
  - Recall @ 0.5: **0.655**

- **Multi-Point Prompts (5 FG + 5 BG)**
  - Mean IoU: **0.620**

- **Box-Refine → Mask**
  - mAP @ IoU ≥ 0.5: **0.730**

> Full charts and tables are available in `Exp_outputs/results/summary.png`.


## Credits & License

This project builds on Meta Research’s **Segment Anything Model** (Apache 2.0).

- **SAM repository**: <https://github.com/facebookresearch/segment-anything>  
- **SAM paper**: [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)  
- **License**: see the [LICENSE](LICENSE) file.

## Contributors

- Jiang Qiushi
- Li Maoyuan
- Sun Mayuan


## Citation

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
