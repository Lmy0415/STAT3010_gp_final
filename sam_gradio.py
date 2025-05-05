import torch
import numpy as np
from PIL import Image
import gradio as gr

from segment_anything import sam_model_registry, SamPredictor

# 设备选择
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", device)

# 加载 SAM
model_type = "vit_l"
checkpoint = "checkpoints/sam_vit_l_0b3195.pth"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# 处理上传图片
def upload_image(img):
    return img, []

# 分割函数
def segment_image_with_boxes(image_with_boxes):
    if image_with_boxes is None:
        return None
    img, boxes = image_with_boxes
    if not boxes:
        return None

    predictor.set_image(img)
    boxes_np = np.array([b["bbox"] for b in boxes], dtype=np.float32)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes_np,
        multimask_output=False
    )
    mask = masks[0]
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGBA")
    base = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255,255,255,0))
    overlay.paste(mask_img, (0,0), mask_img)
    return Image.alpha_composite(base, overlay)

# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## SAM Box Prompt with Real Upload & Draw!")
    with gr.Row():
        with gr.Column():
            uploader = gr.Image(label="Upload Image", type="numpy")
            annotated_img = gr.AnnotatedImage(label="Draw boxes here", type="numpy")
        with gr.Column():
            out = gr.Image(label="Segmentation Result")
    uploader.change(fn=upload_image, inputs=uploader, outputs=annotated_img)
    btn = gr.Button("Segment")
    btn.click(fn=segment_image_with_boxes, inputs=annotated_img, outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
