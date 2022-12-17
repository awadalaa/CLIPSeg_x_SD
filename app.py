import os
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import gradio as gr
from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"


auth_token = os.environ.get("HF_TOKEN") or True
clip_seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clip_seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
sd_inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token).to(device)


def process_image(image, prompt_find, prompt_replace):
  inputs = clip_seg_processor(text=prompt_find, images=image, padding="max_length", return_tensors="pt")

  # predict
  with torch.no_grad():
    outputs = clip_seg_model(**inputs)
    preds = outputs.logits

  filename_mask = f"mask.png"
  plt.imsave(filename_mask, torch.sigmoid(preds))
  mask_image = Image.open(filename_mask).convert("RGB")

  image = sd_inpainting_pipe(prompt=prompt_replace, image=image, mask_image=mask_image).images[0]

  return mask_image,image



title = "Interactive demo: Prompt based inPainting using CLIPSeg x Stable Diffusion"

description = "Demo for prompt based inPainting. It uses CLIPSeg, a CLIP-based model for zero- and one-shot image segmentation. Once it identifies the image segment based on a text mask, or use one of the examples below and click 'submit'. Results will show up in a few seconds."

article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2112.10003'>CLIPSeg: Image Segmentation Using Text and Image Prompts</a> | <a href='https://huggingface.co/docs/transformers/main/en/model_doc/clipseg'>HuggingFace docs</a></p>"

interface = gr.Interface(fn=process_image,
                     inputs=[
                        gr.Image(type="pil"),
                        gr.Textbox(label="What to identify"),
                        gr.Textbox(label="What to replace"),
                      ],
                     outputs=[
                        gr.Image(type="pil"),
                        gr.Image(type="pil"),
                     ],
                     title=title,
                     description=description,
                     article=article)

interface.launch(debug=True)

