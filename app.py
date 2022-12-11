import os
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import gradio as gr
from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2

clip_seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clip_seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

def process_image(image, prompt):
  inputs = clip_seg_processor(text=prompt, images=image, padding="max_length", return_tensors="pt")

  # predict
  with torch.no_grad():
    outputs = clip_seg_model(**inputs)
    preds = outputs.logits

  filename_mask = f"mask.png"
  plt.imsave(filename_mask, torch.sigmoid(preds))
  mask_image = Image.open(filename_mask).convert("RGB")

  return mask_image



title = "Interactive demo: Prompt based inPainting using CLIPSeg x Stable Diffusion"

description = "Demo for prompt based inPainting. It uses CLIPSeg, a CLIP-based model for zero- and one-shot image segmentation. Once it identifies the image segment based on a text mask, or use one of the examples below and click 'submit'. Results will show up in a few seconds."

article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2112.10003'>CLIPSeg: Image Segmentation Using Text and Image Prompts</a> | <a href='https://huggingface.co/docs/transformers/main/en/model_doc/clipseg'>HuggingFace docs</a></p>"

interface = gr.Interface(fn=process_image,
                     inputs=[
                        gr.Image(type="pil"),
                        gr.Textbox(label="What to identify"),
                      ],
                     outputs=[
                        gr.Image(type="pil"),
                     ],
                     title=title,
                     description=description,
                     article=article)

interface.launch(debug=True)

