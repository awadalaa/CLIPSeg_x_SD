from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import gradio as gr
from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

def process_image(image, prompt):
  inputs = processor(text=prompt, images=image, padding="max_length", return_tensors="pt")

  # predict
  with torch.no_grad():
    outputs = model(**inputs)
    preds = outputs.logits

  filename = f"mask.png"
  plt.imsave(filename, torch.sigmoid(preds))
  return Image.open("mask.png").convert("RGB")



title = "Interactive demo: zero-shot image segmentation with CLIPSeg"

description = "Demo for using CLIPSeg, a CLIP-based model for zero- and one-shot image segmentation. To use it, simply upload an image and add a text to mask (identify in the image), or use one of the examples below and click 'submit'. Results will show up in a few seconds."

article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2112.10003'>CLIPSeg: Image Segmentation Using Text and Image Prompts</a> | <a href='https://huggingface.co/docs/transformers/main/en/model_doc/clipseg'>HuggingFace docs</a></p>"

interface = gr.Interface(fn=process_image,
                     inputs=[gr.Image(type="pil"), gr.Textbox(label="Please describe what you want to identify")],
                     outputs=gr.Image(type="pil"),
                     title=title,
                     description=description,
                     article=article)

interface.launch(debug=True)

