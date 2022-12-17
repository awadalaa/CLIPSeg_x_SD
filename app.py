import gradio as gr
import torch
import matplotlib.pyplot as plt
import cv2
import os

from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
from torch.cuda.amp import autocast


device = "cuda" if torch.cuda.is_available() else "cpu"


auth_token = os.environ.get("HF_TOKEN") or True
clip_seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clip_seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
sd_inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token).to(device)

WIDTH=512
HEIGHT=512
DILATE=10
THRESHOLDS=0.1


def dilate_mask(mask_file):
    image = cv2.imread(mask_file, 0)
    kernel  = np.ones((DILATE, DILATE), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    im_bin = (dilated > 127) * 255
    cv2.imwrite(mask_file, im_bin)
    return mask_file

def process_mask(prompt_find, image, THRESHOLDS=0.1):
  inputs = clip_seg_processor(
    text=prompt_find,
    images=image,
    padding="max_length",
    return_tensors="pt"
  )

  # predict
  with torch.no_grad():
    outputs = clip_seg_model(**inputs)
    preds = outputs.logits

  out_img = torch.sigmoid(preds)
  out_img = (out_img - out_img.min()) / out_img.max()
  if isinstance(THRESHOLDS, list):
    if len(THRESHOLDS) >= 2:
      out_img = torch.where(out_img >= THRESHOLDS[1], 1., out_img)
      out_img = torch.where(out_img <= THRESHOLDS[0], 0., out_img)
    else:
      out_img = torch.where(out_img >= THRESHOLDS[0], 1., 0.)
  else:
    out_img = torch.where(out_img >= THRESHOLDS, 1., 0.)

  mask_file="mask.png"
  plt.imsave(mask_file, out_img)
  dilated_mask = dilate_mask(mask_file)

  mask_image = Image.open(dilated_mask)

  return mask_image

def process_inpaint(prompt_replace, image, mask_image):
  image = sd_inpainting_pipe(
    prompt=prompt_replace,
    image=image,
    mask_image=mask_image
  ).images[0]
  return image

def process_image(image, prompt_find, prompt_replace):
  orig_image = image.resize((WIDTH, HEIGHT))
  mask_image = process_mask(prompt_find, orig_image).resize((WIDTH, HEIGHT))
  new_image = process_inpaint(prompt_replace, orig_image, mask_image)

  return new_image, mask_image



title = "Interactive demo: Prompt based inPainting using CLIPSeg x Stable Diffusion"

description = "Demo for prompt based inPainting. It uses CLIPSeg, a CLIP-based model for zero- and one-shot image segmentation. Once it identifies the image segment based on a text mask, or use one of the examples below and click 'submit'. Results will show up in a few seconds."

article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2112.10003'>CLIPSeg: Image Segmentation Using Text and Image Prompts</a> | <a href='https://huggingface.co/docs/transformers/main/en/model_doc/clipseg'>HuggingFace docs</a></p>"

interface = gr.Interface(fn=process_image,
                     inputs=[
                        gr.Image(type="pil"),
                        gr.Textbox(label="What to identify"),
                        gr.Textbox(label="What to replace it with"),
                      ],
                     outputs=[
                        gr.Image(type="pil"),
                        gr.Image(type="pil"),
                     ],
                     title=title,
                     description=description,
                     article=article)

interface.launch(debug=True)

