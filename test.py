import io
import os, sys
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from dall_e          import map_pixels, unmap_pixels, load_model


def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def preprocess(img, target_image_size=256):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)


def reconstruct_with_dalle(x, encoder, decoder, do_preprocess=False):
  # takes in tensor (or optionally, a PIL image) and returns a PIL image
  if do_preprocess:
    x = preprocess(x)
  z_logits = encoder(x)
  z = torch.argmax(z_logits, axis=1)
  
  print(f"DALL-E: latent shape: {z.shape}")
  z = F.one_hot(z, num_classes=encoder.vocab_size).permute(0, 3, 1, 2).float()

  x_stats = decoder(z).float()
  x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
  x_rec = T.ToPILImage(mode='RGB')(x_rec[0])

  return x_rec


def stack_reconstructions(input, x1, x2, x3, titles=[]):
  assert input.size == x1.size == x2.size == x3.size
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (4*w, h))
  img.paste(input, (0,0))
  img.paste(x1, (1*w,0))
  img.paste(x2, (2*w,0))
  img.paste(x3, (3*w,0))
  for i, title in enumerate(titles):
    ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255), font=font) # coordinates, text, color, font
  return img


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# For faster load times, download these files locally and use the local paths instead.
# encoder_dalle = load_model("https://cdn.openai.com/dall-e/encoder.pkl", DEVICE)
# decoder_dalle = load_model("https://cdn.openai.com/dall-e/decoder.pkl", DEVICE)
encoder_dalle = load_model("./encoder.pkl", DEVICE)
decoder_dalle = load_model("./decoder.pkl", DEVICE)


import yaml
import torch
#from omegaconf import OmegaConf
#from taming.models.vqgan import VQModel

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None):
  model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec

titles=["Input", "VQGAN (16384)", "VQGAN (1024)", "DALL-E"]

def reconstruction_pipeline(url, size=320):
  x = preprocess(download_image(url), target_image_size=size)
  x = x.to(DEVICE)
  print(f"input is of size: {x.shape}")
  # x1 = reconstruct_with_vqgan(preprocess_vqgan(x), model16384)
  # x2 = reconstruct_with_vqgan(preprocess_vqgan(x), model1024)
  x3 = reconstruct_with_dalle(x, encoder_dalle, decoder_dalle)
  x1 = x3
  x2 = x3
  #img = stack_reconstructions(custom_to_pil(preprocess_vqgan(x[0])), custom_to_pil(x1[0]), custom_to_pil(x2[0]), x3, titles=titles)
  img = stack_reconstructions(x[0], x1, x2, x3, titles=titles)
  return img


#img = reconstruction_pipeline('https://heibox.uni-heidelberg.de/f/7bb608381aae4539ba7a/?dl=1')
