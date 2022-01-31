from fastapi import FastAPI, Body, Request
from pydantic import BaseModel
from fastapi.responses import Response, HTMLResponse
from fastapi.templating import Jinja2Templates

from utils import generate_examples, load_checkpoint
from model import generator, discriminator
from math import log2
from tqdm import tqdm
import config

import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms.functional as func

import io
from PIL import Image

app = FastAPI()
db = []
templates = Jinja2Templates(directory="frontend")

def show_images_on_web(gen, steps):
    gen.eval()
    alpha = 1.0
    with torch.no_grad():
        noise = torch.randn(1, config.z_dim, 1, 1).to(config.DEVICE)
        img = gen(noise, alpha, steps)
        img = img * 0.5 + 0.5
        img = img.squeeze()
    return func.to_pil_image(img)

class imgresponse(Response):
    def __init__(self, content, *args, **kwargs):
        super().__init__(
            content=img,
            media_type="image/png",
            *args,
            **kwargs,
        )

@app.get("/generate-new-images/")
async def main(request: Request):
    gen = generator(config.z_dim, config.in_ch, img_ch=config.img_ch).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    checkpoint = load_checkpoint(config.CHECKPOINT_GENERATOR, gen, opt_gen, config.LEARNING_RATE)
    img = show_images_on_web(gen, steps=checkpoint['step'])
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img = buf.getvalue()
    return Response(content=img, media_type="image/png")
    #return templates.TemplateResponse("index.html", {"request": imgr})



@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
