from fastapi import FastAPI, Body, Request
from pydantic import BaseModel
from fastapi.responses import Response, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
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
from torchvision.utils import save_image

import io
from PIL import Image

app = FastAPI()
templates = Jinja2Templates(directory="frontend")
app.mount("/static", StaticFiles(directory="static"), name="static")


def show_images_on_web(gen, steps):
    gen.eval()
    alpha = 1.0
    for i in range(10):
        with torch.no_grad():
            noise = torch.randn(1, config.z_dim, 1, 1).to(config.DEVICE)
            img = gen(noise, alpha, steps)
            img_filename = f'img_{i}.png'
            save_image(img * 0.5 + 0.5, os.path.join("static", img_filename))
        #img = img.squeeze()
    #return func.to_pil_image(img)

def main():
    gen = generator(config.z_dim, config.in_ch, img_ch=config.img_ch).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    checkpoint = load_checkpoint(config.CHECKPOINT_GENERATOR, gen, opt_gen, config.LEARNING_RATE)
    show_images_on_web(gen, steps=checkpoint['step'])
    # img_filename = "one.png"
    # save_image(img, os.path.join("static", img_filename))
    # buf = io.BytesIO()
    # img.save(buf, format='JPEG')
    # img = buf.getvalue()
    #img = url_for('static', filename="static/img.png")
    #return HTMLResponse(content=img, media_type="image/png")
    #return templates.TemplateResponse("mixpage.html", {"request": request, "img": img})

@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    main()
    return templates.TemplateResponse("mixpage.html", {"request": request})

