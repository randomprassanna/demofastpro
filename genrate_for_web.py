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

gen = generator(config.z_dim, config.in_ch, img_ch=config.img_ch).to(config.DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr = config.LEARNING_RATE, betas = (0.0, 0.99))
checkpoint = load_checkpoint(config.CHECKPOINT_GENERATOR, gen, opt_gen, config.LEARNING_RATE)


def show_images_on_web(gen, steps):
    gen.eval()
    alpha = 1.0
    with torch.no_grad():
        noise = torch.randn(1, config.z_dim, 1, 1).to(config.DEVICE)
        img = gen(noise, alpha, steps)
        img = img * 0.5 + 0.5
        img = img.squeeze()
    return func.to_pil_image(img)


img = show_images_on_web(gen, steps=checkpoint['step'])

print(type(img))

# print(img.format)
# from PIL import Image
# im = Image.open()
# im.save('img.png')
# print(img.format)






