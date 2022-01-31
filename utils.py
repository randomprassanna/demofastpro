import numpy as np
import os
import random
import torch
import torchvision
from torchvision.utils import save_image

#my files
import config

def plot_to_tensorboard(writer, critic_lossW, gen_lossW, real_preds, fake_preds, tensorboard_step):

    writer.add_scalar("critic_loss", critic_lossW, global_step = tensorboard_step)

    with torch.no_grad():
        real_img_grid = torchvision.utils.make_grid(real_preds[:8], normalize = True)
        fake_img_grid = torchvision.utils.make_grid(fake_preds[:8], normalize=True)
        writer.add_image("real", real_img_grid, global_step=tensorboard_step)
        writer.add_image("fake", fake_img_grid, global_step = tensorboard_step)



def gradient_penalty(critic, real_images, fake_images, alpha, train_step, device='cpu'):
    N, C, H, W = real_images.shape
    epsilon = torch.randn(N, 1, 1, 1).repeat(1, C, H, W)
    epsilon = epsilon.to(config.DEVICE)
    interpolated_images = real_images * epsilon + fake_images * (1 - epsilon)

    # calculate scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty




def save_checkpoint(model, optimizer, alpha, epoch, step, filename = "my_checkpoint.pth.tar", ):
    print("=> saving checkpoint")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "alpha" : alpha,
        "epoch" : epoch,
        "step" : step
    }
    torch.save(checkpoint, filename)




def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=>loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    alpha = checkpoint["alpha"]
    epoch = checkpoint["epoch"]
    step = checkpoint["step"]

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return checkpoint




#fucntion for generating result images
def generate_examples(gen, steps,folder=config.generation_folder, n=100):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, config.z_dim, 1, 1).to(config.DEVICE)
            img = gen(noise, alpha, steps)
            img_filename = f'img_{i}.png'
            save_image(img * 0.5 + 0.5, os.path.join(folder, img_filename))

    gen.train()












