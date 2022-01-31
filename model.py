import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
# equalised weights

class weight_scaled_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.scale = (gain / (in_ch * (kernel_size ** 2))) ** 0.5

        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class pixelnorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, use_pixelnorm=True):
        super().__init__()
        self.conv1 = weight_scaled_conv(in_ch, out_ch)
        self.conv2 = weight_scaled_conv(out_ch, out_ch)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = pixelnorm()
        self.use_pn = use_pixelnorm

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x


class generator(nn.Module):
    def __init__(self, z_dim, in_ch, img_ch=3):
        super().__init__()

        # in = batch, z_dim, 1,1

        self.initial = nn.Sequential(
            pixelnorm(),
            nn.ConvTranspose2d(z_dim, in_ch, 4, 1, 0),  # out = (batch,z_dim, 4,4)
            nn.LeakyReLU(0.2),
            weight_scaled_conv(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            pixelnorm()
        )  # out = (batch,z_dim, 4,4)

        self.rgb0 = weight_scaled_conv(in_ch, img_ch, kernel_size=1, stride=1, padding=0)  # out = (batch, 3, 4, 4)

        self.prog_blocks, self.rgb_layers = (nn.ModuleList([]), nn.ModuleList([self.rgb0]))

        for i in range(len(factors) - 1):  # -1 to prevent index error because of factors[i+1]
            conv_in_c = int(in_ch * factors[i])
            conv_out_c = int(in_ch * factors[i + 1])
            self.prog_blocks.append(conv_block(conv_in_c, conv_out_c))
            self.rgb_layers.append(weight_scaled_conv(conv_out_c, img_ch, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)  #generated is out. out goes in continously.

    def forward(self, x, alpha, steps):
        out = self.initial(x)  ##out = (batch,z_dim, 4,4)


        if steps == 0:
            return self.rgb0(out)  ##out =(batch,3, 4,4)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)




#factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
class discriminator(nn.Module):
    def __init__(self, in_ch, img_ch = 3):
        super().__init__()
        self.progblocks, self.antirgbs = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)


        for i in range(len(factors)-1, 0 , -1):  #8->7->6->5->4->3->2->1->0
            conv_in = int(in_ch * factors[i])
            conv_out = int(in_ch * factors[i-1])
            self.progblocks.append(conv_block(conv_in, conv_out, use_pixelnorm=False))
            self.antirgbs.append(weight_scaled_conv(img_ch, conv_in, kernel_size=1, stride=1, padding=0))


        self.finalantirgb = weight_scaled_conv(img_ch, in_ch, kernel_size=1, stride=1, padding=0)

        self.antirgbs.append(self.finalantirgb)

        ##downscaling will be done using avarage pool

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.finalblock = nn.Sequential(
            weight_scaled_conv(in_ch + 1, in_ch, kernel_size=3, padding=1),  ##512+1=513
            nn.LeakyReLU(0.2),
            weight_scaled_conv(in_ch, in_ch, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            weight_scaled_conv(in_ch, 1, kernel_size=1, padding=0, stride=1)  #out = batch*1*1*1

        )

    def fade_in(self, alpha, downscaled, out):
            """Used to fade in downscaled using avg pooling and output from CNN"""
            # alpha should be scalar within [0, 1], and upscale.shape == generated.shape
            return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        # we take the std for each example (across all channels, and pixels) then we repeat it
        # for a single channel and concatenate it with the image. In this way the discriminator
        # will get information about the variation in the batch/image
        return torch.cat([x, batch_statistics], dim=1)





    def forward(self, x, alpha, steps):
        cur_step = len(self.progblocks) - steps

        # convert from rgb as initial step, this will depend on
        # the image size (each will have it's on rgb layer)
        out = self.leaky(self.antirgbs[cur_step](x))

        if steps == 0:  # i.e, image is 4x4
            out = self.minibatch_std(out)
            return self.finalblock(out).view(out.shape[0], -1)

        # because prog_blocks might change the channels, for down scale we use rgb_layer
        # from previous/smaller size which in our case correlates to +1 in the indexing
        downscaled = self.leaky(self.antirgbs[cur_step + 1](self.avgpool(x)))
        out = self.avgpool(self.progblocks[cur_step](out))

        # the fade_in is done first between the downscaled and the input
        # this is opposite from the generator
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.progblocks)):
            out = self.progblocks[step](out)
            out = self.avgpool(out)

        out = self.minibatch_std(out)
        return self.finalblock(out).view(out.shape[0], -1)

if __name__ == "__main__":
    Z_DIM = 100
    IN_CHANNELS = 256
    gen = generator(Z_DIM, IN_CHANNELS, img_ch=3)
    critic = discriminator(IN_CHANNELS, img_ch=3)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(img_size / 4))
        x = torch.randn((1, Z_DIM, 1, 1))
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At img size: {img_size}")