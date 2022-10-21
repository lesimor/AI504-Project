import os
import random
from pathlib import Path

# load packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio  #### install with "pip install imageio"
from IPython.display import HTML
from constants import *

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid
from constants import BATCH_SIZE, CHANNEL_NUM, NOISE_WIDTH
from utils import save_gif

from vanilla.generator import Generator as VanillaG
from vanilla.discriminator import Discriminator as VanillaD

from dcgan.generator import Generator as DCGanG
from dcgan.discriminator import Discriminator as DCGanD

generator_class = None
discriminator_class = None
model_name = os.getenv("GAN_MODEL_NAME", "dcgan")
if model_name == "dcgan":
    generator_class = DCGanG
    discriminator_class = DCGanD
else:
    model_name = "vanilla"
    generator_class = VanillaG
    discriminator_class = VanillaD


netG = generator_class().cuda()
netD = discriminator_class().cuda()

optimizerD = optim.Adam(netD.parameters(), lr=0.0002)
optimizerG = optim.Adam(netG.parameters(), lr=0.0002)


# noise sampling
noise = torch.randn(NOISE_WIDTH, 100).cuda()

fixed_noise = torch.randn(BATCH_SIZE, NOISE_DIMENSION, 1, 1).cuda()

criterion = nn.BCELoss()

n_epoch = 200
training_progress_images_list = []

transform = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

img_dir = "img_mini" if os.getenv("DEBUG") == "TRUE" else "img"
dataset = dset.ImageFolder(Path.cwd() / "assets" / img_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(n_epoch):
    for i, (data, _) in enumerate(dataloader):
        ####################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
        ###################################################
        # train with real
        netD.zero_grad()
        data = data.cuda()
        batch_size = data.size(0)
        label = torch.ones((batch_size,)).cuda()  # real label = 1
        output = netD(data)
        errD_real = criterion(output, label)
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, NOISE_DIMENSION, 1, 1).cuda()
        fake = netG(noise)
        label = torch.zeros((batch_size,)).cuda()  # fake label = 1
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        D_G_z1 = output.mean().item()

        # Loss backward
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        ########################################
        # (2) Update G network: maximize log(D(G(z))) #
        ########################################
        netG.zero_grad()
        label = torch.ones(
            (batch_size,)
        ).cuda()  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        D_G_z2 = output.mean().item()

        errG.backward()
        optimizerG.step()

    print(
        "[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
        % (epoch, n_epoch, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
    )

    # save the output
    fake = netG(fixed_noise)
    training_progress_images_list = save_gif(
        training_progress_images_list, fake, model_name
    )  # Save fake image while training!

    # Check pointing for every epoch
    torch.save(
        netG.state_dict(),
        Path.cwd() / "assets" / "checkpoint" / f"{model_name}_netG_epoch_{epoch}.pth",
    )
    torch.save(
        netD.state_dict(),
        Path.cwd() / "assets" / "checkpoint" / f"{model_name}_netD_epoch_{epoch}.pth",
    )
