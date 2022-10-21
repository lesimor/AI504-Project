import torch.nn as nn
from constants import *


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(NOISE_DIMENSION, DCGAN_NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(DCGAN_NGF * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(DCGAN_NGF * 8, DCGAN_NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DCGAN_NGF * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(DCGAN_NGF * 4, DCGAN_NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DCGAN_NGF * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(DCGAN_NGF * 2, DCGAN_NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DCGAN_NGF),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(DCGAN_NGF, DCGAN_NC, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output
