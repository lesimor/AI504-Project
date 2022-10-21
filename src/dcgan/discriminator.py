import torch.nn as nn
from constants import *


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(DCGAN_NC, DCGAN_NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(DCGAN_NDF, DCGAN_NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DCGAN_NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(DCGAN_NDF * 2, DCGAN_NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DCGAN_NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(DCGAN_NDF * 4, DCGAN_NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DCGAN_NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(DCGAN_NDF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
