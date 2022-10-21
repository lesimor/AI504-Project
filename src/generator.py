import os
import torch
import torch.nn as nn
from pathlib import Path
import torchvision.utils as utils

from constants import NOISE_WIDTH


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            #########################
            # Define your own generator #
            #########################
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 64),
            nn.Sigmoid(),  # 0 - 1
            #########################
        )

    def forward(self, input):
        #####################################
        # Change the shape of output if necessary #

        # input shape = batch_size, 100
        #####################################

        output = self.main(input)

        #####################################
        # Change the shape of output if necessary #

        # output shape = batch_size, 3, 64, 64
        output = output.view(-1, 3, 64, 64)

        #####################################
        return output


# 제너레이터 생성
generator = Generator().cuda()
ckpt_path = Path.cwd() / "assets" / "checkpoint"
netG_ckpt_name = os.getenv("CHECKPOINT_NAME")
if not netG_ckpt_name:
    raise Exception("You should provide CHECKPOINT_NAME variable")
generator.load_state_dict(torch.load(ckpt_path / netG_ckpt_name))

# 이미지 저장 경로 설정
saving_path = Path.cwd() / "assets" / "artifact" / "generated"
saving_path.mkdir(parents=True, exist_ok=True)

image_num = int(os.getenv("NUM_IMG", 100))
for x in range(image_num):
    noise = torch.randn(NOISE_WIDTH, 100).cuda()
    generated_image = generator(noise)[0]
    utils.save_image(generated_image.cuda().detach(), saving_path / f"{x}.png")
