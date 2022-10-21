import os
import torch
import torch.nn as nn
from pathlib import Path
import torchvision.utils as utils
from dcgan.generator import Generator as DCGanGenerator

from constants import BATCH_SIZE, NOISE_DIMENSION, NOISE_WIDTH


if __name__ == "__main__":
    # 제너레이터 생성
    generator = DCGanGenerator().cuda()
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
        noise = torch.randn(BATCH_SIZE, NOISE_DIMENSION, 1, 1).cuda()
        generated_image = generator(noise)[0]
        utils.save_image(generated_image.cuda().detach(), saving_path / f"{x}.png")
