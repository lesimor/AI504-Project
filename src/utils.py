import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from torchvision.utils import make_grid
import imageio
from IPython.display import HTML
from pathlib import Path
import torchvision.utils as utils

# visualize the first image from the torch tensor


def vis_image(image):
    plt.imshow(image[0].detach().cpu().numpy(), cmap="gray")
    plt.show()


def save_gif(training_progress_images, images, model_name):
    """
    training_progress_images: list of training images generated each iteration
    images: image that is generated in this iteration
    """
    img_grid = make_grid(images.data)
    img_grid = np.transpose(img_grid.detach().cpu().numpy(), (1, 2, 0))
    img_grid = 255.0 * img_grid
    img_grid = img_grid.astype(np.uint8)
    training_progress_images.append(img_grid)
    imageio.mimsave(
        Path.cwd() / "assets" / "gif" / f"{model_name}_training_progress.gif",
        training_progress_images,
    )
    return training_progress_images


# visualize gif file
def vis_gif(training_progress_images):
    fig = plt.figure()

    ims = []
    for i in range(len(training_progress_images)):
        im = plt.imshow(training_progress_images[i], animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    html = ani.to_html5_video()
    HTML(html)
