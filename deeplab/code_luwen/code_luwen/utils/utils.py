from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import ndarray as NDArray
from torch import Tensor
import os


def mean_smoothing(amaps: Tensor, kernel_size: int = 21) -> Tensor:

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


def savefig(
    class_type,
    epoch: int,
    imgs: List[NDArray],
    reconsts: List[NDArray],
    masks: List[NDArray],
    amaps: List[NDArray],
    augs: List[NDArray],
) -> None:

    for i, (img, reconst, mask, amap, aug) in enumerate(zip(imgs, reconsts, masks, amaps, augs)):

        # How to get two subplots to share the same y-axis with a single colorbar
        # https://stackoverflow.com/a/38940369
        grid = ImageGrid(
            fig=plt.figure(figsize=(16, 4)),
            rect=111,
            nrows_ncols=(1, 5),
            axes_pad=0.15,
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.15,
        )

        img = denormalize(img)
        reconst = denormalize(reconst)
        aug = denormalize(aug)

        grid[0].imshow(img)
        grid[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[0].set_title("Input Image", fontsize=14)

        grid[1].imshow(aug)
        grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[1].set_title("Aug Image", fontsize=14)

        grid[2].imshow(reconst)
        grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[2].set_title("Reconstructed Image", fontsize=14)

        grid[3].imshow(img)
        grid[3].imshow(mask[0], alpha=0.3, cmap="Reds")
        grid[3].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[3].set_title("Annotation", fontsize=14)

        grid[4].imshow(img)
        im = grid[4].imshow(amap, alpha=0.3, cmap="jet")
        grid[4].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[4].cax.colorbar(im)
        grid[4].cax.toggle_label(True)
        grid[4].set_title("Anomaly Map", fontsize=14)
        os.makedirs(f"epochs/{class_type}/{epoch}", exist_ok=True)
        plt.savefig(f"epochs/{class_type}/{epoch}/{i}.png", bbox_inches="tight")
        plt.close()


def denormalize(img: NDArray) -> NDArray:

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # mean = np.array([0.5, 0.5, 0.5])
    # std = np.array([0.5, 0.5, 0.5])
    img = (img * std + mean)*255
    return img.astype(np.uint8)
