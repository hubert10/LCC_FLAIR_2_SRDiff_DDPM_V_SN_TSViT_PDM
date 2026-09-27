#!/usr/bin/env python
# coding: utf-8

import os
import random
from glob import glob
import cv2
import re
import torch
import rasterio
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from PIL import Image
from data.data_display import convert_to_color, lut_colors
from utils.hparams import hparams, set_hparams


try:
    # DATA_DIR = os.environ["DATA_DIR"] + "/"
    DATA_DIR = "/hubert_storage"
except Exception:
    DATA_DIR = "D:\kanyamahanga\Datasets"


def load_hr_raster(img_id, rel_path, img_root):
    """
    Load a raster mask from a given image ID and relative path.

    Parameters:
    - img_id: str, e.g. "036385"
    - rel_path: str, e.g. "D055_2018/Z3_UF"
    - msk_root: str, base folder where masks are stored

    Returns:
    - mask: numpy array of the mask
    - profile: rasterio profile (metadata)
    - path: full path to the mask
    """
    # img_filename = f"{img_id}".replace(".png", "tif").replace()
    input_path = os.path.join(rel_path, img_id)

    print("++++++++++input_path+++++++++++++:", input_path)

    # Extract base parts
    folder = input_path.split(os.sep)[0]  # D083-2021
    filename = os.path.basename(input_path)  # IMG_UU-S1-24_4-2.png

    # Remove prefix and extension
    core = filename.replace("IMG_", "").replace(".png", "")  # UU-S1-24_4-2

    tile = core.split("_")[0]  # UU-S1-24

    # Construct output path
    image_path = (
        f"FLAIR_HUB_TOY/{folder}_AERIAL_RGBI/{tile}/{folder}_AERIAL_RGBI_{core}.tif"
    )

    print("++++++++++img_path+++++++++++++:", image_path)
    # FLAIR_HUB_TOY/D038-2021_AERIAL_LABEL-COSIA/UU-S1-24/D038-2021_AERIAL_LABEL-COSIA_UU-S1-24_4-2.tif
    channels = [1, 2, 3]
    with rasterio.open(os.path.join(DATA_DIR, image_path)) as src_img:
        array = src_img.read(channels) if channels else src_img.read()
    return array


def load_mask_raster(img_id, rel_path, msk_root):
    """
    Load a raster mask from a given image ID and relative path.

    Parameters:
    - img_id: str, e.g. "036385"
    - rel_path: str, e.g. "D055_2018/Z3_UF"
    - msk_root: str, base folder where masks are stored

    Returns:
    - mask: numpy array of the mask
    - profile: rasterio profile (metadata)
    - path: full path to the mask
    """
    input_path = os.path.join(rel_path, img_id)

    print("++++++++++input_path+++++++++++++:", input_path)

    # Extract base parts
    folder = input_path.split(os.sep)[0]  # D083-2021
    filename = os.path.basename(input_path)  # IMG_UU-S1-24_4-2.png

    # Remove prefix and extension
    core = filename.replace("IMG_", "").replace(".png", "")  # UU-S1-24_4-2

    tile = core.split("_")[0]  # UU-S1-24

    # Construct output path
    image_path = f"FLAIR_HUB_TOY/{folder}_AERIAL_LABEL-COSIA/{tile}/{folder}_AERIAL_LABEL-COSIA_{core}.tif"

    print("++++++++++img_path+++++++++++++:", image_path)
    # FLAIR_HUB_TOY/D038-2021_AERIAL_LABEL-COSIA/UU-S1-24/D038-2021_AERIAL_LABEL-COSIA_UU-S1-24_4-2.tif
    channels = [1]
    with rasterio.open(os.path.join(DATA_DIR, image_path)) as src_img:
        array = src_img.read(channels) if channels else src_img.read()
    return torch.tensor(array)


def downsample_majority_vote_no_crop(labels, scale_factor=8):
    """
    Downsamples multi-class label maps using majority vote without cropping.

    Args:
        labels (torch.Tensor): Input label maps of shape [N, H, W] (e.g., [N, 512, 512]).
        scale_factor (int): Factor by which to downsample (e.g., 8 for 20cm → 1.6m).

    Returns:
        torch.Tensor: Downsampled label maps of shape [N, H//scale_factor, W//scale_factor].
    """
    N, H, W = labels.shape
    assert (
        H % scale_factor == 0 and W % scale_factor == 0
    ), f"Height and Width must be divisible by scale_factor={scale_factor}"

    output_size = H // scale_factor  # e.g., 512 // 8 = 64
    block_size = scale_factor  # 8

    # Step 1: Reshape to blocks
    labels = labels.view(N, output_size, block_size, output_size, block_size)

    # Step 2: Permute to group pixels in each block
    labels = labels.permute(0, 1, 3, 2, 4)  # [N, out_h, out_w, block_h, block_w]

    # Step 3: Flatten block pixels
    labels = labels.reshape(N, output_size, output_size, block_size * block_size)

    # Step 4: Apply majority vote
    mode, _ = torch.mode(labels, dim=-1)

    return mode  # [N, H//scale_factor, W//scale_factor]


def plot_random_hr_lr_sr(hr_root, lr_root, sr_root):
    # Step 1: Randomly select an HR image
    hr_images = glob(os.path.join(hr_root, "**", "img", "*.png"), recursive=True)
    if not hr_images:
        print("No HR images found.")
        return

    hr_img_path = random.choice(hr_images)
    # print(f"Selected HR image: {hr_img_path}")

    # Step 2: Extract ID and corresponding LR/SR folders
    base_name = os.path.basename(hr_img_path)  # e.g., IMG_077413.png
    img_id = base_name  # .split("_")[-1].split(".")[0]  # '077413'

    # Get the relative path after HR root (e.g., D015_2020/Z1_AA)
    rel_path = os.path.relpath(os.path.dirname(os.path.dirname(hr_img_path)), hr_root)

    lr_folder = os.path.join(
        lr_root, rel_path, "sen", img_id.replace("IMG_", "").replace(".png", "")
    )
    sr_folder = os.path.join(
        sr_root, rel_path, "sen", img_id.replace("IMG_", "").replace(".png", "")
    )

    # Step 3: Load images
    hr_img = load_hr_raster(img_id, rel_path, img_root_folder)
    hr_img = np.transpose(hr_img, (1, 2, 0))
    down_hr_img = cv2.imread(hr_img_path)

    # print(f"down_hr_img: {hr_img_path}")

    down_hr_img = (
        cv2.cvtColor(down_hr_img, cv2.COLOR_BGR2RGB)
        if down_hr_img is not None
        else None
    )

    # Load masks
    msk = load_mask_raster(img_id, rel_path, msk_root_folder)
    # print(" before downsample:", msk.shape)
    down_msk = downsample_majority_vote_no_crop(msk)

    msk_color = convert_to_color(msk[0], palette=lut_colors)
    down_msk_color = convert_to_color(down_msk[0], palette=lut_colors)

    # print("HR image:", hr_img.shape)
    # print("HR mask:", msk.shape)
    # print("Downsampled HR image:", down_hr_img.shape)
    # print("Downsampled mask:", down_msk_color.shape)

    # Load LR and SR images
    lr_imgs = sorted(glob(os.path.join(lr_folder, "*.png")))[:4]
    sr_imgs = sorted(glob(os.path.join(sr_folder, "*.png")))[:4]

    # print(f"Selected LR image: {lr_imgs}")
    # print(f"Selected SR image: {sr_imgs}")

    lr_images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in lr_imgs]
    sr_images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in sr_imgs]

    num_timesteps = max(len(lr_images), len(sr_images))
    fig_rows = 3
    fig_cols = max(num_timesteps, 4)  # make space for 4 items in first row

    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 10))

    # Row 0: HR, HR MSK, downsampled HR, downsampled MSK
    first_row_imgs = [hr_img, msk_color, down_hr_img, down_msk_color]
    titles = ["HR Image", "HR MSK", "Downsampled HR", "Downsampled MSK"]
    for i in range(4):
        axs[0, i].imshow(first_row_imgs[i])
        axs[0, i].axis("off")
        axs[0, i].set_title(titles[i])
    for i in range(4, fig_cols):  # blank out remaining cells
        axs[0, i].axis("off")

    # Row 1: LR
    for i, img in enumerate(lr_images):
        # print("--------------img-----------:", img.shape)
        # min_val = np.min(img)
        # max_val = np.max(img)
        # print(min_val, max_val)

        axs[1, i].imshow(img)
        axs[1, i].axis("off")
        axs[1, i].set_title(f"LR {i}")
    for i in range(len(lr_images), fig_cols):
        axs[1, i].axis("off")

    # Row 2: SR
    for i, img in enumerate(sr_images):
        # print("--------------img-----------:", img.shape)

        axs[2, i].imshow(img)
        axs[2, i].axis("off")
        axs[2, i].set_title(f"SR {i}")
    for i in range(len(sr_images), fig_cols):
        axs[2, i].axis("off")

    plt.tight_layout()
    plt.show()


# root_folder = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aer_LCC_x10_JOINT_SRDiff_SEG_SegFormer_HR_ConvFormer_SR_NIR_OPT_DATA_AUG_LPIPS\\"

root_folder = "D:\\kanyamahanga\\Bigwork\\LCC_FLAIR_HUB_SRDiff_DDPM_V_SN_TSViT_PDM\\"
root_folder = "C:\\Users\\kanyamahanga\\Desktop\\IPI-128\\RESEARCH\\Journal_Paper\\LCC_FLAIR_HUB_SRDiff_DDPM_V_SN_TSViT_PDM\\results\\checkpoints\\misr\\srdiff_maxvit_ltae_ckpt\\results_0_\\"

img_root_folder = "D:\\kanyamahanga\\Datasets\\FLAIR_HUB_TOY\\"
msk_root_folder = "D:\\kanyamahanga\\Datasets\\FLAIR_HUB_TOY\\"

# val_csv: D:\\kanyamahanga\\Datasets\\FLAIR_HUB_TOY\\ALL_CSV\\FLAIR_HUB_DATASET_VALID.csv
# test_csv: D:\\kanyamahanga\\Datasets\\FLAIR_HUB_TOY\\ALL_CSV\\FLAIR_HUB_DATASET_TEST.csv

plot_random_hr_lr_sr(
    hr_root=root_folder + "HR", lr_root=root_folder + "LR", sr_root=root_folder + "SR"
)
