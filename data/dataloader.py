from typing import Dict
from torch.utils.data import Dataset
from data.utils_data.io import read_patch
from data.utils_data.norm import norm
from data.utils_data.augmentations import apply_numpy_augmentations
from data.utils_data.label import reshape_label_ohe
from data.utils_data.elevation import calc_elevation
from data.utils_data.sentinel import (
    reshape_sentinel,
    filter_time_series,
    temporal_average,
)
import os
import torch
import json
import rasterio
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
from utils import utils_dataset
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F


class FLAIRDataSet(Dataset):
    """
    PyTorch Dataset for multimodal remote sensing data.
    Applies normalization, temporal aggregation, cloud filtering, and augmentations.
    Args:
        config (Dict): Configuration dictionary containing model and modality-specific settings, such as
            input channels, normalization parameters, and temporal processing options.
        dict_paths (Dict): A dictionary mapping data modalities (e.g., "AERIAL_RGBI", "LABELS")
            to their corresponding file paths.
        use_augmentations (callable, optional): A callable function or transformation pipeline to
            apply augmentations to the samples. Defaults to None.
     Methods:
        __len__() -> int:
            Returns the number of samples in the dataset.
        __getitem__(index: int) -> Dict:
            Retrieves batch_elements. Applies
            normalization, temporal aggregation, and augmentations (if specified).

    """

    def __init__(
        self, config: Dict, dict_paths: Dict, use_augmentations: callable = None
    ) -> None:
        self.config = config
        if use_augmentations is True:
            self.use_augmentations = apply_numpy_augmentations
        else:
            self.use_augmentations = use_augmentations

        # Data and label setup (same as before)
        self._init_data_paths(dict_paths)
        self._init_label_info(dict_paths)
        self._init_normalization()
        self.ref_date = config["inputs"]["ref_date"]

    def _init_data_paths(self, dict_paths):
        self.list_patch = {}
        enabled = self.config["modalities"]["inputs"]
        for mod, enabled_flag in enabled.items():
            if enabled_flag and mod in dict_paths:
                self.list_patch[mod] = np.array(dict_paths[mod])
                if mod == "SENTINEL2_TS":
                    self.list_patch["SENTINEL2_MSK-SC"] = np.array(
                        dict_paths["SENTINEL2_MSK-SC"]
                    )

        self.dict_dates = {}
        self.dict_spot_dates = {}

        if "SENTINEL2_TS" in enabled:
            self.dict_dates["SENTINEL2_TS"] = dict_paths.get("DATES_S2", {})

        if "SPOT_RGBI" in enabled:
            self.dict_spot_dates["SPOT_RGBI"] = dict_paths.get("DATES_S6", {})

    def _init_label_info(self, dict_paths):
        self.tasks = {}
        for task in self.config["labels"]:
            label_conf = self.config["labels_configs"][task]
            self.tasks[task] = {
                "data_paths": np.array(dict_paths[task]),
                "num_classes": len(label_conf["value_name"]),
                "channels": [label_conf.get("label_channel_nomenclature", 1)],
            }

    def _init_normalization(self):
        self.norm_type = self.config["modalities"]["normalization"]["norm_type"]
        enabled_modalities = self.config["modalities"]["inputs"]
        self.channels = {
            mod: self.config["modalities"]["inputs_channels"].get(mod, [])
            for mod, active in enabled_modalities.items()
            if active
        }
        self.normalization = {
            mod: {
                "mean": self.config["modalities"]["normalization"].get(
                    f"{mod}_means", []
                ),
                "std": self.config["modalities"]["normalization"].get(
                    f"{mod}_stds", []
                ),
            }
            for mod, active in enabled_modalities.items()
            if active
        }

    def upsample_sits_image(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - input_tensor (torch.Tensor): Tensor of shape (C, H, W), e.g., (3, 10, 10).
        - upsample_factor (int): Upsampling factor, e.g., 6.4 → (6.4x spatial resolution).

        Returns:
        - torch.Tensor: upsampled tensor of shape (C, up_H, up_W), e.g.,
          (3, 64, 64)
        """
        # Ensure 3D shape (C, H, W)
        assert input_tensor.ndim == 3, "Input tensor must be (C, H, W)"

        # Add a batch dimension before interpolation (Shape: (1, C, H, W))
        input_tensor = input_tensor.unsqueeze(0)

        # Upsample using bicubic interpolation
        up_tensor = F.interpolate(
            input_tensor,
            size=(64, 64),
            mode="bicubic",
            align_corners=False,
        )
        return up_tensor.squeeze(0)  # shape: (C, up_H, up_W)

    def sat_scale_norm(self, img_lr):
        # scale to [0, 1] and normalize to [-1, 1]
        scaled_img_lr = img_lr / 2_000
        scaled_img_lr = torch.clamp(scaled_img_lr, 0, 1)
        return 2 * scaled_img_lr - 1

    def downsample_single_label_map_majority_vote(self, label: torch.Tensor):
        """
        Downsamples N label maps from 20 cm GSD to 1.6 m GSD
        using majority vote on non-overlapping 8x8 blocks.

        Args:
            label (torch.Tensor): Tensor of shape [N, H, W].
                                H and W must be divisible by 8.

        Returns:
            torch.Tensor: Downsampled label maps of shape [N, H//8, W//8].
        """
        scale_factor = 8
        N, H, W = label.shape
        assert (
            H % scale_factor == 0 and W % scale_factor == 0
        ), f"H and W must be divisible by {scale_factor}"

        h_out = H // scale_factor
        w_out = W // scale_factor

        # Reshape into (N, h_out, 8, w_out, 8)
        label_reshaped = label.view(N, h_out, scale_factor, w_out, scale_factor)

        # Move block dims together -> (N, h_out, w_out, 64)
        label_blocks = label_reshaped.permute(0, 1, 3, 2, 4).reshape(
            N, h_out, w_out, scale_factor * scale_factor
        )

        # Majority vote for each block (N, h_out, w_out)
        mode, _ = torch.mode(label_blocks, dim=-1)

        return mode

    def combine_hr_and_dem(
        self, img: torch.Tensor, dem_elev: torch.Tensor
    ) -> torch.Tensor:
        """
        Combines HR image [4, H, W] with DEM elevation channel [1, H, W]
        to produce [5, H, W].
        """
        # Extract first DEM channel
        dem = dem_elev[0:1, :, :]  # shape: [B, 1, 512, 512]
        # Concatenate along channel dimension
        img_hr = torch.cat([img, dem], dim=0)  # -> [B, 5, 512, 512]
        return img_hr

    def date_to_day_of_year(self, given_date):
        """Convert a given date to day of year"""
        from datetime import datetime

        try:
            # string date format
            day_of_year = datetime.strptime(given_date, "%Y-%m-%d").timetuple().tm_yday
        except TypeError:
            # date type format
            day_of_year = given_date.timetuple().tm_yday

        # print("\nDay of year: ", day_of_year, "\n")
        return day_of_year

    def __len__(self):
        for task in self.tasks.values():
            if len(task["data_paths"]) > 0:
                return len(task["data_paths"])
        return 0

    def __getitem__(self, index):
        batch = {}

        # Supervision
        for task, info in self.tasks.items():
            batch[f"ID_{task}"] = info["data_paths"][index]
            img = info["data_paths"][index].split("/")[-1].split("_")
            area_elem = "_".join([img[0], img[-2], img[-1].split(".")[0]])
            batch["item_name"] = batch[f"ID_{task}"]

        # AERIAL_RGBI
        key = "AERIAL_RGBI"
        if key in self.list_patch:
            data = read_patch(self.list_patch[key][index], self.channels[key])
            batch[key] = norm(
                data,
                self.norm_type,
                self.normalization[key]["mean"],
                self.normalization[key]["std"],
            )

        # DEM_ELEV
        key = "DEM_ELEV"
        if key in self.list_patch and self.list_patch[key][index] is not None:
            zdata = read_patch(self.list_patch[key][index])
            if self.config["modalities"]["pre_processings"]["calc_elevation"]:
                elev_data = calc_elevation(zdata)
                if self.config["modalities"]["pre_processings"][
                    "calc_elevation_stack_dsm"
                ]:
                    elev_data = np.stack((zdata[0, :, :], elev_data[0]), axis=0)
                batch[key] = elev_data
            else:
                batch[key] = zdata
            batch[key] = norm(
                batch[key],
                self.norm_type,
                self.normalization[key]["mean"],
                self.normalization[key]["std"],
            )

        # SPOT_RGBI
        key = "SPOT_RGBI"
        if key in self.list_patch:
            data = read_patch(self.list_patch[key][index], self.channels[key])
            s6_date_dict = self.dict_spot_dates[key][area_elem]
            s6_date = s6_date_dict["spot_dates"][0]
            batch[key] = norm(
                data,
                self.norm_type,
                self.normalization[key]["mean"],
                self.normalization[key]["std"],
            )
            batch["SPOT6_DATE"] = s6_date

        # SENTINEL2_TS
        key = "SENTINEL2_TS"
        if key in self.list_patch:
            s2 = read_patch(self.list_patch[key][index])
            s2 = reshape_sentinel(s2, chunk_size=10)[
                :, [x - 1 for x in self.channels[key]], :, :
            ]

            s2_dates_dict = self.dict_dates[key][area_elem]
            s2_dates = s2_dates_dict["dates"]

            if (
                self.config["modalities"]["pre_processings"]["temporal_diff"]
                == "default"
            ):
                s2_dates_diff = s2_dates_dict["diff_dates"]

            # use the spot6 image as a reference
            else:
                s2_dates_diff = np.array(
                    [(date_sen2 - s6_date).days for date_sen2 in s2_dates]
                )

            # Filter Clouds and Snows
            if self.config["modalities"]["pre_processings"]["filter_sentinel2"]:
                msk = read_patch(self.list_patch["SENTINEL2_MSK-SC"][index])
                msk = reshape_sentinel(msk, chunk_size=2)
                idx_valid = filter_time_series(
                    msk,
                    max_cloud_value=self.config["modalities"]["pre_processings"][
                        "filter_sentinel2_max_cloud"
                    ],
                    max_snow_value=self.config["modalities"]["pre_processings"][
                        "filter_sentinel2_max_snow"
                    ],
                    max_fraction_covered=self.config["modalities"]["pre_processings"][
                        "filter_sentinel2_max_frac_cover"
                    ],
                )
                s2 = s2[np.where(idx_valid)[0]]
                s2_dates = s2_dates[np.where(idx_valid)[0]]
                s2_dates_diff = s2_dates_diff[np.where(idx_valid)[0]]

            # Filter Monthly averages

            if self.config["modalities"]["pre_processings"][
                "temporal_average_sentinel2"
            ]:
                if (
                    self.config["modalities"]["pre_processings"]["temporal_diff"]
                    == "default"
                ):
                    ref_date = self.ref_date
                else:
                    ref_date = s6_date.strftime("%m-%d")

                s2, s2_dates_diff, average_dates = temporal_average(
                    s2,
                    s2_dates,
                    period=self.config["modalities"]["pre_processings"][
                        "temporal_average_sentinel2"
                    ],
                    ref_date=ref_date,
                    nbts=self.config["inputs"]["nbts"],
                )

            # print("------------s2_dates------------:", s2_dates)
            batch[key] = s2
            # batch[key.replace("_TS", "_DATES")] = [self.date_to_day_of_year(x) for x in s2_dates]
            batch[key.replace("_TS", "_DATES")] = average_dates

        # Labels
        for task, info in self.tasks.items():
            label = read_patch(info["data_paths"][index], info["channels"])
            batch[task] = reshape_label_ohe(label, info["num_classes"])

        # Apply numpy augmentations
        if callable(self.use_augmentations):
            input_keys = [
                k for k, v in self.config["modalities"]["inputs"].items() if v
            ]
            label_keys = list(self.config["labels"])
            batch = self.use_augmentations(batch, input_keys, label_keys)

        # Convert to torch tensors
        batch = {
            k: (
                torch.tensor(v, dtype=torch.float32)
                if isinstance(v, (np.ndarray, list)) and "ID_" not in k
                else v
            )
            for k, v in batch.items()
        }

        # print("---------------------s2_dates---------------------:",  s2_dates)

        from datetime import datetime

        # Convert all items to datetime objects (if not already), then format them
        dates = [
            (
                date_sen2
                if isinstance(date_sen2, datetime)
                else datetime.strptime(date_sen2, "%Y-%m-%d")
            ).strftime("%Y-%m-%d")
            for date_sen2 in s2_dates
        ]

        rename_map = {
            "AERIAL_RGBI": "img_hr",
            "DEM_ELEV": "dem_elev",
            "SPOT_RGBI": "img_hr_down",
            "SENTINEL2_TS": "img_lr",
            "SENTINEL2_DATES": "dates",
            "SPOT6_DATE": "spot_date",
            "AERIAL_LABEL-COSIA": "labels",
        }
        batch = {rename_map.get(k, k): v for k, v in batch.items()}
        # we use the positional encoding of the dates as proposed in the paper
        # create a new dict with renamed keys
        img_hr = self.combine_hr_and_dem(batch["img_hr"], batch["dem_elev"])

        img_lr = batch["img_lr"]
        img_lr_outs = []

        # Normalize the SITS bands as it not done like other data
        img_lr = self.sat_scale_norm(img_lr)

        for i in range(img_lr.shape[0]):
            img_lr_outs.append(self.upsample_sits_image(img_lr[i, :, :, :]))
        # torch.Size([2, 3, 40, 40]): T=2, C=3, H=40, W=40

        img_lr_up = torch.stack(img_lr_outs, 0)
        ind = np.argmin(np.abs(batch["dates"]))
        # Match the ground truth radiometry to the reference low resolution input
        # (reference LR inpt is the closest image to the HR)
        labels = torch.as_tensor(batch["labels"], dtype=torch.int32)
        batch["dates_encoding"] = torch.as_tensor(batch["dates"], dtype=torch.float)
        # batch["dates_encoding"] = torch.as_tensor([x / 365.0001 for x in batch["dates"]], dtype=torch.float)

        # print("xt dtype:", batch["dates_encoding"].dtype)
        # print("xt shape:", batch["dates_encoding"].shape)
        # print("xt min:", batch["dates_encoding"].min().item())
        # print("xt max:", batch["dates_encoding"].max().item())
        # print("unique xt:", torch.unique(batch["dates_encoding"]).cpu().tolist())

        # cropping the ground truth images
        labels_sr = self.downsample_single_label_map_majority_vote(labels).long()
        batch["img_lr"] = img_lr
        batch["img_hr"] = img_hr
        batch["img_lr_up"] = img_lr_up
        batch["closest_idx"] = ind
        batch["labels_sr"] = labels_sr
        batch["dates"] = dates  # added for saving time series images during inference
        # TODO: CHECK THE FORMAT OF batch["dates_encoding"]
        return batch
