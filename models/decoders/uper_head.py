import math
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta
from utils.base_layers import ConvModule
from utils.utils_models import resize
from models.decoders.ppm import PPM


class UPerHead(nn.Module, metaclass=ABCMeta):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    Paper: <https://arxiv.org/abs/1807.10221>`_.
    Code : <https://github.com/CSAILVision/unifiedparsing>

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(
        self,
        in_channels,
        channels,
        num_classes,
        in_index=-1,
        input_transform=None,
        pool_scales=(1, 2, 3, 4),
        dropout_ratio=0.1,
        align_corners=False,
        **kwargs,
    ):
        super().__init__()
        # PSP Module
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        self.pool_scales = pool_scales
        # print("---------------------------------------------:", in_channels[-1])
        # print("---------------------------------------------:", self.channels)

        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,  # kernel size
            padding=1,
        )

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(in_channels, self.channels, 1, inplace=False)
            fpn_conv = ConvModule(
                self.channels, self.channels, 3, padding=1, inplace=False
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
        )
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]

        # print("psp_outs x0:", psp_outs[0].shape)

        psp_outs.extend(self.psp_modules(x))

        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        # print("output conc:", output.shape)

        return output

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
            output = self.conv_seg(feat)
            return output

    def forward(self, inputs):
        """Forward function."""
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # print("inputs 0:", inputs[0].shape)
        # print("inputs 1:", inputs[1].shape)
        # print("inputs 2:", inputs[2].shape)
        # print("inputs 3:", inputs[3].shape)

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        laterals_out = laterals.copy()

        for i in range(used_backbone_levels - 1, -1, -1):
            if i == used_backbone_levels - 1:
                laterals_out[i] = laterals[i]
            else:
                prev_shape = laterals[i].shape[-2:]
                laterals_out[i] = laterals[i] + resize(
                    laterals[i + 1],
                    size=prev_shape,
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
        # Starting from
        # fpn_outs[i]: torch.Size([4, 512, 2, 2])
        # fpn_outs[i]: torch.Size([4, 512, 3, 3])
        # fpn_outs[i]: torch.Size([4, 512, 5, 5])

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals_out[i]) for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals_out[-1])

        # The fusion block fuses P1, P2, P3, and P4 to generate the fused features with 1/4
        # resolution and applies a convolution layer (self.cls_seg) to map the dimensions to the category numbers
        # for the purpose to obtain the final segmentation map
        # From EfficientNet paper: Page 6

        # UPSAMPLE all low levels features  EXCPET the
        # first feature map(already at the 10m GSD) to
        # the 10m resolution with "bilinear interpolation"

        for i in range(used_backbone_levels - 1, 0, -1):
            # print("fpn_outs[i]:", i)
            # print("fpn_outs[i]:", fpn_outs[i].shape)
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        multi_levels_feature_maps = fpn_outs.copy()

        # print("multi_levels_feature_maps 0:", multi_levels_feature_maps[0].shape)
        # print("multi_levels_feature_maps 1:", multi_levels_feature_maps[1].shape)
        # print("multi_levels_feature_maps 2:", multi_levels_feature_maps[2].shape)
        # print("multi_levels_feature_maps 3:", multi_levels_feature_maps[3].shape)

        # CONCAT the multi_levels_feats_out using a conv layer: 2D + BN + Relu
        fpn_outs = torch.cat(fpn_outs, dim=1)

        # The combined multi_levels_feats_out maps at 10m GSD
        last_ft_map = self.fpn_bottleneck(fpn_outs)
        cls_sits_ft_map = self.cls_seg(last_ft_map)

        # For Auxiliary Loss calculations at each level of tge UperNet
        # Each feature map is classified at 10m resolution

        # STEP: UPSAMPLE ALL THE 4 FEATURES OF DIFFERENT RESOLUTIONS FROM 40m to 10m
        # WITH AN UPSAMPLING FACTOR OF 4X

        multi_lvls_cls = [
            self.cls_seg(feature) for feature in multi_levels_feature_maps
        ]

        # print("last_ft_map:", last_ft_map.shape)
        # print("cls_sits_ft_map:", cls_sits_ft_map.shape)

        # print("multi_lvls_cls 1:", multi_lvls_cls[1].shape)
        # print("multi_lvls_cls 2:", multi_lvls_cls[2].shape)
        # print("multi_lvls_cls 3:", multi_lvls_cls[3].shape)

        # cls_sits_ft_map: torch.Size([2, 19, 64, 64])

        # multi_lvls_cls 1: torch.Size([2, 19, 64, 64])
        # multi_lvls_cls 2: torch.Size([2, 19, 64, 64])
        # multi_lvls_cls 3: torch.Size([2, 19, 64, 64])

        # torch.Size([4, 512, 40, 40]), torch.Size([4, 13, 40, 40]),  torch.Size([4, 13, 40, 40], [4, 13, 40, 40], [4, 13, 40, 40], [4, 13, 40, 40])
        return cls_sits_ft_map, multi_lvls_cls
