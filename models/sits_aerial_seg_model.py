import torch
import timm
import os
from torch import nn
from utils.hparams import hparams
import torch.nn.functional as F
import torchvision.transforms as T
from timm.layers import create_conv2d
from models.fusion_module.aer_cross_sat_atts import FFCA
from models.encoders.tsvit_model import TSViT_global_attention_spatial_encoder
from models.decoders.unet_sat_decoder import UNetSatDecoder
from models.decoders.unet_aer_decoder import UNetAerDecoder


class SITSAerialSegmenter(nn.Module):
    def __init__(self, gaussian, config):
        super().__init__()
        self.gaussian = gaussian
        self.config = config
        self.backbone_dims = [
            config["models"]["t_convformer"]["embed_dim"] * 2**i
            for i in range(len(config["models"]["t_convformer"]["depths"]))
        ]

        # =========================
        # 1. SR-SITS Encoder
        # =========================
        self.sr_sits_enc = TSViT_global_attention_spatial_encoder()

        # Load pretrained weights (if provided)
        if (
            hparams["cond_net_ckpt"]
            and os.path.exists(hparams["cond_net_ckpt"])
            and not hparams["infer"]
        ):
            self.load_pretrained_weights(self.sr_sits_enc, hparams["cond_net_ckpt"])

        # =========================
        # 2. Aerial Encoder (MaxViT)
        # =========================

        self.aer_net_enc = timm.create_model(
            "maxvit_tiny_tf_512.in1k",
            pretrained=True,
            features_only=True,
            num_classes=hparams["inputs"]["num_classes"],
        )

        # Adapt first conv layer (RGB → multi-channel)
        self._adapt_aerial_input_layer(hparams["inputs"]["num_channels_aer"])

        # Share encoder with latent diffusion module
        self.gaussian.aer_net_enc = self.aer_net_enc

        # =========================
        # 3. Decoders
        # =========================

        encoder_channels = [
            config["models"]["t_convformer"]["embed_dim"],
            config["models"]["t_convformer"]["embed_dim"] * 2,
            config["models"]["t_convformer"]["embed_dim"] * 4,
            config["models"]["t_convformer"]["embed_dim"] * 8,
        ]

        # 3. SR-SITS Decoder from U-NetFormer paper (USED ON DURING TRAINING)
        self.sr_sits_dec = UNetSatDecoder(
            encoder_channels,
            config["models"]["maxvit"][
                "decoder_channels"
            ],  #  remove the last channels dim.
            config["models"]["maxvit"]["dropout"],
            config["models"]["maxvit"]["window_cond_size"],
            config["inputs"]["num_classes"],
        )

        # 4. Aerial Decoder from U-Net Former paper
        self.aer_net_dec = UNetAerDecoder(
            encoder_channels,
            config["models"]["maxvit"]["decoder_channels"],
            config["models"]["maxvit"]["dropout"],
            config["models"]["maxvit"]["window_size"],
            config["inputs"]["num_classes"],
        )

        self.fusion_module = FFCA(
            aer_channels=128,
            sits_channels=128,
            num_heads=8,
        )

    def _adapt_aerial_input_layer(self, in_channels):
        """Modify first conv layer to accept multi-channel input."""
        conv1 = self.aer_net_enc.stem.conv1

        new_conv = create_conv2d(
            in_channels=in_channels,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=1,
            bias=conv1.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = conv1.weight
            nn.init.kaiming_normal_(new_conv.weight[:, 3:])

        self.aer_net_enc.stem.conv1 = new_conv

    def _strip_prefix(self, state_dict, prefix="model.backbone."):
        """Remove prefix from checkpoint keys."""
        return {
            k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
        }

    def _load_checkpoint(self, path):
        """Load checkpoint and extract state_dict."""
        ckpt = torch.load(path, map_location="cpu")
        return ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    def load_pretrained_weights(self, model, weights_path):
        print(f"\nLoading pretrained weights from: {weights_path}")

        pretrained_dict = self._load_checkpoint(weights_path)
        pretrained_dict = self._strip_prefix(pretrained_dict)

        model_dict = model.state_dict()

        # 🔍 Analyze BEFORE loading
        self.analyze_weight_loading(model_dict, pretrained_dict)

        # ✅ Keep only safe weights
        safe_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        model_dict.update(safe_dict)
        model.load_state_dict(model_dict)

        print(
            f"Loaded {len(safe_dict)}/{len(model_dict)} "
            f"({len(safe_dict)/len(model_dict):.2%}) parameters"
        )

    def analyze_weight_loading(self, model_dict, pretrained_dict):
        matched = []
        shape_mismatch = []
        missing = []
        unexpected = []

        # Check pretrained → model

        for k, v in pretrained_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    matched.append(k)
                else:
                    shape_mismatch.append((k, v.shape, model_dict[k].shape))
            else:
                unexpected.append(k)

        # Check model → pretrained
        for k in model_dict.keys():
            if k not in pretrained_dict:
                missing.append(k)

        # ===== PRINT REPORT =====
        print("\n========== WEIGHT LOADING REPORT FOR SR Net ==========")
        print(f"Matched: {len(matched)}")
        print(f"Shape mismatch: {len(shape_mismatch)}")
        print(f"Missing: {len(missing)}")
        print(f"Unexpected: {len(unexpected)}")

        # ---- Shape mismatch details ----
        if shape_mismatch:
            print("\n--- Shape Mismatches ---")
            for k, s1, s2 in shape_mismatch[:20]:
                print(f"{k}")
                print(f"  checkpoint: {s1}")
                print(f"  model     : {s2}")

        # ---- Missing keys ----
        if missing:
            print("\n--- Missing Keys (not found in checkpoint) ---")
            for k in missing[:20]:
                print(k)

        # ---- Unexpected keys ----
        if unexpected:
            print("\n--- Unexpected Keys (not used) ---")
            for k in unexpected[:20]:
                print(k)

        print("===========================================\n")

        return matched, shape_mismatch, missing, unexpected

    # =========================
    # Forward
    # =========================

    def forward(
        self,
        aerial: torch.FloatTensor,
        img_sr: torch.FloatTensor,
        dates: torch.FloatTensor,
    ):
        h_hr, w_hr = aerial.size()[-2:]
        h_sr, w_sr = img_sr.size()[-2:]

        # SR-SITS branch

        ## Encoder
        sat_output, sat_feat = self.sr_sits_enc(img_sr, dates)
        ## Decoder (USE ONLY DURING TRAINING)
        # print("red_temp_feats 0:", red_temp_feats[0].shape)
        # print("red_temp_feats 1:", red_temp_feats[1].shape)
        # print("red_temp_feats 2:", red_temp_feats[2].shape)
        # print("red_temp_feats 3:", red_temp_feats[3].shape)

        # Aerial branch
        hr_0, hr_1, hr_2, hr_3, hr_4 = self.aer_net_enc(aerial)

        # print("---------------SR Reduced Temp Feats-------------------")
        # print("red_temp_feats 0:", red_temp_feats[0].shape)
        # print("red_temp_feats 1:", red_temp_feats[1].shape)
        # print("red_temp_feats 2:", red_temp_feats[2].shape)

        # # red_temp_feats 0: torch.Size([2, 64, 64, 64])
        # # red_temp_feats 1: torch.Size([2, 128, 32, 32])
        # # red_temp_feats 2: torch.Size([2, 256, 16, 16])
        # print()

        # print("--------------- Multi res SR Feats-------------------")
        # print("multi_lvls_cls 0:", multi_lvls_cls[0].shape)
        # print("multi_lvls_cls 1:", multi_lvls_cls[1].shape)
        # print("multi_lvls_cls 2:", multi_lvls_cls[2].shape)
        # # multi_lvls_cls 0: torch.Size([2, 13, 64, 64])
        # # multi_lvls_cls 1: torch.Size([2, 13, 64, 64])
        # # multi_lvls_cls 2: torch.Size([2, 13, 64, 64])

        # Fusion FFCA
        fus_2 = self.fusion_module(hr_2, sat_feat)

        # print("---------------SR Reduced Temp Feats-------------------")
        # print("fusion outputs 2:", res2.shape)
        # print("fusion outputs 3:", res3.shape)
        # print("fusion outputs 4:", res4.shape)
        # # fusion outputs 2: torch.Size([2, 128, 64, 64])
        # # fusion outputs 3: torch.Size([2, 256, 32, 32])
        # # fusion outputs 4: torch.Size([2, 512, 16, 16])
        # print()

        # Decoder
        logits = self.aer_net_dec(hr_0, hr_1, fus_2, hr_3, hr_4, h_hr, w_hr)
        return sat_output, logits


# Description of GRID attention introduced in TConvFormer

#  Imagine you have a 6×6 image, and you want each pixel to "see" other pixels globally.

# Step 1: Split into grid

# Divide the 6×6 image into 2×2 grids, so you have 9 grids in total. Each grid has 2×2 pixels.

# Step 2: Grid-attention with dilation

# Instead of computing attention for all 36 pixels (which is expensive), you:
# First compute local attention within each grid (2×2 → small and fast).
# Then compute attention across grids, but using dilated connections (e.g., only attend to every 2nd grid in each direction).
# This way, even distant pixels can influence each other, without doing full 36×36 attention.
