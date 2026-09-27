import os.path
import json
import torch
import numpy as np
import torch.nn as nn
from trainer import Trainer
from utils.hparams import hparams
from utils.utils import load_ckpt
from models.denoiser.unet import UNet
from models.diffusion.ddpm import GaussianDiffusion
from models.sits_aerial_seg_model import SITSAerialSegmenter
from losses.focal_smooth import FocalLossWithSmoothing
from models.encoders.t_convformer import TConvFormer


class SRDiffTrainer(Trainer):
    def build_model(self):
        hidden_size = hparams["hidden_size"]
        dim_mults = hparams["unet_dim_mults"]
        dim_mults = [int(x) for x in dim_mults.split("|")]

        self.criterion_aer = FocalLossWithSmoothing(
            hparams["inputs"]["num_classes"],
            gamma=2,
            alpha=1,
            lb_smooth=0.2,
        )
        self.criterion_sat = FocalLossWithSmoothing(
            hparams["inputs"]["num_classes"],
            gamma=2,
            alpha=1,
            lb_smooth=0.2,
        )

        self.loss_aux_sat_weight = hparams["hyperparams"]["loss_aux_sat_weight"]
        self.loss_main_sat_weight = hparams["hyperparams"]["loss_main_sat_weight"]

        self.denoise_net = UNet(
            hidden_size,
            out_dim=hparams["inputs"]["num_channels_sat"],
            cond_dim=hparams["rrdb_num_feat"],
            dim_mults=dim_mults,
        )

        self.cond_net = TConvFormer(
            input_size=(
                hparams["inputs"]["lr_patch_size"],
                hparams["inputs"]["lr_patch_size"],
            ),
            stem_channels=64,
            block_channels=hparams["models"]["t_convformer"][
                "block_channels"
            ],  # [64, 128, 256, 512]
            block_layers=hparams["models"]["t_convformer"][
                "block_layers"
            ],  # [2, 2, 5, 2]
            head_dim=32,
            stochastic_depth_prob=0.2,
            partition_size=hparams["models"]["maxvit"]["window_cond_size"],
        )

        # During training, we load the pretrained cond parameters
        # But for inference, we only load the whole classification model

        if not hparams["infer"]:
            # if not hparams["resume"] and not hparams["infer"]:
            if hparams["cond_net_ckpt"] != "" and os.path.exists(
                hparams["cond_net_ckpt"]
            ):
                weights_path = hparams["cond_net_ckpt"]
                if torch.cuda.is_available():
                    old_dict = torch.load(weights_path, weights_only=False)
                else:
                    old_dict = torch.load(
                        weights_path, map_location=torch.device("cpu")
                    )
                model_dict = self.cond_net.state_dict()
                old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
                model_dict.update(old_dict)
                self.cond_net.load_state_dict(model_dict)

        self.gaussian = GaussianDiffusion(
            denoise_net=self.denoise_net,
            cond_net=self.cond_net,
            timesteps=hparams["timesteps"],
            loss_type=hparams["loss_type"],
        )

        self.model = SITSAerialSegmenter(gaussian=self.gaussian, config=hparams)
        if hparams["infer"]:
            if hparams["diff_net_ckpt"] != "" and os.path.exists(
                hparams["diff_net_ckpt"]
            ):
                load_ckpt(self.model, hparams["diff_net_ckpt"])

        # what is used for?
        self.global_step = 0
        return self.model

    def build_optimizer(self, model):
        params = list(model.parameters())
        params = [p for p in params if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=hparams["lr"])
        return optimizer

    def build_scheduler(self, optimizer):
        # 1. Scheduler type
        # It uses torch.optim.lr_scheduler.MultiStepLR, which reduces the learning
        # rate (LR) by a factor (gamma) at specific training steps (called milestones)

        # 2. Milestones
        # This means the LR will drop twice:
        # First at 50% of decay_steps
        # Then at 90% of decay_steps
        # Example: if decay_steps = 100000, milestones = [50000, 90000].

        # 3. Gamma factor
        # At each milestone, the LR is multiplied by 0.1 (reduced by 10×).
        # Example: if LR starts at 0.001,
        # at step 50k → LR becomes 0.0001
        # at step 90k → LR becomes 0.00001.

        # 4. Effect
        # This creates a piecewise-constant decay schedule:
        # LR stays constant in between milestones.
        # At the milestone steps, LR suddenly drops.

        scheduler_param = {
            "milestones": [
                np.floor(hparams["decay_steps"] * 0.5),
                np.floor(hparams["decay_steps"] * 0.9),
            ],
            "gamma": 0.1,
        }
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_param)

    def training_step(self, batch):
        img_hr = batch["img_hr"]  # torch.Size([4, 5, 512, 512])
        img_hr_down = batch["img_hr_down"]  # torch.Size([4, 5, 64, 64])
        img_lr = batch["img_lr"]  # torch.Size([4, 2, 3, 10, 10])
        img_lr_up = batch["img_lr_up"]  # torch.Size([4, 2, 3, 64, 64])
        labels = batch["labels"]  # torch.Size([4, 2, 3, 64, 64])
        labels_sr = batch["labels_sr"]  # torch.Size([4, 2, 3, 64, 64])
        dates = batch["dates_encoding"]
        closest_idx = batch["closest_idx"]  # torch.Size([4, 2, 3, 64, 64])

        # print("img_hr:", img_hr.shape)
        # print("dem_elev:", dem_elev.shape)

        # print("img_lr:", img_lr.shape)
        # print("img_lr_up:", img_lr_up.shape)
        # print("labels:", labels.shape)
        # print("labels_sr:", labels_sr.shape)

        # print("Min:", dates.min().item())
        # print("Max:", dates.max().item())

        # call gaussian diffusion model for SR-prediction this should also
        # return the SR-SITS images alongside the diffusion losses
        losses, _, _, img_sr = self.model.gaussian(
            img_hr,
            img_hr_down,
            img_lr,
            img_lr_up,
            dates,
            closest_idx,
        )

        # print("img_sr train:", img_sr.shape)
        # print("sc_img_hr:", sc_img_hr.shape)

        # for classification branches
        cls_sits, aer_outputs = self.model(img_hr, img_sr, dates)

        # print("labels:", labels.shape)
        # print("aer_outputs:", aer_outputs.shape)

        labels_sr = torch.argmax(labels_sr, dim=1) if labels_sr.ndim == 4 else labels_sr
        labels = torch.argmax(labels, dim=1) if labels.ndim == 4 else labels

        # Auxiliary losses
        # The CE loss for the SITS classification branch is done at 1m GSD

        # print("multi_outputs 0:", multi_outputs[0].shape)
        # print("multi_outputs 1:", multi_outputs[1].shape)
        # print("multi_outputs 2:", multi_outputs[2].shape)

        # aux_loss1 = self.criterion_sat(multi_outputs[2], labels_sr)
        # aux_loss2 = self.criterion_sat(multi_outputs[1], labels_sr)
        # aux_loss3 = self.criterion_sat(multi_outputs[0], labels_sr)

        # loss for main SITS classification branch
        loss_main_sat = self.criterion_sat(cls_sits, labels_sr)

        # # Total loss for SITS branch
        # loss_sat = self.loss_main_sat_weight * loss_main_sat + (
        #     self.loss_aux_sat_weight * aux_loss1
        #     + self.loss_aux_sat_weight * aux_loss2
        #     + self.loss_aux_sat_weight * aux_loss3
        # )

        # print("loss_sat:", loss_sat)
        # print("loss_sr:", losses["sr"])

        # Loss for AER branch
        loss_aer = self.criterion_aer(aer_outputs, labels.long())

        # The CE loss for the SITS classification branch is done at 1.6m GSD
        # that combines the loss from the SR-diffusion model and the SITS
        #  segmentation branch

        losses["sr"] = hparams["hyperparams"]["loss_weights_aer_sat"][1] * (
            losses["sr"] + self.loss_main_sat_weight * loss_main_sat
        )

        # The CE loss for the AER classification branch is done at 20cm GSD
        losses["aer"] = hparams["hyperparams"]["loss_weights_aer_sat"][0] * loss_aer

        total_loss = sum(losses.values())
        return losses, total_loss

    def sample_and_test(self, sample):
        # Sample images and calculate evaluation metrics
        # Used for inference mode
        ret = {k: [] for k in self.metric_keys}
        ret["n_samples"] = 0
        img_hr = sample["img_hr"]
        img_hr_down = sample["img_hr_down"]
        img_lr = sample["img_lr"]
        img_lr_up = sample["img_lr_up"]
        labels = sample["labels"]
        dates = sample["dates_encoding"]
        closest_idx = sample["closest_idx"]  # torch.Size([4, 2, 3, 160, 160])

        img_sr, final_loss = self.model.gaussian.sample(
            img_hr,  # HR image for conditioning
            img_lr,
            img_lr_up,
            img_hr_down,
            dates,
            closest_idx,
        )

        # print("img_sr infer:", img_sr.shape)
        # during sampling, only the aer branch is used
        _, aer_outputs = self.model(img_hr, img_sr, dates)
        proba = torch.softmax(aer_outputs, dim=1)
        preds = torch.argmax(proba, dim=1)
        labels = labels.argmax(dim=1)

        # print("preds:", preds.shape)
        # print("labels:", labels.shape)

        # Loop over batch
        for b in range(img_sr.shape[0]):
            s = self.measure.measure(
                img_sr[b][int(closest_idx[b].item()), :, :, :],  # SR image at t
                img_hr_down[b],  # reference HR image
                img_lr[b][int(closest_idx[b].item()), :, :, :],  # LR input at t
                preds[b],
                labels[b],
            )
            ret["psnr"].append(s["psnr"])
            ret["ssim"].append(s["ssim"])
            ret["lpips"].append(s["lpips"])
            ret["mae"].append(s["mae"])
            ret["mse"].append(s["mse"])
            ret["shift_mae"].append(s["shift_mae"])
            ret["miou"].append(s["miou"])

            ret["n_samples"] += 1
        return img_sr, preds, ret, final_loss
