import os
import random
import torch
import importlib
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from data.datamodule import FlairDataModule
from utils.metrics import generate_miou, generate_mf1s, generate_metrics
from utils.utils_dataset import (
    save_image_to_nested_folder,
    save_hr_image_to_nested_folder,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"
# sys.path.append('../')

from utils.utils import (
    move_to_cuda,
    load_checkpoint,
    save_checkpoint,
    tensors_to_scalars,
    Measure,
)
from utils.hparams import hparams, set_hparams
from torchvision import transforms as cT

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

warnings.filterwarnings("ignore", ".*does not have many workers.*")
from data.utils_data.paths import get_datasets
from tasks.module_setup import build_data_module
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class Trainer:
    def __init__(self, hparams):
        self.logger = self.build_tensorboard(
            save_dir=hparams["work_dir"], name="tb_logs"
        )
        d_train, d_val, d_test = get_datasets(hparams)

        self.d_train = d_train
        self.d_val = d_val
        self.d_test = d_test

        self.measure = Measure()
        self.dataset_cls = None
        self.metric_keys = [
            "psnr",
            "ssim",
            "lpips",
            "mae",
            "mse",
            "shift_mae",
            "miou",
        ]
        self.work_dir = hparams["work_dir"]
        self.first_val = True
        self.config = hparams
        self.device = device
        self.datamodule = FlairDataModule(hparams)

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def build_tensorboard(self, save_dir, name, **kwargs):
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir, **kwargs)

    def build_train_dataloader(self, subset=True):
        dataset_train = self.datamodule._create_FLAIRDataSet(
            dict_paths=self.d_train,
            use_augmentations=hparams["modalities"]["pre_processings"][
                "use_augmentation"
            ],
        )

        dataloader = self.datamodule._create_dataloader(
            dataset_train,
            batch_size=hparams["batch_size"],
            shuffle=True,
            # num_workers=hparams["num_workers"],
            drop_last=True,
            # collate_fn=pad_collate_train,
        )
        return dataloader

    def build_val_dataloader(self, subset=True):
        dataset_val = self.datamodule._create_FLAIRDataSet(
            dict_paths=self.d_val,
            use_augmentations=hparams["modalities"]["pre_processings"][
                "use_augmentation"
            ],
        )
        dataloader = self.datamodule._create_dataloader(
            dataset_val,
            batch_size=hparams["eval_batch_size"],
            shuffle=False,
            # num_workers=hparams["num_workers"],
            drop_last=True,
            # collate_fn=pad_collate_train,
        )
        return dataloader

    def build_test_dataloader(self, subset=False):
        dataset_test = self.datamodule._create_FLAIRDataSet(
            dict_paths=self.d_test,
            use_augmentations=hparams["modalities"]["pre_processings"][
                "use_augmentation"
            ],
        )

        dataloader = self.datamodule._create_dataloader(
            dataset_test,
            batch_size=hparams["test_batch_size"],
            shuffle=False,
            # num_workers=hparams["num_workers"],
            drop_last=False,
            # collate_fn=pad_collate_predict,
        )
        return dataloader

    def build_model(self):
        raise NotImplementedError

    def sample_and_test(self, sample):
        raise NotImplementedError

    def build_optimizer(self, model):
        raise NotImplementedError

    def build_scheduler(self, optimizer):
        raise NotImplementedError

    def training_step(self, batch):
        raise NotImplementedError

    @rank_zero_only
    def print_model_parameters(self, model: nn.Module) -> None:
        """
        Print the total and trainable number of parameters in the model,
        broken down by component.
        Args:
            model (nn.Module): Full model containing latent_diff, encoders, decoder, and fusion modules.
        """

        def count_params(module: nn.Module | None):
            if module is None:
                return 0, 0
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        # Parameter counts
        cond_total, cond_train = count_params(model.gaussian.cond_net)
        denoise_total, denoise_train = count_params(model.gaussian.denoise_net)
        sits_enc_total, sits_enc_train = count_params(model.sr_sits_enc)
        sits_dec_total, sits_dec_train = count_params(model.sr_sits_dec)
        aer_enc_total, aer_enc_train = count_params(model.aer_net_enc)
        aer_dec_total, aer_dec_train = count_params(model.aer_net_dec)
        fusion_total, fusion_train = count_params(model.fusion_module)

        total_params = (
            cond_total
            + denoise_total
            + sits_enc_total
            + sits_dec_total
            + aer_enc_total
            + aer_dec_total
            + fusion_total
        )

        total_trainable = (
            cond_train
            + denoise_train
            + sits_enc_train
            + sits_dec_train
            + aer_enc_train
            + aer_dec_train
            + fusion_train
        )

        # Build table
        table = " " + "-" * 126 + "\n"
        table += (
            "| {:<25} | {:<25} | {:<17} | {:>15} | {:>15} |\n"
            "| {} | {} | {} | {} | {} |\n"
        ).format(
            "Model component",
            "Sub-module",
            "Type",
            "Total params",
            "Trainable",
            "-" * 25,
            "-" * 25,
            "-" * 17,
            "-" * 15,
            "-" * 15,
        )

        rows = [
            ("Conditioning", "cond_net", "conditioning net", cond_total, cond_train),
            ("Denoising", "denoise_net", "denoising net", denoise_total, denoise_train),
            (
                "SITS Encoder",
                "sr_sits_enc",
                "task sits encoder",
                sits_enc_total,
                sits_enc_train,
            ),
            (
                "SITS Decoder",
                "sr_sits_dec",
                "task sits Decoder",
                sits_dec_total,
                sits_dec_train,
            ),
            (
                "Aer Encoder",
                "aer_net_enc",
                "task aer encoder",
                aer_enc_total,
                aer_enc_train,
            ),
            (
                "Aer Decoder",
                "aer_net_dec",
                "task decoder",
                aer_dec_total,
                aer_dec_train,
            ),
            ("Fusion", "fusion_module", "fusion module", fusion_total, fusion_train),
        ]

        for component, submodule, module_type, total, trainable in rows:
            table += "| {:<25} | {:<25} | {:<17} | {:>15,} | {:>15,} |\n".format(
                component, submodule, module_type, total, trainable
            )

        # Footer
        table += "|" + "-" * 126 + "|\n"
        table += "| {:<25}   {:<25}   {:<17}   {:>15,}   {:>15,} |\n".format(
            "Total parameters", "", "", total_params, total_trainable
        )
        table += " " + "-" * 126

        print("")
        print(table)
        print("")

    def train(self):
        model = self.build_model()
        self.print_model_parameters(model)

        optimizer = self.build_optimizer(model)
        self.global_step = training_step = load_checkpoint(
            model, optimizer, hparams["work_dir"]
        )
        self.scheduler = scheduler = self.build_scheduler(optimizer)
        dataloader = self.build_train_dataloader()

        train_pbar = tqdm(dataloader)

        list_loss = []
        val_list = []
        val_steps = []
        val_loss_list = []
        epoch = 0

        # Load existing logs if they exist
        train_loss_path = hparams["work_dir"] + "/train_loss.csv"
        val_res_path = hparams["work_dir"] + "/val_res.csv"

        try:
            previous_train_loss = pd.read_csv(train_loss_path, sep=";")
            list_loss = previous_train_loss["l"].tolist()
        except Exception:
            previous_train_loss = pd.DataFrame()

        try:
            previous_val_res = pd.read_csv(val_res_path, sep=";")
            val_list = previous_val_res["val_metrics"].tolist()
            val_steps = previous_val_res["train_step"].tolist()
            if not hparams["train_diffsr"] and "val_loss" in previous_val_res.columns:
                val_loss_list = previous_val_res["val_loss"].tolist()
        except Exception:
            previous_val_res = pd.DataFrame()

        while self.global_step < hparams["max_updates"] + 1:
            c = 0
            loss_ = 0

            for idx, batch in enumerate(train_pbar):
                if (training_step % hparams["val_check_interval"] == 0) and (
                    training_step != 0
                ):
                    if training_step not in val_steps:
                        with torch.no_grad():
                            model.eval()
                            val_res, val_step, val_loss = self.validate(training_step)
                            val_list.append(val_res)
                            val_steps.append(val_step)
                            if not hparams["train_diffsr"]:
                                val_loss_list.append(val_loss)

                # if training_step % hparams["save_ckpt_interval"] == 0:
                #     save_checkpoint(
                #         model,
                #         optimizer,
                #         self.work_dir,
                #         training_step,
                #         hparams["num_ckpt_keep"],
                #     )

                if training_step % hparams["save_ckpt_interval"] == 0:
                    val_miou = None
                    if (
                        val_list
                        and isinstance(val_list[-1], dict)
                        and "val/miou" in val_list[-1]
                    ):
                        val_miou = round(val_list[-1]["val/miou"], 2)

                    save_checkpoint(
                        model,
                        optimizer,
                        self.work_dir,
                        training_step,
                        hparams["num_ckpt_keep"],
                        val_miou=val_miou,
                    )

                model.train()
                batch = move_to_cuda(batch)
                losses, total_loss = self.training_step(batch)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                training_step += 1

                loss_ += total_loss.detach().item()
                c += 1
                scheduler.step()
                self.global_step = training_step
                if training_step % 1000 == 0:
                    self.log_metrics(
                        {f"tr/{k}": v for k, v in losses.items()}, training_step
                    )
                train_pbar.set_postfix(**tensors_to_scalars(losses))

                list_loss.append(loss_ / c if c != 0 else 0.0)
                epoch += 1

            if epoch % 1 == 0:
                save_loss = pd.DataFrame(
                    {
                        "training_step": [i for i in range(len(list_loss))],
                        "l": list_loss,
                    }
                )
                save_loss.to_csv(train_loss_path, sep=";", index=False)

                save_val = pd.DataFrame(
                    {"train_step": val_steps, "val_metrics": val_list}
                )
                if not hparams["train_diffsr"]:
                    save_val["val_loss"] = val_loss_list
                print("val_loss_list:", val_loss_list)
                save_val.to_csv(val_res_path, sep=";", index=False)

    def validate(self, training_step):
        val_dataloader = self.build_val_dataloader()
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        val_loss = 0
        c = 0
        val_metrics = {k: [] for k in self.metric_keys}

        for _, batch in pbar:
            # for batch_idx, batch in pbar:
            batch = move_to_cuda(batch)
            _, _, ret, loss = self.sample_and_test(batch)

            if not hparams["train_diffsr"]:
                val_loss += loss.detach().item()
                c += 1
            metrics = {}
            metrics.update({k: np.mean(ret[k]) for k in self.metric_keys})
            for k in self.metric_keys:
                val_metrics[k].append(np.mean(ret[k]))
            pbar.set_postfix(**tensors_to_scalars(metrics))
        if not hparams["train_diffsr"]:
            val_loss = val_loss / c
        if hparams["infer"]:
            print("Val results:", metrics)
        else:
            if not self.first_val:
                self.log_metrics(
                    {f"val/{k}": v for k, v in metrics.items()}, training_step
                )
                print("Val results:", metrics)
                return (
                    {f"val/{k}": np.mean(v) for k, v in val_metrics.items()},
                    training_step,
                    val_loss,
                )
            else:
                print("Sanity val results:", metrics)
                return (
                    {f"val/{k}": np.mean(v) for k, v in val_metrics.items()},
                    training_step,
                    val_loss,
                )
        self.first_val = False

    # Run Inference
    def test(self):
        model = self.build_model()
        # print(model)
        optimizer = self.build_optimizer(model)
        load_checkpoint(model, optimizer, hparams["work_dir"])
        optimizer = None
        self.results = {k: [] for k in self.metric_keys}
        self.results["key"] = []
        self.n_samples = 0
        self.gen_dir = f"{hparams['work_dir']}/results_{self.global_step}_{hparams['gen_dir_name']}"

        if hparams["test_save_png"]:
            os.makedirs(f"{self.gen_dir}/SR", exist_ok=True)
            os.makedirs(f"{self.gen_dir}/PR", exist_ok=True)

        self.model.sample_tqdm = False
        torch.backends.cudnn.benchmark = False
        if hparams["test_save_png"]:
            if hparams["test_diff"]:
                if hasattr(self.model.gaussian.denoise_net, "make_generation_fast_"):
                    self.model.gaussian.denoise_net.make_generation_fast_()
            # os.makedirs(f"{self.gen_dir}/RRDB", exist_ok=True)
            os.makedirs(f"{self.gen_dir}/HR", exist_ok=True)
            os.makedirs(f"{self.gen_dir}/LR", exist_ok=True)
            os.makedirs(f"{self.gen_dir}/UP", exist_ok=True)

        with torch.no_grad():
            model.eval()
            test_dataloader = self.build_test_dataloader()
            pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

            for _, batch in pbar:
                move_to_cuda(batch)
                gen_dir = self.gen_dir
                item_name = batch["item_name"]
                img_hr = batch["img_hr_down"]
                # img_hr = img_hr.permute(0, 2, 3, 1)
                img = batch["img_hr"]
                img_lr = batch["img_lr"]
                img_lr_up = batch["img_lr_up"]
                dates = batch["dates"]
                img_sr, preds, ret, loss = self.sample_and_test(batch)

                if img_sr is not None:
                    metrics = list(self.metric_keys)
                    for k in metrics:
                        self.results[k] += ret[k]
                    self.n_samples += ret["n_samples"]
                    print(
                        {k: np.mean(self.results[k]) for k in metrics},
                        "total:",
                        self.n_samples,
                    )

                    if hparams["test_save_png"] and img_sr is not None:
                        img_hr = self.tensor2img(img_hr)

                        # For single image batch size, we can use the following code
                        if hparams["test_batch_size"] == 1:
                            img_lr = [
                                self.tensor2img(im[None, ...])
                                for im in img_lr.squeeze()
                            ]

                            img_lr_up = [
                                self.tensor2img(im[None, ...])
                                for im in img_lr_up.squeeze()
                            ]

                            img_sr = [
                                self.tensor2img(im[None, ...])
                                for im in img_sr.squeeze()
                            ]

                        for _, hr_g, lr, lr_up, pred, _ in zip(
                            img_sr, img_hr, img_lr, img_lr_up, preds, dates
                        ):
                            # Save high-resolution ground truth image

                            hr_g = Image.fromarray(hr_g[:, :, :3])
                            save_hr_image_to_nested_folder(
                                hr_g, item_name[0], "HR", "img", None, base_dir=gen_dir
                            )

                            # Save pixel-wise predictions
                            pred = pred.cpu().numpy().astype("uint8")

                            output_file = Path(
                                gen_dir,
                                "PR",
                                f"PRED_{item_name[0].split('/')[-1]}",
                            )

                            Image.fromarray(pred).save(
                                f"{output_file}", compression="tiff_lzw"
                            )

                            if hparams["test_batch_size"] == 1:
                                dates = [date for date in dates]

                                lr = [Image.fromarray(im[0]) for im in img_lr]
                                for e, (im, date) in enumerate(zip(lr, dates[0])):
                                    save_image_to_nested_folder(
                                        im,
                                        item_name[0],
                                        "LR",
                                        "sen",
                                        f"{e}_{date}",
                                        base_dir=gen_dir,
                                    )

                                lr_up = [Image.fromarray(im[0]) for im in img_lr_up]
                                for e, (im, date) in enumerate(zip(lr_up, dates[0])):
                                    save_image_to_nested_folder(
                                        im,
                                        item_name[0],
                                        "UP",
                                        "sen",
                                        f"{e}_{date}",
                                        base_dir=gen_dir,
                                    )

                                sr = [Image.fromarray(im[0]) for im in img_sr]
                                for e, (im, date) in enumerate(zip(sr, dates[0])):
                                    save_image_to_nested_folder(
                                        im,
                                        item_name[0],
                                        "SR",
                                        "sen",
                                        f"{e}_{date}",
                                        base_dir=gen_dir,
                                    )

            self.results = {
                k: self.results[k]
                for k in ["psnr", "ssim", "lpips", "mae", "mse", "shift_mae", "miou"]
            }
            res = pd.DataFrame(self.results)
            res.to_csv(hparams["work_dir"] + "/test_results.csv", sep=";")

    def generate_metrics(self):
        # Compute mIoU over the predictions - not done here as the test
        #  labels are not available, but if needed, you can use the
        #  generate_miou function from metrics.py

        csv_path = Path(hparams["paths"]["test_csv"])
        df = pd.read_csv(csv_path, sep=";")
        gt_paths = df[hparams["labels"][0]].tolist()
        pred_msk = os.path.join(self.gen_dir, "PR")
        generate_metrics(
            hparams, gt_paths, pred_msk, self.gen_dir, hparams["labels"][0]
        )

    # utils
    def log_metrics(self, metrics, step):
        metrics = self.metrics_to_scalars(metrics)
        logger = self.logger
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    def tensor2img(self, img):
        img = np.round((img.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5)
        img = img.clip(min=0, max=255).astype(np.uint8)
        return img


if __name__ == "__main__":
    set_hparams()
    pkg = ".".join(hparams["trainer_cls"].split(".")[:-1])
    cls_name = hparams["trainer_cls"].split(".")[-1]
    trainer = getattr(importlib.import_module(pkg), cls_name)(hparams)
    if not hparams["infer"]:
        trainer.train()
    else:
        trainer.test()
        trainer.generate_metrics()


# Contributions:

# python trainer.py --config_file=./configs/train_cond/ --exp_name highresnet_ltae_ckpt --hparams="cond_net_ckpt=./results/checkpoints/misr/highresnet_ltae_ckpt" --reset
# python trainer.py --config_file=./configs/train/ --exp_name srdiff_maxvit_ltae_ckpt --reset
# python trainer.py --config_file=./configs/train_main/ --exp_name srdiff_highresnet_ltae_ckpt --hparams="cond_net_ckpt=./results/checkpoints/misr/srdiff_highresnet_ltae_ckpt" --infer


# scp -r D:\kanyamahanga\Datasets\FLAIR_HUB\data nhgnkany@transfer.cluster.uni-hannover.de:/bigwork/nhgnkany/FLAIR_HUB
# scp -r "D:/kanyamahanga/Datasets/FLAIR_HUB/data/*" nhgnkany@transfer.cluster.uni-hannover.de:/bigwork/nhgnkany/FLAIR_HUB/


# sudo sshfs eouser@74.63.7.39:/hubert_storage/  -o _netdev,user,idmap=user,allow_other,default_permissions,uid=1000,gid=1000,IdentityFile=/home/eouser/.ssh/flair,umask=000
# sudo sshfs -o IdentityFile=/home/eouser/.ssh/flair.pem eouser@74.63.7.39:/hubert_storage /hubert_storage
