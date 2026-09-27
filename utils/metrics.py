import os
import torch
import json
from typing import Dict
from pathlib import Path
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import rasterio
from data.utils_data.io import DATA_DIR
from utils.metrics_core import (
    class_IoU,
    overall_accuracy,
    class_precision,
    class_recall,
    class_fscore,
)


def generate_miou(hparams: str, path_truth: str, path_pred: str) -> list:
    #################################################################################################

    def calc_miou(cm_array):
        """
        Specifically, we calculate the per-patch confusion matrix and per-class
        Intersection over Union (IoU) without excluding pixels belonging to
        the 'other' class, even though they represent a marginal part of the test-set.
        However, when computing the mean IoU (mIoU), we do remove the IoU of the 'other'
        class due to its association with majority or lower quality level pixels or very
        underrepresented land cover.
        """
        m = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            ious = np.diag(cm_array) / (
                cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array)
            )
        m = np.nansum(ious[:-1]) / (np.logical_not(np.isnan(ious[:-1]))).sum()
        return m.astype(float), ious[:-1]

    #################################################################################################

    patch_confusion_matrices = []

    for gt_path in tqdm(path_truth, desc=f"Metrics", unit="img"):
        gt_path = Path(gt_path)
        pred_path = Path(path_pred) / f"PRED_{gt_path.name}"
        channel = hparams["labels_configs"].get("label_channel_nomenclature", 1)
        with rasterio.open(os.path.join(DATA_DIR, gt_path), "r") as src_gt:
            target = src_gt.read(channel)
        with rasterio.open(pred_path, "r") as src_pred:
            pred = src_pred.read(1)

        target = torch.from_numpy(target)
        target = target.detach().cpu().numpy()

        patch_confusion_matrices.append(
            confusion_matrix(
                target.flatten(),
                pred.flatten(),
                labels=list(range(hparams["inputs"]["num_classes"])),
            )
        )
    sum_confmat = np.sum(patch_confusion_matrices, axis=0)
    mIou, ious = calc_miou(sum_confmat)
    return mIou, ious


def generate_mf1s(hparams, path_truth: str, path_pred: str) -> list:
    #################################################################################################
    def get_data_paths(path, filter):
        for path in Path(path).rglob(filter):
            yield path.resolve().as_posix()

    def get_confusion_metrics(confusion_matrix):
        """Computes confusion metrics out of a confusion matrix (N classes)
        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            Confusion matrix [N x N]
        Returns
        -------
        metrics : dict
            a dictionary holding all computed metrics
        Notes
        -----
        Metrics are: 'percentages', 'precisions', 'recalls', 'f1s', 'mf1', 'oa'
        """
        tp = np.diag(confusion_matrix)
        tp_fn = np.sum(confusion_matrix, axis=0)
        tp_fp = np.sum(confusion_matrix, axis=1)

        has_no_rp = tp_fn == 0
        has_no_pp = tp_fp == 0

        tp_fn[has_no_rp] = 1
        tp_fp[has_no_pp] = 1

        percentages = tp_fn / np.sum(confusion_matrix)
        precisions = tp / tp_fp
        recalls = tp / tp_fn

        p_zero = precisions == 0
        precisions[p_zero] = 1

        f1s = 2 * (precisions * recalls) / (precisions + recalls)
        ious = tp / (tp_fn + tp_fp - tp)

        precisions[has_no_pp] *= 0.0
        precisions[p_zero] *= 0.0
        recalls[has_no_rp] *= 0.0

        f1s[p_zero] *= 0.0
        f1s[percentages == 0.0] = np.nan
        ious[percentages == 0.0] = np.nan

        mf1 = np.nanmean(f1s[:-1])
        miou = np.nanmean(ious[:-1])

        oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)

        metrics = {
            "percentages": percentages,
            "precisions": precisions,
            "recalls": recalls,
            "f1s": f1s,
            "mf1": mf1,
            "ious": ious,
            "miou": miou,
            "oa": oa,
        }
        return metrics

    patch_confusion_matrices = []

    for gt_path in tqdm(path_truth, desc=f"Metrics", unit="img"):
        gt_path = Path(gt_path)
        pred_path = Path(path_pred) / f"PRED_{gt_path.name}"
        channel = hparams["labels_configs"].get("label_channel_nomenclature", 1)
        with rasterio.open(os.path.join(DATA_DIR, gt_path), "r") as src_gt:
            target = src_gt.read(channel)
        with rasterio.open(pred_path, "r") as src_pred:
            pred = src_pred.read(1)

        target = torch.from_numpy(target)
        target = target.detach().cpu().numpy()

        patch_confusion_matrices.append(
            confusion_matrix(
                target.flatten(),
                pred.flatten(),
                labels=list(range(hparams["inputs"]["num_classes"])),
            )
        )
    sum_confmat = np.sum(patch_confusion_matrices, axis=0)
    metrics = get_confusion_metrics(sum_confmat)
    return metrics["mf1"], metrics["f1s"], metrics["oa"]


def compute_and_save_metrics(
    confmat: np.ndarray, config: Dict, output_dir: str, task: str, mode: str = "predict"
) -> None:
    """
    Computes segmentation evaluation metrics from a confusion matrix and saves them to disk.
    Metrics computed:
        - Per-class IoU, precision, recall, and F1-score
        - Mean IoU (mIoU)
        - Overall accuracy
        - Weighted class importance (per task and modality)
    Also logs the results in human-readable format and stores:
        - metrics.json with all results
        - confmat_<mode>.npy with raw confusion matrix
    Args:
        confmat (np.ndarray): Confusion matrix of shape (num_classes, num_classes).
        config (Dict): Configuration dictionary containing task and class metadata.
        output_dir (str): Directory where results will be saved.
        task (str): Name of the current task (used to retrieve class info).
        mode (str): Operational mode label, e.g., "predict" or "val". Used in file naming.
    Returns:
        None
    """
    label_config = config["labels_configs"][task]
    class_names = label_config["value_name"]
    num_classes = len(class_names)

    value_weights = label_config.get("value_weights", {})
    default_weight = value_weights.get("default", 1)
    default_exceptions = value_weights.get("default_exceptions", {}) or {}
    default_weights = [default_weight] * num_classes
    for i, weight in default_exceptions.items():
        default_weights[i] = weight

    active_modalities = [
        mod for mod, is_active in config["modalities"]["inputs"].items() if is_active
    ]
    per_modality_exceptions = value_weights.get("per_modality_exceptions", {}) or {}

    modality_weights = {}
    for mod in active_modalities:
        modality_weights[mod] = default_weights.copy()
        mod_exceptions = per_modality_exceptions.get(mod)
        if mod_exceptions:
            for i, weight in mod_exceptions.items():
                modality_weights[mod][i] = weight

    weights_array = np.array(default_weights)
    used_indices = np.where(weights_array != 0)[0]

    confmat_cleaned = confmat[np.ix_(used_indices, used_indices)]
    class_names_cleaned = [class_names[i] for i in used_indices]
    default_weights_cleaned = [default_weights[i] for i in used_indices]
    modality_weights_cleaned = {
        mod: [modality_weights[mod][i] for i in used_indices]
        for mod in active_modalities
    }

    per_c_ious, avg_ious = class_IoU(confmat_cleaned, len(used_indices))
    ovr_acc = overall_accuracy(confmat_cleaned)
    per_c_precision, avg_precision = class_precision(confmat_cleaned)
    per_c_recall, avg_recall = class_recall(confmat_cleaned)
    per_c_fscore, avg_fscore = class_fscore(per_c_precision, per_c_recall)

    metrics = {
        "Avg_metrics_name": [
            "mIoU",
            "Overall Accuracy",
            "F-score",
            "Precision",
            "Recall",
        ],
        "Avg_metrics": [avg_ious, ovr_acc, avg_fscore, avg_precision, avg_recall],
        "classes": class_names_cleaned,
        "per_class_iou": list(per_c_ious),
        "per_class_fscore": list(per_c_fscore),
        "per_class_precision": list(per_c_precision),
        "per_class_recall": list(per_c_recall),
        "per_class_default_weight": default_weights_cleaned,
        "per_class_modality_weights": modality_weights_cleaned,
    }

    out_folder_metrics = Path(
        output_dir, f"metrics_{config['paths']['out_model_name']}", task
    )
    out_folder_metrics.mkdir(exist_ok=True, parents=True)
    np.save(out_folder_metrics / f"confmat_{mode}.npy", confmat)
    with open(out_folder_metrics / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTask: {task} - Global Metrics:")
    print("-" * (90 + 15 * len(active_modalities)))
    for name, value in zip(metrics["Avg_metrics_name"], metrics["Avg_metrics"]):
        print(f"{name:<20s} {value:<.4f}")
    print("-" * (90 + 15 * len(active_modalities)) + "\n")

    header = "{:<6} {:<25} {:<10} {:<10} {:<10} {:<10} {:<15}".format(
        "Idx", "Class", "IoU", "F-score", "Precision", "Recall", "w.TASK"
    )
    for mod in active_modalities:
        header += f" {'w.' + mod:<15}"
    print(header)
    print("-" * (90 + 15 * len(active_modalities)))

    for i, class_name in enumerate(class_names_cleaned):
        row = "{:<6} {:<25} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<15}".format(
            i,
            class_name,
            per_c_ious[i],
            per_c_fscore[i],
            per_c_precision[i],
            per_c_recall[i],
            default_weights_cleaned[i],
        )
        for mod in active_modalities:
            row += f" {modality_weights_cleaned[mod][i]:<15}"
        print(row)
    print("\n")

    unused_indices = np.where(weights_array == 0)[0]
    if len(unused_indices) > 0:
        print("0-weighted classes for task")
        print("-" * 35)
        for idx in unused_indices:
            class_label = class_names[idx]
            print(f"{idx:<6} {class_label}")
        print("\n")


def generate_metrics(
    hparams: str, path_truth: str, path_pred: str, output_dir: str, task: str
) -> list:
    #################################################################################################

    patch_confusion_matrices = []

    for gt_path in tqdm(path_truth, desc=f"Metrics", unit="img"):
        gt_path = Path(gt_path)
        pred_path = Path(path_pred) / f"PRED_{gt_path.name}"
        channel = hparams["labels_configs"].get("label_channel_nomenclature", 1)
        with rasterio.open(os.path.join(DATA_DIR, gt_path), "r") as src_gt:
            target = src_gt.read(channel)
        with rasterio.open(pred_path, "r") as src_pred:
            pred = src_pred.read(1)

        target = torch.from_numpy(target)
        target = target.detach().cpu().numpy()

        patch_confusion_matrices.append(
            confusion_matrix(
                target.flatten(),
                pred.flatten(),
                labels=list(range(hparams["inputs"]["num_classes"])),
            )
        )
    # confusion_matrix_path = Path(hparams["paths"]["out_folder"])
    sum_confmat = np.sum(patch_confusion_matrices, axis=0)
    compute_and_save_metrics(sum_confmat, hparams, output_dir, task)
    return
