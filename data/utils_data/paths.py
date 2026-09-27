import os
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Set
from data.utils_data.sentinel_dates import get_sentinel_dates_mtd
from data.utils_data.spot_dates import get_spot_dates_mtd


def extract_sentinel_patch_ids(dicts: List[Dict]) -> Set[str]:
    """
    Extracts unique patch IDs from Sentinel file paths in a list of dictionaries.
    Looks for keys: 'SENTINEL2_TS'and parses patch IDs from filenames.
    Args:
        dicts (List[Dict]): List of dictionaries with optional Sentinel file paths.
    Returns:
        Set[str]: Set of unique patch IDs.
    """
    patch_ids = set()
    for d in dicts:
        if d is None:
            continue
        for key in ["SENTINEL2_TS"]:
            for path in d.get(key, []):
                fname = path.split("/")[-1]
                patch_id = fname.replace(f"_{key}", "").replace(".tif", "")
                patch_ids.add(patch_id)
    return patch_ids


def extract_spot_patch_ids(dicts: List[Dict]) -> Set[str]:
    """
    Extracts unique patch IDs from Sentinel file paths in a list of dictionaries.
    Looks for keys: 'SPOT_RGBI' and parses patch IDs from filenames.
    Args:
        dicts (List[Dict]): List of dictionaries with optional Sentinel file paths.
    Returns:
        Set[str]: Set of unique patch IDs.
    """
    patch_ids = set()
    for d in dicts:
        if d is None:
            continue
        for key in ["SPOT_RGBI"]:
            for path in d.get(key, []):
                fname = path.split("/")[-1]
                patch_id = fname.replace(f"_{key}", "").replace(".tif", "")
                patch_ids.add(patch_id)
    return patch_ids


def get_paths(config: Dict[str, Any], split: str = "train") -> Dict:
    """
    Retrieves paths to data files based on the provided configuration and split type.
    Args:
        config (dict): A configuration dictionary that includes paths to CSV files,
        and modality activation.
        split (str): The data split type, which can be 'train', 'val', or 'test'.
    Returns:
        dict: A dictionary containing paths for each modality.
    Raises:
        SystemExit: If an invalid split is specified or the CSV file path is invalid.
    """

    if split == "train":
        csv_path = config["paths"]["train_csv"]
    elif split == "val":
        csv_path = config["paths"]["val_csv"]
    elif split == "test":
        csv_path = config["paths"]["test_csv"]
    else:
        print("Invalid split specified.")
        raise SystemExit()

    if csv_path is not None and os.path.isfile(csv_path) and csv_path.endswith(".csv"):
        paths = pd.read_csv(csv_path, sep=";")
    else:
        print(f"Invalid .csv file path for {split} split.")
        raise SystemExit()

    dict_paths = {modality: [] for modality in config["modalities"]["inputs"].keys()}

    for modality, is_active in config["modalities"]["inputs"].items():
        if is_active == True and modality in paths.columns:
            dict_paths[modality] = paths[modality].tolist()

    for label_mod in config["labels"]:
        dict_paths[label_mod] = paths[label_mod].tolist()

    if config["modalities"]["inputs"]["SENTINEL2_TS"]:
        dict_paths["SENTINEL2_MSK-SC"] = paths["SENTINEL2_MSK-SC"].tolist()
    else:
        dict_paths["SENTINEL2_MSK-SC"] = []

    return dict_paths


def get_datasets(
    config: Dict[str, Any]
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Get the datasets for training, validation, and testing.
    Args:
        config (Dict[str, Any]): Configuration dictionary.
    Returns:
        Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]: Datasets for training, validation, and testing.
    """
    dict_train, dict_val, dict_test = None, None, None

    if config["tasks"]["train"]:
        dict_train = get_paths(config, split="train")
        dict_val = get_paths(config, split="val")

    if config["tasks"]["predict"]:
        dict_test = get_paths(config, split="test")

    all_dicts = [dict_train, dict_val, dict_test]

    used_patch_ids = extract_sentinel_patch_ids(all_dicts)
    spot_used_patch_ids = extract_spot_patch_ids(all_dicts)

    dates_s2 = get_sentinel_dates_mtd(config, used_patch_ids)
    dates_s6 = get_spot_dates_mtd(config, spot_used_patch_ids)

    for d in [dict_train, dict_val, dict_test]:
        if d is not None:
            d["DATES_S2"] = dates_s2
            d["DATES_S6"] = dates_s6

    return dict_train, dict_val, dict_test
