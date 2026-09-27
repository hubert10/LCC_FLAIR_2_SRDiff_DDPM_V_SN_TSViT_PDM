import datetime
import geopandas as gpd
import numpy as np
import json
from typing import Dict, Tuple, Any, Set


def prepare_spot_dates(
    config: Dict[str, Any], file_path: str, patch_ids: Set[str]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Processes acquisition dates for selected Sentinel patches and computes their temporal offsets
    from a reference date defined in the config.
    Args:
        config (Dict[str, Any]): Configuration dictionary containing the reference date under
            config['models']['multitemp_model']['ref_date'] in 'MM-DD' format.
        file_path (str): Path to the GeoJSON or GeoPackage file with acquisition metadata.
        patch_ids (Set[str]): Set of patch IDs to include.
    Returns:
        Dict[str, Dict[str, np.ndarray]]: A dictionary mapping each patch ID to:
            - 'dates': numpy array of acquisition datetime objects.
            - 'diff_dates': numpy array of day offsets from the reference date.
    """
    gdf = gpd.read_file(file_path, engine="pyogrio", use_arrow=True)
    gdf = gdf[gdf["patch_id"].isin(patch_ids)]

    dict_dates = {}
    for _, row in gdf.iterrows():
        patch_id = row["patch_id"]
        date_str = json.loads(row["date"])
        dates_array = []
        try:
            original_date = datetime.datetime.strptime(str(date_str), "%Y%m%d")
            dates_array.append(original_date)
        except ValueError as e:
            print(f"Invalid date encountered: {date_str}. Error: {e}")

        dict_dates[patch_id] = {"spot_dates": np.array(dates_array)}
    return dict_dates


def get_spot_dates_mtd(
    config: dict, patch_ids: set
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Retrieve sentinel dates metadata based on the provided configuration.
    Args:
        config (dict): Configuration dictionary.
    Returns:
        tuple: Dictionaries with area_id as keys and acquisition_dates as values for Spot6
    """
    assert isinstance(config, dict), "config must be a dictionary"

    dates_s6 = {}

    s6_used = config["modalities"]["inputs"].get("SPOT_RGBI", False)

    if not (s6_used):
        return dates_s6

    if s6_used:
        dates_s6 = prepare_spot_dates(
            config,
            config["paths"]["global_mtd_folder"] + "GLOBAL_SPOT_MTD_DATES.gpkg",
            patch_ids,
        )
    return dates_s6
