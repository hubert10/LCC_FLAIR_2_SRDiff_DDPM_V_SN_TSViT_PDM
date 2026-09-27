import datetime
import geopandas as gpd
import numpy as np
import json
from typing import Dict, Tuple, Any, Set


# Agricultural and land cover classification often depends on seasonal changes.
#  In many parts of the world, May 15th could be a crucial date because it might
#  represent the peak of a particular growing season, a planting season, or a
# critical stage in crop development.
# Remote sensing data (such as from SITS) often captures land cover changes
# throughout the year, and May 15th might represent an average or key point in the seasonal cycle for the crops or vegetation types being studied.


def prepare_sentinel_dates(
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

    ref_month, ref_day = map(int, config["inputs"]["ref_date"].split("-"))

    dict_dates = {}
    for _, row in gdf.iterrows():
        patch_id = row["patch_id"]
        acquisition_dates = json.loads(row["acquisition_dates"])

        dates_array = []
        diff_dates_array = []
        for date_str in acquisition_dates.values():
            try:
                original_date = datetime.datetime.strptime(date_str, "%Y%m%d")
                reference_date = datetime.datetime(
                    original_date.year, ref_month, ref_day
                )
                diff_days = (original_date - reference_date).days
                dates_array.append(original_date)
                diff_dates_array.append(diff_days)
            except ValueError as e:
                print(f"Invalid date encountered: {date_str}. Error: {e}")

        dict_dates[patch_id] = {
            "dates": np.array(dates_array),
            "diff_dates": np.array(diff_dates_array),
        }
    return dict_dates


def get_sentinel_dates_mtd(
    config: dict, patch_ids: set
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Retrieve sentinel dates metadata based on the provided configuration.
    Args:
        config (dict): Configuration dictionary.
    Returns:
        tuple: Dictionaries with area_id as keys and acquisition_dates as values for Sentinel2
    """
    assert isinstance(config, dict), "config must be a dictionary"

    dates_s2 = {}

    sen2_used = config["modalities"]["inputs"].get("SENTINEL2_TS", False)

    if not (sen2_used):
        return dates_s2

    if sen2_used:
        dates_s2 = prepare_sentinel_dates(
            config,
            config["paths"]["global_mtd_folder"] + "GLOBAL_SENTINEL2_MTD_DATES.gpkg",
            patch_ids,
        )
    return dates_s2
