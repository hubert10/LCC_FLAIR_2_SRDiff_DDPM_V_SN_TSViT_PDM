from omegaconf import DictConfig, ListConfig
from rich import get_console
from rich.style import Style
from rich.tree import Tree
from datetime import timedelta
from typing import Optional
from pytorch_lightning.utilities.rank_zero import rank_zero_only


@rank_zero_only
def print_recap(
    config: dict,
    dict_train: Optional[dict] = None,
    dict_val: Optional[dict] = None,
    dict_test: Optional[dict] = None,
) -> None:
    """
    Prints content of the given config using a tree structure.
    Args:
        config (dict): The configuration dictionary.
        dict_train (Optional[dict]): Training data dictionary.
        dict_val (Optional[dict]): Validation data dictionary.
        dict_test (Optional[dict]): Test data dictionary.
    Filters some sections if verbose_config is False.
    """

    def walk_config(
        d: dict,
        prefix: str = "",
        filter_section: bool = False,
        active_inputs: Optional[set] = None,
        parent_key: Optional[str] = None,
    ) -> None:
        """
        Recursive config printer with optional filtering logic for inactive inputs.
        """
        for k, v in d.items():

            if active_inputs is not None:
                if parent_key in {"inputs_channels", "aux_loss", "modality_dropout"}:
                    if k not in active_inputs:
                        continue
                elif parent_key == "normalization":
                    if k.endswith("_means") or k.endswith("_stds"):
                        base = k.replace("_means", "").replace("_stds", "")
                        if base not in active_inputs:
                            continue

            if isinstance(v, dict):
                if filter_section and all(
                    x in [False, 0, None, "", [], {}] for x in v.values()
                ):
                    continue
                print(f"{prefix}|- {k}:")
                walk_config(
                    v, prefix + "|   ", filter_section, active_inputs, parent_key=k
                )
            elif isinstance(v, list):
                if not filter_section or v:
                    print(f"{prefix}|- {k}: {v}")
            else:
                if not filter_section or v not in [False, 0, None, "", [], {}]:
                    print(f"{prefix}|- {k}: {v}")

    verbose = config.get("saving", {}).get("verbose_config", True)
    inputs = config.get("modalities", {}).get("inputs", {})
    active_inputs = {k for k, v in inputs.items() if v}

    print("Configuration Tree:")
    for section_key, section_value in config.items():
        if isinstance(section_value, dict):
            print(f"|- {section_key}:")
            if section_key == "modalities":
                walk_config(
                    section_value,
                    prefix="|   ",
                    filter_section=not verbose,
                    active_inputs=active_inputs,
                )
            else:
                walk_config(section_value, prefix="|   ", filter_section=not verbose)
        else:
            print(f"|- {section_key}: {section_value}")

    list_keys = [
        "AERIAL_RGBI",
        "AERIAL-RLT_PAN",
        "DEM_ELEV",
        "SPOT_RGBI",
        "SENTINEL2_TS",
        "SENTINEL1-ASC_TS",
        "SENTINEL1-DESC_TS",
    ]
    for label in config.get("labels", []):
        list_keys.append(label)

    print("\n[---DATA SPLIT---]")
    if config["tasks"].get("train", False):
        print("[TRAIN]")
        for key in list_keys:
            if dict_train and dict_train.get(key):
                print(f"- {key:20s}: {len(dict_train[key])} samples")
        print("[VAL]")
        for key in list_keys:
            if dict_val and dict_val.get(key):
                print(f"- {key:20s}: {len(dict_val[key])} samples")

    if config["tasks"].get("predict", False):
        print("[TEST]")
        for key in list_keys:
            if dict_test and dict_test.get(key):
                print(f"- {key:20s}: {len(dict_test[key])} samples")


@rank_zero_only
def print_inference_time(tt, config):
    tt = tt * (config["num_nodes"] * config["gpus_per_node"])
    print("", "", "#" * 80, " " * 28 + "--- INFERENCE TIME ---", sep="\n")
    print("- nodes: ", config["num_nodes"])
    print("- gpus per nodes: ", config["gpus_per_node"])
    print(
        "[MAX FOR VALID MODEL] : 0:25:00 HH:MM:SS",
        f"[CURRENT]             : {str(timedelta(seconds=tt))} HH:MM:SS",
        "",
        sep="\n",
    )
    if tt > 1500:
        print("[X] INFERENCE TOO LONG")
    else:
        print("[V] INFERENCE TIME BELOW MAX !", "\n\n")
    print("#" * 80, "\n\n")


@rank_zero_only
def print_metrics(miou, ious):
    classes = [
        "building",
        "greenhouse",
        "swimming_pool",
        "impervious surface",
        "pervious surface",
        "bare soil",
        "water",
        "snow",
        "herbaceous vegetation",
        "agricultural land",
        "plowed land",
        "vineyard",
        "deciduous",
        "coniferous",
        "brushwood",
        "clear cut",
        "ligneous",
        "mixed",
        "undefined",
    ]
    print("\n")
    print("-" * 40)
    print(" " * 8, "Model mIoU : ", round(miou, 4))
    print("-" * 40)
    print("{:<25} {:<15}".format("Class", "iou"))
    print("-" * 40)
    for k, v in zip(classes, ious):
        print("{:<25} {:<15}".format(k, round(v, 5)))
    print("\n\n")


@rank_zero_only
def print_config(config: DictConfig) -> None:
    """Print content of given config using Rich library and its tree structure.
    Args: config: Config to print to console using a Rich tree.
    """

    def walk_config(tree: Tree, config: DictConfig):
        """Recursive function to accumulate branch."""
        for group_name, group_option in config.items():
            if isinstance(group_option, dict):
                # print('HERE', group_name)
                branch = tree.add(
                    str(group_name), style=Style(color="yellow", bold=True)
                )
                walk_config(branch, group_option)
            elif isinstance(group_option, ListConfig):
                if not group_option:
                    # print('THERE')
                    tree.add(
                        f"{group_name}: []", style=Style(color="yellow", bold=True)
                    )
                else:
                    # print('THA')
                    tree.add(
                        f"{str(group_name)}: {group_option}",
                        style=Style(color="yellow", bold=True),
                    )
            else:
                if group_name == "_target_":
                    # print('THI')
                    tree.add(
                        f"{str(group_name)}: {group_option}",
                        style=Style(color="white", italic=True, bold=True),
                    )
                else:
                    # print('THO')
                    tree.add(
                        f"{str(group_name)}: {group_option}",
                        style=Style(color="yellow", bold=True),
                    )

    tree = Tree(
        ":deciduous_tree: Configuration Tree ",
        style=Style(color="white", bold=True, encircle=True),
        guide_style=Style(color="bright_green", bold=True),
        expanded=True,
        highlight=True,
    )
    walk_config(tree, config)
    get_console().print(tree)


@rank_zero_only
def print_iou_metrics(miou, ious):
    """
    mIoU computes the IoU for each class and avrerages the results
    over the class set
    """
    classes = [
        "building",
        "greenhouse",
        "swimming_pool",
        "impervious surface",
        "pervious surface",
        "bare soil",
        "water",
        "snow",
        "herbaceous vegetation",
        "agricultural land",
        "plowed land",
        "vineyard",
        "deciduous",
        "coniferous",
        "brushwood",
        "clear cut",
        "ligneous",
        "mixed",
        "undefined",
    ]

    print("\n")
    print("-" * 40)
    print(" " * 8, f"Model mIoU : ", round(miou, 4))
    print("-" * 40)
    print("{:<25} {:<15}".format("Class", "ious"))
    print("-" * 40)
    for k, v in zip(classes, ious):
        print("{:<25} {:<15}".format(k, v))
    print("\n\n")


@rank_zero_only
def print_f1_metrics(mf1, f1s):
    classes = [
        "building",
        "greenhouse",
        "swimming_pool",
        "impervious surface",
        "pervious surface",
        "bare soil",
        "water",
        "snow",
        "herbaceous vegetation",
        "agricultural land",
        "plowed land",
        "vineyard",
        "deciduous",
        "coniferous",
        "brushwood",
        "clear cut",
        "ligneous",
        "mixed",
        "undefined",
    ]

    print("\n")
    print("-" * 40)
    print(" " * 8, f"Model mf1 : ", round(mf1, 4))
    print("-" * 40)
    print("{:<25} {:<15}".format("Class", "f1s"))
    print("-" * 40)
    for k, v in zip(classes, f1s):
        print("{:<25} {:<15}".format(k, v))
    print("\n\n")


@rank_zero_only
def print_overall_accuracy(metric):
    """
    OA accounts for the precision of the prediction regardless
    of the class distribution!
    """

    print("\n")
    print("-" * 40)
    print(" " * 8, f"Model OA : ", round(metric, 4))
    print("-" * 40)
