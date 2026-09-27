import os
import zipfile


def zip_subfolders(root_folder, output_folder=None):
    """
    Zips each subfolder inside root_folder into a separate ZIP file.

    :param root_folder: Path containing subfolders to zip
    :param output_folder: Where to save ZIP files (optional).
                          Default = same as root_folder.
    """
    root_folder = os.path.abspath(root_folder)
    if output_folder is None:
        output_folder = root_folder
    else:
        output_folder = os.path.abspath(output_folder)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)

        # Only zip directories (subfolders)
        if os.path.isdir(item_path):
            zip_path = os.path.join(output_folder, f"{item}.zip")
            print(f"Zipping: {item_path} → {zip_path}")

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for folder, subfolders, files in os.walk(item_path):
                    for file in files:
                        full_path = os.path.join(folder, file)
                        # Write file with relative path inside zip
                        rel_path = os.path.relpath(full_path, item_path)
                        zipf.write(full_path, arcname=os.path.join(item, rel_path))

    print("Done.")


zip_subfolders(
    "D:\kanyamahanga\Datasets\FLAIR_HUB_TOY",
    "D:\kanyamahanga\Datasets\FLAIR_HUB_TOY_ZIP",
)
