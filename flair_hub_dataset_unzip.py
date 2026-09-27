import os
import zipfile


def unzip_and_delete(folder_path: str):
    """
    Finds all ZIP files in a folder, extracts them into the same folder,
    and deletes the ZIP files afterwards.

    Args:
        folder_path (str): Path to the folder containing ZIP files.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".zip"):
            zip_path = os.path.join(folder_path, filename)
            extract_folder = folder_path

            print(f"Unzipping: {zip_path}")

            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_folder)
                print(f"Extracted to: {extract_folder}")

                # Delete the ZIP file after successful extraction
                os.remove(zip_path)
                print(f"Deleted ZIP: {zip_path}\n")

            except Exception as e:
                print(f"Failed to unzip {zip_path}: {e}")

    print("All ZIP files processed.")


# unzip_and_delete("D:\kanyamahanga\Datasets\FLAIR_HUB_TOY")
unzip_and_delete("/hubert_storage/FLAIR_HUB")
