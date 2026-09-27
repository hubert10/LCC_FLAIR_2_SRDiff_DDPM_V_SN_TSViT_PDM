from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="IGNF/FLAIR-HUB",
    repo_type="dataset",
    local_dir="D:\kanyamahanga\Datasets\FLAIR_HUB",
)
