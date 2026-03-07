from huggingface_hub import snapshot_download

# Download an entire model repository
local_dir_path = snapshot_download(repo_id="Yunzhe/Hyper-SET", local_dir="./pretrained_weights/")
print(f"Repository downloaded to: {local_dir_path}")
