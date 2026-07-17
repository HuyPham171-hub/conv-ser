from huggingface_hub import HfApi, login
import os
import tarfile
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(f"[ERROR] HF_TOKEN is not set in {ENV_PATH}")
login(token=HF_TOKEN)

REPO_ID = "HuyPham171/iemocap-dualband-spectrograms"
LOCAL_DIR = Path(r"D:\Resfes\Project\Ser\data\Embeddings\Visual\DualBand_Spectrograms")
TAR_PATH = LOCAL_DIR / "spectrograms.tar"

def main():
    api = HfApi()
    
    print(f"[INFO] Creating dataset repository: {REPO_ID}...")
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    
    # 1. Pack all .pt files into a single .tar archive to bypass the 10k file limit per directory
    if not TAR_PATH.exists():
        print(f"[INFO] Packing .pt files into {TAR_PATH.name} (This avoids the HF 10k file limit)...")
        with tarfile.open(TAR_PATH, "w") as tar:
            for pt_file in LOCAL_DIR.glob("*.pt"):
                tar.add(pt_file, arcname=pt_file.name)
        print("[SUCCESS] Packing complete.")
    else:
        print(f"[INFO] {TAR_PATH.name} already exists. Skipping packing.")

    # 2. Upload only the archive and metadata files
    print(f"[INFO] Uploading to Hugging Face...")
    allow_patterns = ["*.tar", "*.csv"]
    
    api.upload_folder(
        folder_path=LOCAL_DIR,
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=allow_patterns, # Only upload the archive and metadata
        token=HF_TOKEN,
        commit_message="Upload packed spectrograms and metadata"
    )
    
    print("[SUCCESS] Upload complete!")

if __name__ == "__main__":
    main()