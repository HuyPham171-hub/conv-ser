from huggingface_hub import HfApi, login
import os
import tarfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(f"[ERROR] HF_TOKEN is not set in {ENV_PATH}")
login(token=HF_TOKEN)

REPO_ID = "HuyPham171/iemocap-stage1-checkpoints"
LOCAL_DIR = Path(r"D:\Resfes\Project\Ser\checkpoints\cross_attention_stage1")
TAR_PATH = LOCAL_DIR / "stage1_checkpoints.tar"

def main():
    api = HfApi()
    
    print(f"[INFO] Creating dataset repository: {REPO_ID}...")
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    
    # 1. Pack all content into a single .tar archive to prevent directory file limit issues
    if not TAR_PATH.exists():
        print(f"[INFO] Packing Stage 1 checkpoints into {TAR_PATH.name}...")
        with tarfile.open(TAR_PATH, "w") as tar:
            # Recursively find all files inside cross_attention_stage1 (excluding the tar itself)
            for path in LOCAL_DIR.rglob("*"):
                if path.is_file() and path != TAR_PATH:
                    # Maintain structural positioning relative to LOCAL_DIR
                    arcname = path.relative_to(LOCAL_DIR)
                    tar.add(path, arcname=arcname)
        print("[SUCCESS] Packing complete.")
    else:
        print(f"[INFO] {TAR_PATH.name} already exists. Skipping packing.")

    # 2. Upload only the packed archive to Hugging Face
    print(f"[INFO] Uploading packed archive to Hugging Face...")
    allow_patterns = ["*.tar"]
    
    api.upload_folder(
        folder_path=LOCAL_DIR,
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=allow_patterns, 
        token=HF_TOKEN,
        commit_message="Upload packed Stage 1 checkpoints for Soft Gating routing"
    )
    
    print("[SUCCESS] Stage 1 Checkpoints Upload complete!")

if __name__ == "__main__":
    main()