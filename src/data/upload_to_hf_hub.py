import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

# ==========================================
# 1. ENVIRONMENT SETUP
# ==========================================
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

def get_required_env(env_name):
    value = os.getenv(env_name)
    if not value:
        raise ValueError(f"[ERROR] {env_name} is not set in {ENV_PATH}")
    return value

# ==========================================
# 2. CONFIGURATION
# ==========================================
api = HfApi()
TOKEN = get_required_env("HF_TOKEN")
REPO_ID = "HuyPham171/iemocap-sentiment-clean"  
LOCAL_DIR = r"D:\Resfes\Project\iemocap_hf_upload" 

# ==========================================
# 3. UPLOAD EXECUTION
# ==========================================
def upload_dataset():
    print(f"[INFO] Initializing connection to Hugging Face Hub...")
    print(f"[INFO] Target Repository: {REPO_ID}")

    # --- PHASE 1: CLEAN UP OLD REPOSITORY ASSETS ---
    # Since the new structure only requires metadata.csv and wav.zip, 
    # we completely remove any legacy files or the old 'wav/' directory if they exist on the Hub.
    items_to_delete = ["wav", "metadata.csv", "wav.zip"]
    for item in items_to_delete:
        try:
            api.delete_file(
                repo_id=REPO_ID,
                repo_type="dataset",
                path_in_repo=item,
                token=TOKEN
            )
            print(f"[SUCCESS] Deleted old '{item}' from Hub.")
        except Exception:
            # Safely ignore if the file or folder does not exist
            pass

    # --- PHASE 2: UPLOAD NEW FLAT DATASTRUCTURE ---
    print(f"\n[INFO] Source Directory: {LOCAL_DIR}")
    print(f"[INFO] Starting upload process (Uploading metadata.csv and wav.zip)...")
    
    try:
        api.upload_folder(
            folder_path=LOCAL_DIR,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=TOKEN,
        )
        print("\n[SUCCESS] All audio files and metadata have been uploaded perfectly!")
    except Exception as e:
        print(f"\n[ERROR] Upload failed: {e}")

if __name__ == "__main__":
    upload_dataset()