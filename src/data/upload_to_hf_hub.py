import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

# ==========================================
# 1. ENVIRONMENT SETUP
# ==========================================
# Safely resolve the .env file path (Assuming script is in src/data/)
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
# Initialize the Hugging Face API client
api = HfApi()

# Retrieve the token securely from the environment variables
# Make sure to add HF_TOKEN=your_token_here to your .env file
TOKEN = get_required_env("HF_TOKEN")

# Target repository on Hugging Face (Replace 'username' with your actual HF username)
REPO_ID = "HuyPham171/iemocap-sentiment-clean"  

# Local directory containing the prepared dataset (wav/ and metadata.csv)
LOCAL_DIR = r"D:\Resfes\Project\iemocap_hf_upload" 

# ==========================================
# 3. UPLOAD EXECUTION
# ==========================================
def upload_dataset():
    print(f"[INFO] Initializing connection to Hugging Face Hub...")
    print(f"[INFO] Target Repository: {REPO_ID}")
    print(f"[INFO] Source Directory: {LOCAL_DIR}")
    print(f"\n[INFO] Starting upload process (This may take some time depending on your bandwidth)...")
    
    api.upload_folder(
        folder_path=LOCAL_DIR,
        repo_id=REPO_ID,
        repo_type="dataset",
        token=TOKEN
    )
    
    print("\n[SUCCESS] All audio files and metadata have been uploaded perfectly!")

if __name__ == "__main__":
    upload_dataset()