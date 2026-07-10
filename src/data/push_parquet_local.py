import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset, Audio, Features, Value  # <-- SỬA DÒNG NÀY (Thêm Features, Value)
from huggingface_hub import login
import pyarrow as pa

# ==========================================
# 1. ENVIRONMENT & AUTHENTICATION SETUP
# ==========================================
# Automatically resolve the .env path from src/data/ back to the root directory
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

def get_required_env(env_name):
    value = os.getenv(env_name)
    if not value:
        raise ValueError(f"[ERROR] {env_name} is not set in {ENV_PATH}")
    return value

# Retrieve the token securely and log into Hugging Face Hub
TOKEN = get_required_env("HF_TOKEN")
login(token=TOKEN)

# ==========================================
# 2. CONFIGURATION
# ==========================================
REPO_ID = "HuyPham171/iemocap-sentiment-clean"
LOCAL_BASE_DIR = r"D:\Resfes\Project\iemocap_hf_upload"

CSV_PATH = os.path.join(LOCAL_BASE_DIR, "metadata.csv")
WAV_DIR = os.path.join(LOCAL_BASE_DIR, "wav")

# ==========================================
# 3. PARQUET COMPILATION & UPLOAD
# ==========================================
def main():
    # Read metadata via Pandas (Safely bypassing HF AudioFolder risks)
    print("[INFO] Loading metadata...")
    df = pd.read_csv(CSV_PATH)

    # Map each file_name to its absolute local path for processing
    df['audio'] = df['file_name'].apply(lambda x: os.path.join(WAV_DIR, x))

    # Convert to Hugging Face Dataset format
    print("[INFO] Packaging dataset...")
    custom_features = Features({
        'file_name': Value('string'),
        'label': Value('int64'),
        'Utterance_ID': Value('string'),
        'Session': Value('int64'),
        'audio': Value('string') 
    })
    ds = Dataset.from_pandas(df, features=custom_features)

    # Cast to Audio Feature (Reads wav files and encodes them directly into binary rows)
    print("[INFO] Embedding audio and compiling Parquet...")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Push to Hugging Face Hub
    print("[INFO] Uploading to Hugging Face Hub (this may take a few minutes)...")
    ds.push_to_hub(REPO_ID)

    print("[SUCCESS] Dataset successfully converted to Parquet and uploaded to the Hub!")

if __name__ == "__main__":
    main()