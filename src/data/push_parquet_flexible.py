import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset, Audio, Features, Value
from huggingface_hub import login

# ==========================================
# 1. ENVIRONMENT & AUTHENTICATION SETUP
# ==========================================
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

def get_required_env(env_name):
    value = os.getenv(env_name)
    if not value:
        raise ValueError(f"[ERROR] {env_name} is not set in {ENV_PATH}")
    return value

TOKEN = get_required_env("HF_TOKEN")
login(token=TOKEN)

# ==========================================
# 2. FLEXIBLE TASK CONFIGURATION
# ==========================================
# Toggle between "EMOTION" and "SENTIMENT" depending on your target upload
TASK = "EMOTION"  # Change to "SENTIMENT" if you ever need to re-upload the other one

LOCAL_BASE_DIR = r"D:\Resfes\Project\iemocap_hf_upload"
CSV_PATH = os.path.join(LOCAL_BASE_DIR, "metadata.csv")
WAV_DIR = os.path.join(LOCAL_BASE_DIR, "wav")

# Dynamically resolve target repository and labels based on the selected task
if TASK == "EMOTION":
    REPO_ID = "HuyPham171/iemocap-emotion-clean"
    TARGET_LABEL_COLUMN = "emotion_label"
elif TASK == "SENTIMENT":
    REPO_ID = "HuyPham171/iemocap-sentiment-clean"
    TARGET_LABEL_COLUMN = "sentiment_label"
else:
    raise ValueError(f"[ERROR] Invalid TASK mode: {TASK}. Choose 'EMOTION' or 'SENTIMENT'.")

# ==========================================
# 3. PARQUET COMPILATION & UPLOAD
# ==========================================
def main():
    print(f"[INFO] Initializing upload pipeline for task: {TASK}")
    print(f"[INFO] Target Repository: {REPO_ID}")
    
    print("[INFO] Loading unified metadata.csv...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"[ERROR] Unified metadata file not found at: {CSV_PATH}. Please run prepare_hf_dataset.py first.")
        
    df = pd.read_csv(CSV_PATH)

    # Standardize the target label column to 'label' as required by Hugging Face Trainer API
    print(f"[INFO] Mapping column '{TARGET_LABEL_COLUMN}' as the primary training 'label'...")
    if TARGET_LABEL_COLUMN not in df.columns:
        raise KeyError(f"[ERROR] Column '{TARGET_LABEL_COLUMN}' missing from metadata. Check your prepare_hf_dataset.py generation.")
    
    df['label'] = df[TARGET_LABEL_COLUMN]
    df['audio'] = df['file_name'].apply(lambda x: os.path.join(LOCAL_BASE_DIR, x))

    print("[INFO] Defining flexible dataset schema...")
    custom_features = Features({
        'file_name': Value('string'),
        'label': Value('int64'),            # Dynamic label assigned above
        'emotion_label': Value('int64'),    # Kept for research reference
        'sentiment_label': Value('int64'),  # Kept for research reference
        'Utterance_ID': Value('string'),
        'Session': Value('int64'),
        'Raw_Emotion': Value('string'),
        'audio': Value('string') 
    })
    
    ds = Dataset.from_pandas(df, features=custom_features)

    print("[INFO] Embedding audio data and compiling Parquet format...")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    print(f"[INFO] Uploading to target repository: {REPO_ID} (This may take a while)...")
    ds.push_to_hub(REPO_ID)

    print(f"[SUCCESS] Unified dataset successfully processed and uploaded to {REPO_ID} under {TASK} configuration!")

if __name__ == "__main__":
    main()