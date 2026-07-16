import os
import shutil
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import Dataset, Audio, Features, Value
from huggingface_hub import login

# ==========================================
# 1. PATH & ENVIRONMENT CONFIGURATION
# ==========================================
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

# Source data paths
RESCUED_CSV_PATH = Path(r"d:\Resfes\Project\Ser\data\DataFrames\iemocap_metadata_xxx_rescued.csv")
IEMOCAP_ROOT = Path(r"d:\Resfes\Project\IEMOCAP_full_release")

# NEW staging directory for rescued data to avoid overriding clean dataset
TARGET_BASE_DIR = Path(r"D:\Resfes\Project\iemocap_rescued_hf_upload")
TARGET_WAV_DIR = TARGET_BASE_DIR / "wav"
TARGET_METADATA_PATH = TARGET_BASE_DIR / "metadata.csv"

TARGET_WAV_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face configuration
TOKEN = os.getenv("HF_TOKEN")
if not TOKEN:
    raise ValueError(f"[ERROR] HF_TOKEN is not set in {ENV_PATH}")
login(token=TOKEN)

REPO_ID = "HuyPham171/iemocap-sentiment-rescued"

def main():
    if not RESCUED_CSV_PATH.exists():
        raise FileNotFoundError(f"[ERROR] Rescued CSV metadata not found at: {RESCUED_CSV_PATH}")

    print(f"[INFO] Reading rescued metadata from: {RESCUED_CSV_PATH}")
    df = pd.read_csv(RESCUED_CSV_PATH)
    
    # Standardize column name to 'label' as required by HF Trainer API
    df['label'] = df['Stage1_Label']

    # ==========================================
    # PHASE A: AUDIO FILE TRANSFER (LOCAL WORK)
    # ==========================================
    success_count = 0
    missing_count = 0
    
    print("\n[INFO] Extracting raw audio files for rescued utterances...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying Rescued Audio"):
        session_name = f"Session{row['Session']}"
        dialog_id = row['Dialog_ID']
        utterance_id = row['Utterance_ID']
        
        source_audio_path = IEMOCAP_ROOT / session_name / "sentences" / "wav" / dialog_id / f"{utterance_id}.wav"
        target_audio_path = TARGET_WAV_DIR / f"{utterance_id}.wav"
        
        if source_audio_path.exists():
            if not target_audio_path.exists():
                shutil.copy2(source_audio_path, target_audio_path)
            success_count += 1
        else:
            missing_count += 1
            
    print(f" -> Successfully mapped: {success_count} files to {TARGET_WAV_DIR}")
    if missing_count > 0:
        print(f" -> [WARNING] Skipped {missing_count} missing source wav files.")

    # ==========================================
    # PHASE B: PARQUET PARSING & HUB UPLOAD
    # ==========================================
    print("\n[INFO] Compiling Hugging Face metadata schema...")
    
    # Generate schema columns matching clean dataset format
    df["file_name"] = df["Utterance_ID"].apply(lambda uid: f"wav/{uid}.wav")
    df['audio'] = df['file_name'].apply(lambda x: os.path.join(TARGET_BASE_DIR, x))

    custom_features = Features({
        'file_name': Value('string'),
        'label': Value('int64'),
        'Utterance_ID': Value('string'),
        'Session': Value('int64'),
        'Raw_Emotion': Value('string'),
        'audio': Value('string') 
    })
    
    # Filter only schema columns
    hf_df = df[["file_name", "label", "Utterance_ID", "Session", "Raw_Emotion", "audio"]]
    
    # Save a local metadata reference copy
    hf_df.to_csv(TARGET_METADATA_PATH, index=False)
    
    # Build Hugging Face Dataset objects
    ds = Dataset.from_pandas(hf_df, features=custom_features)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    print(f"\n[INFO] Compiling Parquet and pushing to target repository: {REPO_ID}")
    ds.push_to_hub(REPO_ID)
    
    print(f"\n[VICTORY] Rescued dataset successfully processed and uploaded to {REPO_ID}!")

if __name__ == "__main__":
    main()