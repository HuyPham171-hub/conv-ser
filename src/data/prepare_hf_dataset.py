import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
METADATA_PATH = Path(r"d:\Resfes\Project\Ser\data\DataFrames\iemocap_metadata.csv")
IEMOCAP_ROOT = Path(r"d:\Resfes\Project\IEMOCAP_full_release")

TARGET_BASE_DIR = Path(r"D:\Resfes\Project\iemocap_hf_upload")
TARGET_WAV_DIR = TARGET_BASE_DIR / "wav"
TARGET_METADATA_PATH = TARGET_BASE_DIR / "metadata.csv"

# Ensure target directories exist
TARGET_WAV_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. UNIFIED PIPELINE (COPY WAVES & GENERATE METADATA)
# ==========================================
def prepare_huggingface_dataset():
    print(f"[INFO] Reading baseline metadata from: {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
    
    # Define mapping dictionaries for both tasks
    emotion_map = {
        'ang': 0, 'dis': 1, 'exc': 2, 'fea': 3, 'fru': 4, 
        'hap': 5, 'neu': 6, 'sad': 7, 'sur': 8
    }
    
    sentiment_map = {
        'ang': 0, 'sad': 0, 'fea': 0, 'fru': 0, 'dis': 0, # Negative
        'neu': 1, 'sur': 1,                               # Neutral
        'hap': 2, 'exc': 2                                # Positive
    }
    
    # Filter out invalid emotions (e.g., 'xxx', 'oth')
    filtered_df = df[df["Raw_Emotion"].isin(emotion_map.keys())].copy()
    print(f"[INFO] Found {len(filtered_df)} valid utterances for the unified dataset.")
    
    # --- PHASE A: AUDIO FILE TRANSFER ---
    success_count = 0
    missing_count = 0
    
    print("\n[INFO] Initializing audio file transfer pipeline...")
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Copying Audio Files"):
        session_name = f"Session{row['Session']}"
        dialog_id = row['Dialog_ID']
        utterance_id = row['Utterance_ID']
        
        source_audio_path = IEMOCAP_ROOT / session_name / "sentences" / "wav" / dialog_id / f"{utterance_id}.wav"
        target_audio_path = TARGET_WAV_DIR / f"{utterance_id}.wav"
        
        if source_audio_path.exists():
            # I/O Optimization: Only copy if the file does not already exist
            if not target_audio_path.exists():
                shutil.copy2(source_audio_path, target_audio_path)
            success_count += 1
        else:
            missing_count += 1
            
    print(f" -> Valid files mapped: {success_count} files in {TARGET_WAV_DIR}")
    if missing_count > 0:
        print(f" -> [WARNING] Skipped {missing_count} missing source files on disk.")

    # --- PHASE B: HUGGING FACE METADATA GENERATION ---
    print("\n[INFO] Generating unified Hugging Face metadata.csv...")
    
    filtered_df["file_name"] = filtered_df["Utterance_ID"].apply(lambda uid: f"wav/{uid}.wav")
    
    # Inject both granular emotions and macro-sentiments
    filtered_df["emotion_label"] = filtered_df["Raw_Emotion"].map(emotion_map)
    filtered_df["sentiment_label"] = filtered_df["Raw_Emotion"].map(sentiment_map)
    
    # Construct the final unified DataFrame
    hf_df = filtered_df[[
        "file_name", 
        "emotion_label", 
        "sentiment_label", 
        "Utterance_ID", 
        "Session", 
        "Raw_Emotion"
    ]]
    
    hf_df.to_csv(TARGET_METADATA_PATH, index=False)
    
    print(f"[SUCCESS] Unified metadata successfully written to: {TARGET_METADATA_PATH}")

if __name__ == "__main__":
    prepare_huggingface_dataset()