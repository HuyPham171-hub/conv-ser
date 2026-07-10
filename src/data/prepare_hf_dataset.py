import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
# Ground truth paths matching your local infrastructure
METADATA_PATH = Path(r"d:\Resfes\Project\Ser\data\DataFrames\iemocap_metadata.csv")
IEMOCAP_ROOT = Path(r"d:\Resfes\Project\IEMOCAP_full_release")

# Target structure for Hugging Face upload
TARGET_BASE_DIR = Path(r"D:\Resfes\Project\iemocap_hf_upload")
TARGET_WAV_DIR = TARGET_BASE_DIR / "wav"
TARGET_METADATA_PATH = TARGET_BASE_DIR / "metadata.csv"

# Ensure the target directories exist before execution
TARGET_WAV_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. UNIFIED PIPELINE (COPY WAVES & GENERATE METADATA)
# ==========================================
def prepare_huggingface_dataset():
    # Load your baseline metadata sheet
    print(f"[INFO] Reading baseline metadata from: {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
    
    # Define your 3-class macro-sentiment target keys
    sentiment_map = {
        'hap': 2, 'exc': 2, 
        'neu': 1, 'sur': 1, 
        'ang': 0, 'sad': 0, 'fea': 0, 'fru': 0
    }
    
    # Filter metadata to keep only the valid emotional classes
    filtered_df = df[df["Raw_Emotion"].isin(sentiment_map.keys())].copy()
    print(f"[INFO] Found {len(filtered_df)} utterances matching the target 3-class sentiment criteria.")
    
    # --- PHASE A: AUDIO FILE TRANSFER ---
    success_count = 0
    missing_count = 0
    
    print("\n[INFO] Initializing audio file transfer pipeline...")
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Copying Audio Files"):
        session_name = f"Session{row['Session']}"
        dialog_id = row['Dialog_ID']
        utterance_id = row['Utterance_ID']
        
        # Reconstruct the absolute source path of the utterance
        source_audio_path = IEMOCAP_ROOT / session_name / "sentences" / "wav" / dialog_id / f"{utterance_id}.wav"
        target_audio_path = TARGET_WAV_DIR / f"{utterance_id}.wav"
        
        # Check file existence to avoid runtime failures
        if source_audio_path.exists():
            shutil.copy2(source_audio_path, target_audio_path)
            success_count += 1
        else:
            missing_count += 1
            
    print(f" -> Successfully copied : {success_count} files to {TARGET_WAV_DIR}")
    if missing_count > 0:
        print(f" -> [WARNING] Skipped {missing_count} missing source files on disk.")

    # --- PHASE B: HUGGING FACE METADATA GENERATION ---
    print("\n[INFO] Generating Hugging Face compliant metadata.csv...")
    
    # Create 'file_name' column formatted relatively for Hugging Face (e.g., Ses01F_impro01_F000.wav)
    filtered_df["file_name"] = filtered_df["Utterance_ID"].apply(lambda uid: f"{uid}.wav")
    
    # Rename 'Emotion' to 'label' and map to numerical classes
    filtered_df["label"] = filtered_df["Raw_Emotion"].map(sentiment_map)
    
    hf_df = filtered_df[["file_name", "label", "Utterance_ID", "Session"]].copy()
    hf_df = hf_df.dropna(how='all')                  
    hf_df = hf_df.dropna(subset=['file_name'])        
    hf_df['file_name'] = hf_df['file_name'].astype(str)
    
    # Export to CSV without the index
    hf_df.to_csv(TARGET_METADATA_PATH, index=False, lineterminator='\n')
    
    print(f"[SUCCESS] Metadata successfully written to: {TARGET_METADATA_PATH}")
    print("\n[SUCCESS] Unified pipeline completed. The dataset is ready for Hugging Face upload!")

if __name__ == "__main__":
    prepare_huggingface_dataset()