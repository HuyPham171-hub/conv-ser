"""
Biometric Feature Extractor for IEMOCAP
Extracts Speaker Gender, Duration, and F0_Mean and appends them to the metadata.
"""

import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import warnings

# Ignore librosa warnings about short audio files
warnings.filterwarnings('ignore', category=UserWarning)

# =====================================================================
# 1. PATH CONFIGURATION
# =====================================================================
METADATA_PATH = r"d:\Resfes\Project\Ser\data\DataFrames\iemocap_metadata.csv"
IEMOCAP_ROOT = r"d:\Resfes\Project\IEMOCAP_full_release"

print("[INFO] Loading existing metadata...")
df = pd.read_csv(METADATA_PATH)

# Initialize new columns with NaN
df['Speaker_Gender'] = None
df['Duration_sec'] = np.nan
df['F0_Mean'] = np.nan

print("[INFO] Starting biometric feature extraction...")

# =====================================================================
# 2. EXTRACTION LOOP
# =====================================================================
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Audio Files"):
    utt_id = row['Utterance_ID']
    dialog_id = row['Dialog_ID']
    session_num = row['Session']
    
    # ---------------------------------------------------
    # A. Extract Gender from Utterance ID
    # Example: 'Ses01F_impro01_F000' -> Split by '_' -> 'F000' -> First char 'F'
    # ---------------------------------------------------
    speaker_code = utt_id.split('_')[-1]
    gender = 'F' if speaker_code.startswith('F') else 'M'
    df.at[index, 'Speaker_Gender'] = gender
    
    # ---------------------------------------------------
    # B. Construct the exact path to the .wav file
    # Format: IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav
    # ---------------------------------------------------
    session_folder = f"Session{session_num}"
    wav_path = os.path.join(IEMOCAP_ROOT, session_folder, "sentences", "wav", dialog_id, f"{utt_id}.wav")
    
    if not os.path.exists(wav_path):
        continue # Skip if file is missing for any reason
        
    try:
        # Load audio file (sr=16000 is standard for speech processing)
        y, sr = librosa.load(wav_path, sr=16000)
        
        # ---------------------------------------------------
        # C. Calculate Duration in seconds
        # ---------------------------------------------------
        duration = librosa.get_duration(y=y, sr=sr)
        df.at[index, 'Duration_sec'] = round(duration, 3)
        
        # ---------------------------------------------------
        # D. Calculate F0 (Fundamental Frequency)
        # We use librosa.yin (faster than pyin) for bulk processing.
        # fmin=50Hz (deep male), fmax=400Hz (high female/screaming)
        # ---------------------------------------------------
        # YIN algorithm returns F0 array. Unvoiced frames (silence) are usually NaN or out of bounds.
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        
        # Filter out extreme anomalies and calculate the mean
        valid_f0 = f0[(f0 > 50) & (f0 < 400)]
        if len(valid_f0) > 0:
            df.at[index, 'F0_Mean'] = round(np.mean(valid_f0), 3)
        else:
            df.at[index, 'F0_Mean'] = 0.0 # Fallback for purely silent/unvoiced files
            
    except Exception as e:
        print(f"\n[WARNING] Failed to process {utt_id}: {e}")

# =====================================================================
# 3. SAVE THE UPDATED METADATA
# =====================================================================
# Save over the old file to maintain the single source of truth
df.to_csv(METADATA_PATH, index=False)

print("\n" + "="*50)
print(f"[SUCCESS] Biometric features extracted and appended to {METADATA_PATH}")
print(f"Missing F0 Samples: {df['F0_Mean'].isna().sum()}")
print("="*50)