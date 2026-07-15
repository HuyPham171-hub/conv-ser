import re
import pandas as pd
from pathlib import Path

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
# Update these paths to match your actual local layout
INPUT_METADATA_PATH = r"d:\Resfes\Project\Ser\data\DataFrames\iemocap_metadata.csv"
OUTPUT_METADATA_PATH = r"d:\Resfes\Project\Ser\data\DataFrames\iemocap_metadata_clean.csv"

def clean_transcript(text):
    """
    Cleans raw dialogue text by removing acoustic annotations like [laughter], [gasp], 
    transcription artifacts, and stripping redundant whitespaces.
    """
    if pd.isna(text):
        return ""
    # Convert to string and replace text inside brackets/parentheses
    text = str(text)
    text = re.sub(r'\[.*?\]', '', text) # Removes [laughter], [gasp], etc.
    text = re.sub(r'\(.*?\)', '', text) # Removes (laughter), (gasp), etc.
    # Remove multiple spaces and strip ends
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("[INFO] Starting Metadata Cleansing Pipeline...")
    
    # Check input file existence
    input_file = Path(INPUT_METADATA_PATH)
    if not input_file.exists():
        raise FileNotFoundError(f"[ERROR] Source metadata not found at: {INPUT_METADATA_PATH}")
        
    # Load raw metadata
    df = pd.read_csv(INPUT_METADATA_PATH)
    print(f"[INFO] Raw metadata contains {len(df)} initial rows.")
    
    # 2. DEFINE SENTIMENT ROUTING MAP
    # 0 = Negative, 1 = Neutral, 2 = Positive
    sentiment_map = {
        'ang': 0, 'sad': 0, 'fea': 0, 'fru': 0, 'dis': 0, # Negative
        'neu': 1, 'sur': 1,                               # Neutral / Ambiguous
        'hap': 2, 'exc': 2                                # Positive
    }
    
    # 3. FILTER VALID EMOTIONS (Drops 'xxx', 'oth')
    print("[INFO] Filtering out unmapped emotions ('xxx', 'oth')...")
    df_clean = df[df["Raw_Emotion"].isin(sentiment_map.keys())].copy()
    
    # 4. INJECT NUMERICAL LABELS FOR MULTIMODAL PIPELINE
    df_clean["Sentiment_Label"] = df_clean["Raw_Emotion"].map(sentiment_map)
    
    # 5. TEXT CLEANING FOR ROBERTA LEXICAL ANCHOR
    print("[INFO] Cleaning structural noise from 'Transcript' column...")
    df_clean["Transcript"] = df_clean["Transcript"].apply(clean_transcript)
    
    # Drop rows where transcripts became completely empty after cleaning (if any)
    df_clean = df_clean[df_clean["Transcript"] != ""].reset_index(drop=True)
    
    # 6. VERIFY & SAVE RESULTS
    print(f"\n[SUCCESS] Cleansing complete. Total clean rows: {len(df_clean)}")
    print("\n--- Sentiment Label Distribution ---")
    counts = df_clean["Sentiment_Label"].value_counts().rename(index={0: "Negative (0)", 1: "Neutral (1)", 2: "Positive (2)"})
    print(counts)
    
    # Save clean metadata to disk
    df_clean.to_csv(OUTPUT_METADATA_PATH, index=False)
    print(f"\n[INFO] Cleaned metadata successfully saved to: {OUTPUT_METADATA_PATH}")

if __name__ == "__main__":
    main()