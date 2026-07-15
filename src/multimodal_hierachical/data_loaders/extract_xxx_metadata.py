import pandas as pd
from pathlib import Path

# ==========================================
# PATH CONFIGURATION
# ==========================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2] # Adjust if the script is placed in a different directory structure

INPUT_METADATA_PATH = PROJECT_ROOT / "data" / "DataFrames" / "iemocap_metadata.csv"
OUTPUT_XXX_PATH = PROJECT_ROOT / "data" / "DataFrames" / "iemocap_metadata_xxx.csv"

def main():
    print("[INFO] Starting extraction of non-consensus data (xxx)...")
    
    if not INPUT_METADATA_PATH.exists():
        raise FileNotFoundError(f"[ERROR] Source file not found at: {INPUT_METADATA_PATH}")

    # Load raw metadata
    df = pd.read_csv(INPUT_METADATA_PATH)
    total_rows = len(df)
    
    # Filter rows specifically with the 'xxx' label
    df_xxx = df[df['Raw_Emotion'] == 'xxx'].copy()
    
    xxx_count = len(df_xxx)
    print(f"[INFO] Total initial utterances: {total_rows}")
    print(f"[INFO] Number of 'xxx' utterances found: {xxx_count}")
    
    if xxx_count == 0:
        print("[WARNING] No 'xxx' labels found. Please double-check the source file.")
        return

    # Export to a new CSV file
    OUTPUT_XXX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_xxx.to_csv(OUTPUT_XXX_PATH, index=False)
    
    print(f"\n[SUCCESS] Extraction complete! File saved at: {OUTPUT_XXX_PATH}")

if __name__ == "__main__":
    main()