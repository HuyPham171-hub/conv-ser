import pandas as pd
from pathlib import Path

# =====================================================================
# 1. PATH RESOLUTION & CONFIGURATION
# =====================================================================
# Adjust the root path according to your project structure
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]

# Input: The non-consensus dataset extracted previously
XXX_METADATA_PATH = PROJECT_ROOT / "data" / "DataFrames" / "iemocap_metadata_xxx.csv"

# Output: The strictly filtered dataset containing rescued samples
RESCUED_OUTPUT_PATH = PROJECT_ROOT / "data" / "DataFrames" / "iemocap_metadata_xxx_rescued.csv"

def main():
    print("[INFO] Starting Rule-based Thresholding for Sentiment Pseudo-labeling...")
    
    if not XXX_METADATA_PATH.exists():
        raise FileNotFoundError(f"[ERROR] Source file not found at: {XXX_METADATA_PATH}")

    # Load the ambiguous metadata
    df = pd.read_csv(XXX_METADATA_PATH)
    initial_count = len(df)
    print(f"[INFO] Loaded {initial_count} ambiguous 'xxx' utterances.")

    # Initialize the target column with -1 (Discard/Unknown)
    # 0 = Negative, 1 = Neutral, 2 = Positive
    df['Stage1_Label'] = -1 
    
    # =====================================================================
    # 2. APPLY HEURISTIC RULES (V-A-D BOUNDARIES)
    # =====================================================================
    
    # Rule 1: High confidence Negative (Valence is extremely low)
    mask_negative = df['Valence'] <= 2.5
    
    # Rule 2: High confidence Positive (Valence is extremely high)
    mask_positive = df['Valence'] >= 3.5
    
    # Rule 3: High confidence Neutral (Valence is perfectly balanced AND low energy)
    # Arousal <= 3.0 prevents aggressive/excited outliers from bleeding into Neutral
    mask_neutral = (df['Valence'] >= 2.75) & (df['Valence'] <= 3.25) & (df['Arousal'] <= 3.0)

    # Apply the masks to assign Stage 1 Sentiment labels
    df.loc[mask_negative, 'Stage1_Label'] = 0
    df.loc[mask_neutral, 'Stage1_Label'] = 1
    df.loc[mask_positive, 'Stage1_Label'] = 2

    # =====================================================================
    # 3. FILTER AND EXPORT RESCUED DATA
    # =====================================================================
    
    # Keep only the rows that successfully passed the thresholding rules
    df_rescued = df[df['Stage1_Label'] != -1].copy()
    
    # Add a flag to distinguish pseudo-labels from human ground-truth later on
    df_rescued['is_pl'] = 1 
    
    # Save the rescued dataframe
    RESCUED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_rescued.to_csv(RESCUED_OUTPUT_PATH, index=False)

    # =====================================================================
    # 4. REPORT STATISTICS
    # =====================================================================
    rescued_count = len(df_rescued)
    discarded_count = initial_count - rescued_count
    
    print("\n" + "="*60)
    print("[STATISTICS] RESCUE OPERATION SUMMARY")
    print("="*60)
    print(f"Total 'xxx' samples processed : {initial_count}")
    print(f"Successfully rescued samples  : {rescued_count} ({(rescued_count/initial_count)*100:.2f}%)")
    print(f"Discarded samples (Gray Zone) : {discarded_count} ({(discarded_count/initial_count)*100:.2f}%)")
    print("-" * 60)
    
    if rescued_count > 0:
        # Map back to string names for printing
        sentiment_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        distribution = df_rescued['Stage1_Label'].map(sentiment_names).value_counts()
        print("Rescued Sentiment Distribution:")
        print(distribution.to_string())
        
    print("="*60)
    print(f"[SUCCESS] Rescued metadata strictly saved at: {RESCUED_OUTPUT_PATH}")

if __name__ == "__main__":
    main()