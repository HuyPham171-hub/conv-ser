import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

def get_required_path(env_name):
    """Fetches a directory path from the environment and expands user variables."""
    value = os.getenv(env_name)
    if not value:
        raise ValueError(f"[ERROR] {env_name} is not set in {ENV_PATH}")
    return Path(value).expanduser()

IEMOCAP_ROOT_DIR = get_required_path("IEMOCAP_ROOT_DIR")
DATAFRAMES_DIR = get_required_path("DATAFRAMES_DIR")

# Define the absolute path to the target metadata CSV sheet
METADATA_CSV_PATH = DATAFRAMES_DIR / "iemocap_metadata.csv"


def extract_transcripts(iemocap_root):
    """
    Scans through the 5 IEMOCAP sessions, parses the .txt files in the 
    transcriptions directory, and extracts Utterance_IDs with their corresponding text.
    """
    print("[INFO] Starting to scan transcription files...")
    transcript_dict = {}
    
    # Iterate through Session 1 to Session 5
    for session in range(1, 6):
        session_name = f"Session{session}"
        transcriptions_dir = os.path.join(iemocap_root, session_name, 'dialog', 'transcriptions')
        
        if not os.path.exists(transcriptions_dir):
            print(f"[WARNING] Directory not found: {transcriptions_dir}")
            continue
            
        # Retrieve all .txt files in the directory
        txt_files = list(transcriptions_dir.glob('*.txt'))
        
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    # Skip empty lines
                    if not line:
                        continue
                        
                    # Target format: "Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me."
                    # Split string at the first occurrence of "]: "
                    if "]: " in line:
                        parts = line.split("]: ", 1)
                        
                        # Parse the prefix: "Ses01F_impro01_F000 [006.2901-008.2357"
                        id_and_time = parts[0].split(" ")
                        utterance_id = id_and_time[0].strip()
                        
                        # Parse the text component
                        text = parts[1].strip()
                        
                        transcript_dict[utterance_id] = text

    print(f"[SUCCESS] Extracted {len(transcript_dict)} utterances from text files.")
    return transcript_dict

def merge_transcripts_to_csv(transcript_dict, csv_path):
    """
    Loads the current metadata file, appends the Transcript column, and overwrites the CSV.
    """
    print(f"[INFO] Loading current metadata from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    original_shape = df.shape
    
    # Map text from the dictionary to the DataFrame based on Utterance_ID
    df['Transcript'] = df['Utterance_ID'].map(transcript_dict)
    
    # Verify if any utterances are missing text alignments
    missing_text_count = df['Transcript'].isna().sum()
    if missing_text_count > 0:
        print(f"[WARNING] {missing_text_count} utterances in CSV have no matching text ID.")
        # Fill missing elements with empty strings to prevent NaN errors in RoBERTa pipeline
        df['Transcript'] = df['Transcript'].fillna("")
        
    # Overwrite the CSV file
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Metadata updated! Original shape: {original_shape} -> New shape: {df.shape}")
    print(f"[INFO] 'Transcript' column appended successfully.")

if __name__ == "__main__":
    extracted_texts = extract_transcripts(IEMOCAP_ROOT_DIR)
    merge_transcripts_to_csv(extracted_texts, METADATA_CSV_PATH)