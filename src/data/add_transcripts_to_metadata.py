import os
import pandas as pd
import glob

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
IEMOCAP_ROOT = r"D:\Resfes\Project\IEMOCAP_full_release"
METADATA_CSV_PATH = r"d:\Resfes\Project\Ser\data\DataFrames\iemocap_metadata.csv"

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
        txt_files = glob.glob(os.path.join(transcriptions_dir, '*.txt'))
        
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
    extracted_texts = extract_transcripts(IEMOCAP_ROOT)
    merge_transcripts_to_csv(extracted_texts, METADATA_CSV_PATH)