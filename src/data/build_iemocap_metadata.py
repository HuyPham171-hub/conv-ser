import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# =====================================================================
# 1. PATH CONFIGURATION (Matches your specific structure)
# =====================================================================
IEMOCAP_ROOT_DIR = r"d:\Resfes\Project\IEMOCAP_full_release"
OUTPUT_CSV_PATH = r"d:\Resfes\Project\Ser\data\DataFrames\iemocap_metadata.csv"

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

# =====================================================================
# 2. MAPPING LOGIC (Based on your 2-Stage Hierarchical Specification)
# =====================================================================
def map_stage1_sentiment(emotion):
    """ Coarse Sentiment Classification (3 Classes) """
    negatives = ['ang', 'sad', 'fea', 'dis', 'fru']
    neutrals = ['neu', 'cal']
    positives = ['hap', 'sur', 'exc']
    
    if emotion in negatives: return 0
    elif emotion in neutrals: return 1
    elif emotion in positives: return 2
    else: return -1 # Undefined labels (xxx, oth) will be masked during training

def map_stage2_fine_grained(emotion):
    """ Fine-Grained Negative Classification (5 Classes) """
    mapping = {'ang': 0, 'sad': 1, 'fea': 2, 'dis': 3, 'fru': 4}
    # Only map if it belongs to the Negative group, otherwise return -1
    return mapping.get(emotion, -1)

# =====================================================================
# 3. TEXT FILE PARSING ALGORITHM
# =====================================================================
def parse_iemocap_evaluations():
    print(f"[INFO] Scanning EmoEvaluation .txt files at: {IEMOCAP_ROOT_DIR}")
    
    # Find all .txt files inside the EmoEvaluation folder
    # Expected standard file format: Ses01F_impro01.txt
    txt_files = list(Path(IEMOCAP_ROOT_DIR).rglob("dialog/EmoEvaluation/*.txt"))
    
    # Ignore actor self-evaluation files if any exist (e.g., hidden files starting with a dot)
    txt_files = [f for f in txt_files if not f.name.startswith('.')]
    
    parsed_data = []
    
    # Regex pattern to capture the exact line containing the primary emotion label
    # Standard IEMOCAP format: [00:06.2900 - 00:08.2300]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]
    pattern = re.compile(r'^\[(.*?) - (.*?)\]\t(.*?)\t(.*?)\t\[(.*?),\s*(.*?),\s*(.*?)\]')
    
    for file_path in tqdm(txt_files, desc="Parsing Text Files"):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
            for line in lines:
                match = pattern.match(line)
                if match:
                    start_time, end_time, utt_id, raw_emotion, val, aro, dom = match.groups()
                    
                    # 1. Extract Dialogue ID
                    # Split string: Ses01F_impro01_F000 -> Ses01F_impro01
                    dialog_id = "_".join(utt_id.split("_")[:-1])
                    
                    # 2. Extract Turn Order Sequence
                    # Convert to integer for chronological sorting (e.g., Ses01F_impro01_F000 -> F000 -> 0)
                    turn_order = int(utt_id.split("_")[-1][1:])
                    
                    # 3. Extract Session ID (Used for LOSO cross-validation setup)
                    # Ses01F_impro01 -> The 4th character represents the session index number '1'
                    session_id = int(utt_id[3])
                    
                    # 4. Classify Scenario Setup (Improvised vs Scripted)
                    is_impro = 1 if 'impro' in dialog_id else 0
                    
                    # Apply custom hierarchical mapping logic
                    stage1_label = map_stage1_sentiment(raw_emotion)
                    stage2_label = map_stage2_fine_grained(raw_emotion)
                    
                    parsed_data.append({
                        'Utterance_ID': utt_id,
                        'Dialog_ID': dialog_id,
                        'Session': session_id,
                        'Turn_Order': turn_order,
                        'Is_Impro': is_impro,
                        'Raw_Emotion': raw_emotion,
                        'Stage1_Label': stage1_label,
                        'Stage2_Label': stage2_label,
                        'Valence': float(val),
                        'Arousal': float(aro),
                        'Dominance': float(dom)
                    })
                    
    # Convert list to DataFrame and sort systematically
    df = pd.DataFrame(parsed_data)
    
    # Sort by Session -> Dialogue ID -> Turn Order to build an absolute chronological timeline
    df = df.sort_values(by=['Session', 'Dialog_ID', 'Turn_Order']).reset_index(drop=True)
    
    # Export structured output to CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print("\n" + "="*60)
    print(f"[SUCCESS] Parsed and stored a total of {len(df)} utterances.")
    print(f"[STATISTICS] Sample distribution by Session:\n{df['Session'].value_counts().sort_index()}")
    print(f"[OUTPUT] Metadata saved at: {OUTPUT_CSV_PATH}")
    print("="*60)

if __name__ == "__main__":
    parse_iemocap_evaluations()