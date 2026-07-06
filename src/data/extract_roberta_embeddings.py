import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

def get_required_path(env_name):
    """Fetches a directory path from the environment and expands user variables."""
    value = os.getenv(env_name)
    if not value:
        raise ValueError(f"[ERROR] {env_name} is not set in {ENV_PATH}")
    return Path(value).expanduser()

# Dynamically build paths using environment variables
DATAFRAMES_DIR = get_required_path("DATAFRAMES_DIR")
EMBEDDINGS_DIR = get_required_path("EMBEDDINGS_DIR")

# Define target file pathways matching your dynamic pipeline architecture
METADATA_PATH = DATAFRAMES_DIR / "iemocap_metadata.csv"
OUTPUT_NPY_PATH = EMBEDDINGS_DIR / "iemocap_roberta_embeddings.npy"

# Ensure the output embeddings directory exists safely before running the pipeline
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128  # Standard length, sufficient for short dialogue utterances

requested_device = os.getenv("DEVICE", "auto").lower()
if requested_device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif requested_device == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError("[ERROR] DEVICE=cuda requested but PyTorch cannot see a GPU.")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[INFO] Utilizing compute device: {device}")

# ==========================================
# 2. MODEL INITIALIZATION
# ==========================================
print(f"[INFO] Loading {MODEL_NAME} tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)

# Freeze the model completely (No gradient computation)
model.eval()
for param in model.parameters():
    param.requires_grad = False

# ==========================================
# 3. EXTRACTION PIPELINE
# ==========================================
def extract_lexical_features():
    print(f"[INFO] Loading metadata from: {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
    
    # Ensure Transcript column exists and handle missing values
    if 'Transcript' not in df.columns:
        raise ValueError("Column 'Transcript' not found. Run the transcript extraction script first.")
    
    df['Transcript'] = df['Transcript'].fillna("")
    
    embeddings_dict = {}
    
    print(f"[INFO] Extracting 768-D semantic vectors for {len(df)} utterances...")
    
    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df)):
            utt_id = row['Utterance_ID']
            text = str(row['Transcript'])
            
            # Tokenize text
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Extract the [CLS] token representation (index 0 of the sequence)
            # Shape: (1, seq_len, 768) -> (768,)
            cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            
            embeddings_dict[utt_id] = cls_embedding
            
    # Save to disk
    os.makedirs(os.path.dirname(OUTPUT_NPY_PATH), exist_ok=True)
    np.save(OUTPUT_NPY_PATH, embeddings_dict)
    print(f"[SUCCESS] Saved {len(embeddings_dict)} RoBERTa embeddings to {OUTPUT_NPY_PATH}")

if __name__ == "__main__":
    extract_lexical_features()