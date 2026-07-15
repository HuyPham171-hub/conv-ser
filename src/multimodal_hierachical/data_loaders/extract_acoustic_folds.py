import os
import sys
import gc
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import AutoFeatureExtractor, Wav2Vec2Model

# ==========================================
# 1. DYNAMIC PATH RESOLUTION & IMPORTS
# ==========================================
# Fix: Force python to recognize the project root directory
# __file__ is: src/multimodal_hierachical/data_loaders/extract_acoustic_folds.py
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2] # Navigates up to D:\Resfes\Project\Ser

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force path resolution for the inner src directory if needed
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Safe Import using absolute package path resolved via sys.path injection
try:
    from multimodal_hierachical.utils.checkpoint_discovery import get_best_checkpoint
except ImportError:
    try:
        # Fallback for systems where 'src' folder acts as root directly
        from src.multimodal_hierachical.utils.checkpoint_discovery import get_best_checkpoint
    except ImportError as e:
        raise ImportError(
            f"[ERROR] Failed to import 'get_best_checkpoint'. Base exception: {str(e)}. "
            f"Ensure checkpoint_discovery.py contains 'def get_best_checkpoint' "
            f"and is located correctly."
        )

# Target compute device initialization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define absolute/relative data directories based on PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
IEMOCAP_ROOT = DATA_DIR / "IEMOCAP_full_release"
CLEAN_METADATA_PATH = DATA_DIR / "DataFrames" / "iemocap_metadata_clean.csv"

# Define base architecture and checkpoint path configs
BASE_MODEL_ID = "facebook/wav2vec2-base"
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints" / "wav2vec2_stage1"

# Target directory for the generated acoustic embeddings
OUTPUT_DIR = PROJECT_ROOT / "data" / "Embeddings" / "Acoustic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Automatically create the directory if it does not exist

# ==========================================
# 2. AUDIO PATH RECONSTRUCTION
# ==========================================
def build_audio_path(row: pd.Series) -> Path:
    """
    Reconstructs the absolute physical path to the .wav file using metadata.
    Format: IEMOCAP_ROOT/SessionX/sentences/wav/Dialog_ID/Utterance_ID.wav
    """
    session_folder = f"Session{row['Session']}"
    return IEMOCAP_ROOT / session_folder / "sentences" / "wav" / str(row['Dialog_ID']) / f"{row['Utterance_ID']}.wav"

# ==========================================
# 3. CORE EXTRACTION PIPELINE
# ==========================================
def main():
    print(f"[INFO] Initializing Fold-Aware Acoustic Extraction...")
    print(f"[INFO] Target Compute Device: {DEVICE}")
    
    if not CLEAN_METADATA_PATH.exists():
        raise FileNotFoundError(f"[ERROR] Clean metadata not found at: {CLEAN_METADATA_PATH}")

    # Load ground truth metadata
    df = pd.read_csv(CLEAN_METADATA_PATH)
    total_utterances = len(df)
    print(f"[INFO] Loaded Ground Truth Metadata: {total_utterances} clean utterances.")

    # Initialize the static feature extractor (No weights, just audio preprocessing rules)
    processor = AutoFeatureExtractor.from_pretrained(BASE_MODEL_ID)

    # ---------------------------------------------------------
    # 5-FOLD CROSS-VALIDATION EXTRACTION LOOP
    # ---------------------------------------------------------
    for fold in range(1, 6):
        print(f"\n{'='*60}")
        print(f"🚀 STARTING EXTRACTION FOR FOLD {fold}")
        print(f"{'='*60}")

        fold_dir = CHECKPOINTS_ROOT / f"fold_{fold}"
        if not fold_dir.exists():
            print(f"[WARNING] Skipping Fold {fold}. Directory not found: {fold_dir}")
            continue

        # Utilize the Auto-Discovery module to fetch the optimal weights
        best_checkpoint_path = get_best_checkpoint(fold_dir, metric_key="eval_uar", greater_is_better=True)
        if not best_checkpoint_path:
            print(f"[ERROR] Could not resolve best checkpoint for Fold {fold}. Skipping.")
            continue
            
        print(f"[INFO] Injecting optimal weights from: {best_checkpoint_path.name}")

        # Load the base model architecture and inject fine-tuned weights
        # We use Wav2Vec2Model instead of ForSequenceClassification to get raw hidden states
        model = Wav2Vec2Model.from_pretrained(best_checkpoint_path).to(DEVICE)
        model.eval() # Freeze computation graph

        # Dictionary to hold the {utterance_id: 768-D tensor} mapping
        fold_embeddings = {}

        # Iterate through the entire dataset
        # Note: We extract ALL files per fold. In the Multimodal stage, Fold 1 will use 
        # fold_1.pt embeddings to ensure its Test Set was processed by an unbiased model.
        for index, row in tqdm(df.iterrows(), total=total_utterances, desc=f"Processing Fold {fold}"):
            utt_id = str(row['Utterance_ID'])
            audio_path = build_audio_path(row)

            if not audio_path.exists():
                print(f"\n[WARNING] Missing audio file: {audio_path}. Skipping.")
                continue

            try:
                # 1. Load and resample audio strictly to 16kHz
                waveform, _ = librosa.load(audio_path, sr=16000)
                
                # 2. Extract input values
                inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(DEVICE)

                # 3. Forward pass without gradient calculation to save VRAM
                with torch.no_grad():
                    # output_hidden_states=True is default in Wav2Vec2Model, 
                    # last_hidden_state shape: (batch_size=1, sequence_length, hidden_size=768)
                    outputs = model(input_values)
                
                # 4. Apply Temporal Mean Pooling (Compress sequence_length -> 1D Vector)
                # Shape becomes: (768,)
                pooled_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0).cpu()
                
                fold_embeddings[utt_id] = pooled_embedding

            except Exception as e:
                print(f"\n[ERROR] Failed to process {utt_id}: {str(e)}")
                continue

        # Serialize and save the dictionary as a PyTorch .pt file
        output_file = OUTPUT_DIR / f"acoustic_embeddings_fold_{fold}.pt"
        torch.save(fold_embeddings, output_file)
        print(f"[SUCCESS] Fold {fold} embeddings safely serialized to: {output_file}")

        # MLOps Memory Management: Purge the VRAM before loading the next fold's model
        del model
        del fold_embeddings
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"[SUCCESS] FOLD-AWARE ACOUSTIC EXTRACTION PIPELINE COMPLETED.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()