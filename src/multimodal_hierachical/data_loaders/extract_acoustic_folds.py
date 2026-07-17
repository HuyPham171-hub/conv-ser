import os
import sys
import gc
import torch
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float32
import librosa
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from dotenv import load_dotenv

# ==========================================
# 1. DYNAMIC PATH RESOLUTION & CONFIGURATION
# ==========================================
ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
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
CHECKPOINTS_BASE_DIR = get_required_path("CHECKPOINTS_DIR")
IEMOCAP_ROOT_DIR = get_required_path("IEMOCAP_ROOT_DIR")

# Define target file pathways
# Note: Đảm bảo file này chứa toàn bộ các Utterance cần thiết (bao gồm cả xxx/oth nếu Stage 3 cần)
METADATA_PATH = DATAFRAMES_DIR / "iemocap_metadata.csv"
CHECKPOINT_DIR = CHECKPOINTS_BASE_DIR / "wav2vec2_sentiment"

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[3] # Navigates up to D:\Resfes\Project\Ser

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Target compute device initialization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define base architecture and checkpoint path configs
BASE_MODEL_ID = "facebook/wav2vec2-base"

# Target directory for the generated acoustic embeddings
OUTPUT_DIR = EMBEDDINGS_DIR / "Acoustic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. AUDIO PATH RECONSTRUCTION
# ==========================================
def build_audio_path(row: pd.Series) -> Path:
    """
    Reconstructs the absolute physical path to the .wav file using metadata.
    Format: IEMOCAP_ROOT_DIR/SessionX/sentences/wav/Dialog_ID/Utterance_ID.wav
    """
    session_folder = f"Session{row['Session']}"
    return IEMOCAP_ROOT_DIR / session_folder / "sentences" / "wav" / str(row['Dialog_ID']) / f"{row['Utterance_ID']}.wav"

# ==========================================
# 3. CORE EXTRACTION PIPELINE
# ==========================================
def main():
    print(f"[INFO] Initializing Fold-Aware Acoustic Extraction...")
    print(f"[INFO] Target Compute Device: {DEVICE}")
    
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"[ERROR] Metadata not found at: {METADATA_PATH}")

    # Load ground truth metadata
    df = pd.read_csv(METADATA_PATH)
    total_utterances = len(df)
    print(f"[INFO] Loaded Metadata: {total_utterances} utterances.")

    # Initialize the static feature extractor
    processor = AutoFeatureExtractor.from_pretrained(BASE_MODEL_ID)

    # ---------------------------------------------------------
    # 5-FOLD CROSS-VALIDATION EXTRACTION LOOP
    # ---------------------------------------------------------
    for fold in range(1, 6):
        print(f"\n{'='*60}")
        print(f"🚀 STARTING EXTRACTION FOR FOLD {fold}")
        print(f"{'='*60}")

        # Trỏ trực tiếp vào thư mục best_model đã được lưu sạch sẽ từ quá trình Train
        best_checkpoint_path = CHECKPOINT_DIR / f"fold_{fold}" / "best_model"
        
        if not best_checkpoint_path.exists():
            print(f"[WARNING] Skipping Fold {fold}. Best model directory not found: {best_checkpoint_path}")
            continue
            
        print(f"[INFO] Injecting optimal weights from: {best_checkpoint_path}")

        # Load the base model architecture and inject fine-tuned weights
        model = Wav2Vec2Model.from_pretrained(best_checkpoint_path).to(DEVICE)
        model.eval() # Freeze computation graph

        # Dictionary to hold the {utterance_id: 768-D tensor} mapping
        fold_embeddings = {}

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
                    outputs = model(input_values)
                
                # 4. Apply Temporal Mean Pooling (Compress sequence_length -> 1D Vector)
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