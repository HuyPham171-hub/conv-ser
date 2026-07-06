import os
from pathlib import Path
import torch
import numpy as np
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from dotenv import load_dotenv

# =====================================================================
# 1. PATH CONFIGURATION (USING .ENV)
# =====================================================================
# Resolve the path to the .env file located at the project root
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

def get_required_path(env_name):
    """Fetches a directory path from the environment and expands user variables."""
    value = os.getenv(env_name)
    if not value:
        raise ValueError(f"[ERROR] {env_name} is not set in {ENV_PATH}")
    return Path(value).expanduser()

# Load paths from environment variables
IEMOCAP_ROOT_DIR = get_required_path("IEMOCAP_ROOT_DIR")
EMBEDDINGS_DIR = get_required_path("EMBEDDINGS_DIR")

if not IEMOCAP_ROOT_DIR.exists():
    raise FileNotFoundError(f"[ERROR] IEMOCAP_ROOT_DIR does not exist: {IEMOCAP_ROOT_DIR}")

# Ensure the output directory exists
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 2. INITIALIZE BASE WAV2VEC2 MODEL (VANILLA)
# =====================================================================
# Target the official vanilla base model directly from Hugging Face Hub
MODEL_NAME = "facebook/wav2vec2-base"
requested_device = os.getenv("DEVICE", "auto").lower()

if requested_device == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif requested_device == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "DEVICE=cuda was requested, but PyTorch cannot see a CUDA GPU. "
            "Install a CUDA-enabled PyTorch build, then run this script again."
        )
    device = torch.device("cuda")
elif requested_device == "cpu":
    device = torch.device("cpu")
else:
    raise ValueError("DEVICE must be one of: auto, cuda, cpu")

print(f"[INFO] Using compute device: {device}")
print(f"[INFO] Downloading/Loading Vanilla model: {MODEL_NAME}...")

# Initialize the feature extractor and the base architecture directly from Hugging Face
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)

if device.type == "cuda":
    print(f"[INFO] GPU detected: {torch.cuda.get_device_name(0)}")

# Freeze model parameters for feature extraction
model.eval()

# =====================================================================
# 3. FUNCTION TO EXTRACT ACOUSTIC EMBEDDING
# =====================================================================
def extract_acoustic_embedding(file_path):
    """
    Load an audio file using Librosa, automatically resample to 16kHz,
    extract hidden state features using Wav2Vec2, and apply Mean Pooling 
    to generate a fixed-size 768-dimensional embedding vector.
    """
    # Load audio and automatically resample to 16000Hz (Wav2Vec2 strict requirement)
    waveform, _ = librosa.load(file_path, sr=16000)
    
    # Preprocess the 1D signal using Transformer's Feature Extractor
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    
    # Extract hidden features without calculating gradients (reduces VRAM/GPU usage)
    with torch.no_grad():
        outputs = model(input_values)
        
    # Retrieve the final hidden state matrix: (batch_size=1, time_steps, hidden_size=768)
    last_hidden_state = outputs.last_hidden_state
    
    # MEAN POOLING: Compute the average across the temporal dimension (dim=1)
    utterance_embedding = torch.mean(last_hidden_state, dim=1).squeeze().cpu().numpy()
    
    return utterance_embedding

# =====================================================================
# 4. DIRECTORY TRAVERSAL & PIPELINE EXECUTION
# =====================================================================
def run_feature_extraction_pipeline():
    embedding_store = {}
    
    print(f"[INFO] Scanning nested directory structure at: {IEMOCAP_ROOT_DIR}")
    
    # Target exactly the .wav files inside the sentences/wav/ structure
    wav_files = list(IEMOCAP_ROOT_DIR.rglob("sentences/wav/*/*.wav"))
    
    total_files = len(wav_files)
    if total_files == 0:
        print("[ERROR] No .wav files found! Please check the IEMOCAP_ROOT_DIR path.")
        return
        
    print(f"[SUCCESS] Mapped and found {total_files} utterance files.")
    print("[INFO] Starting Acoustic Embeddings extraction process...")
    
    # Run loop with a progress bar
    for file_path in tqdm(wav_files, desc="Processing Audio"):
        try:
            # Extract Utterance ID (Filename without .wav extension)
            utterance_id = file_path.stem 
            
            # Execute deep learning feature extraction
            embedding = extract_acoustic_embedding(str(file_path))
            
            # Store the key-value pair in the Dictionary memory
            embedding_store[utterance_id] = embedding
            
        except Exception as e:
            print(f"\n[WARNING] Error processing file {file_path.name}: {str(e)}")
            continue

    # Serialize the dictionary and save it to the hard drive as a .npy file
    output_file_path = EMBEDDINGS_DIR / "iemocap_vanilla_embeddings.npy"
    np.save(output_file_path, embedding_store)
    
    print("\n" + "="*60)
    print(f"[STEP 1 COMPLETED] Successfully extracted: {len(embedding_store)}/{total_files} samples.")
    print(f"[OUTPUT FILE] Feature Store saved at: {output_file_path}")
    print("="*60)

if __name__ == "__main__":
    run_feature_extraction_pipeline()