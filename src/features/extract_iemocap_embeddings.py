import os
from pathlib import Path
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from tqdm import tqdm

# =====================================================================
# 1. ABSOLUTE PATH CONFIGURATION (Matches your specific structure)
# =====================================================================
# Path to the raw IEMOCAP dataset
IEMOCAP_ROOT_DIR = r"d:\Resfes\Project\IEMOCAP_full_release"

# Path to save the extracted .npy file
OUTPUT_DIR = r"d:\Resfes\Project\Ser\data\Embeddings"

# Auto-create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 2. INITIALIZE BASE WAV2VEC2 MODEL
# =====================================================================
MODEL_NAME = r"d:\Resfes\Project\Ser\checkpoints\wav2vec2_sentiment"

# Auto-detect compute device (Prioritize CUDA GPU for speed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using compute device: {device}")

print(f"[INFO] Loading Wav2Vec2 model ({MODEL_NAME})...")
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
model.eval()  # Set to evaluation mode (freeze Dropout/BatchNorm)

# =====================================================================
# 3. FUNCTION TO EXTRACT ACOUSTIC EMBEDDING
# =====================================================================
def extract_acoustic_embedding(file_path):
    """
    Load audio file, resample to 16kHz, extract hidden features,
    and apply Mean Pooling to generate a fixed 768-dimensional embedding vector.
    """
    # Load raw audio waveform
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Standardize sampling rate (Wav2Vec2 strictly requires 16000Hz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Convert tensor to a flat 1D array (Squeeze channel dimension)
    waveform = waveform.squeeze().numpy()
    
    # Preprocess input signal using Transformer's Feature Extractor
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    
    # Extract hidden features without calculating gradients (saves GPU memory)
    with torch.no_grad():
        outputs = model(input_values)
        
    # Get the last hidden state matrix: shape (batch_size=1, time_steps, hidden_size=768)
    last_hidden_state = outputs.last_hidden_state
    
    # MEAN POOLING: Calculate the mean along the time axis (dim=1)
    # This removes sequence length variations, keeping a static feature vector (768,)
    utterance_embedding = torch.mean(last_hidden_state, dim=1).squeeze().cpu().numpy()
    
    return utterance_embedding

# =====================================================================
# 4. DIRECTORY TRAVERSAL & PIPELINE EXECUTION
# =====================================================================
def run_feature_extraction_pipeline():
    embedding_store = {}
    
    print(f"[INFO] Scanning nested directory structure at: {IEMOCAP_ROOT_DIR}")
    
    # Use rglob to scan through all Sessions (1 to 5)
    # Target exactly the .wav files inside the sentences/wav/ structure
    wav_files = list(Path(IEMOCAP_ROOT_DIR).rglob("sentences/wav/*/*.wav"))
    
    total_files = len(wav_files)
    if total_files == 0:
        print("[ERROR] No .wav files found! Please check the IEMOCAP_ROOT_DIR path.")
        return
        
    print(f"[SUCCESS] Mapped and found {total_files} utterance files.")
    print("[INFO] Starting Acoustic Embeddings extraction process...")
    
    # Run loop with a progress bar
    for file_path in tqdm(wav_files, desc="Processing Audio"):
        try:
            # Extract Utterance ID (Filename without .wav extension, e.g., Ses01F_impro01_F000)
            utterance_id = file_path.stem 
            
            # Execute deep learning feature extraction
            embedding = extract_acoustic_embedding(str(file_path))
            
            # Store the key-value pair in the Dictionary memory
            embedding_store[utterance_id] = embedding
            
        except Exception as e:
            # Log warning if an audio file is corrupted or empty
            print(f"\n[WARNING] Error processing file {file_path.name}: {str(e)}")
            continue

    # Serialize the dictionary and save it to the hard drive as a .npy file
    output_file_path = os.path.join(OUTPUT_DIR, "iemocap_static_embeddings_step1.npy")
    np.save(output_file_path, embedding_store)
    
    print("\n" + "="*60)
    print(f"[STEP 1 COMPLETED] Successfully extracted: {len(embedding_store)}/{total_files} samples.")
    print(f"[OUTPUT FILE] Feature Store saved at: {output_file_path}")
    print("="*60)

if __name__ == "__main__":
    run_feature_extraction_pipeline()