import os
import sys
import torch
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float32
import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# 1. DYNAMIC PATH RESOLUTION & CONFIGURATION
# ==========================================
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

def get_required_path(env_name):
    """Fetches a directory path from the environment and expands user variables."""
    value = os.getenv(env_name)
    if not value:
        raise ValueError(f"[ERROR] {env_name} is not set in {ENV_PATH}")
    return Path(value).expanduser()

DATAFRAMES_DIR = get_required_path("DATAFRAMES_DIR")
EMBEDDINGS_DIR = get_required_path("EMBEDDINGS_DIR")
IEMOCAP_ROOT_DIR = get_required_path("IEMOCAP_ROOT_DIR")

# Input: Metadata containing all utterances including xxx/oth
METADATA_PATH = DATAFRAMES_DIR / "iemocap_metadata.csv"

# Output: Directory to store individual Dual-Band .pt tensors
OUTPUT_DIR = EMBEDDINGS_DIR / "Visual" / "DualBand_Spectrograms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DSP Configuration
SAMPLE_RATE = 16000
CUTOFF_FREQ = 3000.0  # 3 kHz boundary for dual-band
N_FFT = 512           # Window size
HOP_LENGTH = 256      # Stride (controls the time-axis resolution)
N_MELS = 128          # Frequency-axis resolution (Image Height)

# ==========================================
# 2. SIGNAL PROCESSING FILTERS (DSP)
# ==========================================
def apply_butterworth_filter(data: np.ndarray, cutoff: float, fs: int, btype: str, order: int = 5) -> np.ndarray:
    """
    Applies a zero-phase Butterworth filter to the audio signal.
    btype: 'lowpass' or 'highpass'
    """
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype=btype, analog=False)
    # Use filtfilt for zero-phase filtering to prevent temporal shifting of acoustic events
    filtered_signal = filtfilt(b, a, data)
    return filtered_signal

def extract_log_mel_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Transforms the 1D waveform into a 2D Log Mel Spectrogram.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        n_mels=N_MELS
    )
    # Convert to log scale (Decibels) to match human auditory perception
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Instance Normalization (Standard Score: mean=0, std=1)
    # Crucial for stable ResNet gradient descent
    mean = np.mean(log_mel)
    std = np.std(log_mel)
    if std > 1e-6:
        log_mel = (log_mel - mean) / std
    else:
        log_mel = log_mel - mean
        
    return log_mel

def build_audio_path(row: pd.Series) -> Path:
    session_folder = f"Session{row['Session']}"
    return IEMOCAP_ROOT_DIR / session_folder / "sentences" / "wav" / str(row['Dialog_ID']) / f"{row['Utterance_ID']}.wav"

# ==========================================
# 3. CORE EXTRACTION PIPELINE
# ==========================================
def main():
    print("[INFO] Initializing Dual-Band Spectrogram Extraction Pipeline...")
    
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"[ERROR] Metadata not found at: {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)
    total_utterances = len(df)
    print(f"[INFO] Processing {total_utterances} audio files.")
    print(f"[INFO] DSP Parameters -> SR: {SAMPLE_RATE}, Cutoff: {CUTOFF_FREQ}Hz, Mels: {N_MELS}")

    missing_files = []

    for index, row in tqdm(df.iterrows(), total=total_utterances, desc="Extracting Visual Tensors"):
        utt_id = str(row['Utterance_ID'])
        audio_path = build_audio_path(row)
        target_tensor_path = OUTPUT_DIR / f"{utt_id}.pt"

        # Skip if already extracted (useful for resuming interrupted processes)
        if target_tensor_path.exists():
            continue

        if not audio_path.exists():
            missing_files.append(audio_path)
            continue

        try:
            # 1. Load raw audio
            waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
            
            # Trim leading/trailing silence to focus on actual speech energy
            waveform, _ = librosa.effects.trim(waveform, top_db=30)

            # 2. Apply Dual-Band Filters
            # Channel 0: Low-pass (Prosody, Pitch)
            y_low = apply_butterworth_filter(waveform, CUTOFF_FREQ, SAMPLE_RATE, btype='lowpass')
            # Channel 1: High-pass (Turbulent Noise, Tension)
            y_high = apply_butterworth_filter(waveform, CUTOFF_FREQ, SAMPLE_RATE, btype='highpass')

            # 3. Generate Visual Spectrograms
            spec_low = extract_log_mel_spectrogram(y_low, SAMPLE_RATE)
            spec_high = extract_log_mel_spectrogram(y_high, SAMPLE_RATE)

            # 4. Stack into a 3D Tensor: (Channels, Height, Width) -> (2, N_MELS, Time_Frames)
            # Both specs are guaranteed to have the exact same shape derived from the same waveform
            dual_band_tensor = torch.tensor(np.stack([spec_low, spec_high], axis=0), dtype=torch.float32)

            # 5. Serialize individually to disk
            torch.save(dual_band_tensor, target_tensor_path)

        except Exception as e:
            print(f"\n[ERROR] Failed to process {utt_id}: {str(e)}")
            continue

    print(f"\n{'='*60}")
    if missing_files:
        print(f"[WARNING] Extraction completed with {len(missing_files)} missing audio files.")
    else:
        print(f"[SUCCESS] ALL Dual-Band Spectrograms flawlessly extracted to: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()