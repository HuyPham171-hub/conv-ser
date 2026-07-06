import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import AudioAugmentor
from src.features.extract_iemocap_embeddings import (
    ENV_PATH,
    IEMOCAP_ROOT_DIR,
    EMBEDDINGS_DIR,
    device,
    model,
    processor,
)

load_dotenv(ENV_PATH)

NOISY_EMBEDDINGS_DIR = Path(
    os.getenv("NOISY_EMBEDDINGS_DIR", str(EMBEDDINGS_DIR))
).expanduser()
AWGN_SNR_DB = float(os.getenv("AWGN_SNR_DB", "15"))

NOISY_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
augmentor = AudioAugmentor(sr=16000)


def extract_noisy_acoustic_embedding(file_path):
    """
    Load audio, inject AWGN, then extract a Wav2Vec2 mean-pooled embedding.
    This is intentionally separate from the clean extraction pipeline.
    """
    waveform, _ = librosa.load(file_path, sr=16000)
    noisy_waveform = augmentor.add_awgn(waveform, snr_db=AWGN_SNR_DB)

    inputs = processor(
        noisy_waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)

    return torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()


def run_noisy_feature_extraction_pipeline():
    embedding_store = {}

    print(f"[INFO] Scanning nested directory structure at: {IEMOCAP_ROOT_DIR}")
    print(f"[INFO] Running separate NOISY branch with AWGN SNR: {AWGN_SNR_DB:g} dB")

    wav_files = list(IEMOCAP_ROOT_DIR.rglob("sentences/wav/*/*.wav"))
    total_files = len(wav_files)

    if total_files == 0:
        print("[ERROR] No .wav files found! Please check the IEMOCAP_ROOT_DIR path.")
        return

    print(f"[SUCCESS] Mapped and found {total_files} utterance files.")
    print("[INFO] Starting noisy acoustic embeddings extraction process...")

    for file_path in tqdm(wav_files, desc="Processing Noisy Audio"):
        try:
            utterance_id = file_path.stem
            embedding_store[utterance_id] = extract_noisy_acoustic_embedding(str(file_path))
        except Exception as e:
            print(f"\n[WARNING] Error processing file {file_path.name}: {str(e)}")
            continue

    snr_label = f"{AWGN_SNR_DB:g}".replace(".", "p")
    output_file_path = (
        NOISY_EMBEDDINGS_DIR / f"iemocap_static_embeddings_noisy_awgn_{snr_label}db.npy"
    )
    np.save(output_file_path, embedding_store)

    print("\n" + "=" * 60)
    print(
        f"[NOISY BRANCH COMPLETED] Successfully extracted: "
        f"{len(embedding_store)}/{total_files} samples."
    )
    print(f"[OUTPUT FILE] Noisy feature store saved at: {output_file_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_noisy_feature_extraction_pipeline()
