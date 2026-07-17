import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import torchaudio.transforms as T
import pandas as pd

class DualBandResNetDataset(Dataset):
    def __init__(self, metadata_df, spectrogram_dir, stage1_outputs, is_train=True):
        """
        Multimodal Dataset for Stage 2 Dual-Band Spectrograms with RAM Caching & SpecAugment.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.spec_dir = spectrogram_dir
        self.stage1_outputs = stage1_outputs
        self.is_train = is_train
        
        # 1. RAM Cache to eliminate disk I/O bottlenecks after Epoch 1
        self.cache = {}

        # Fine-grained Negative Emotion Mapping (Target for Stage 2)
        self.fine_grained_map = {
            'ang': 0, # Anger
            'sad': 1, # Sadness
            'fru': 2, # Frustration
            'dis': 3, # Disgust
            'fea': 4  # Fear
        }
        
        # 2. Initialize SpecAugment only for training to combat Overfitting
        if self.is_train:
            # Masking up to 15 frequency bins (out of typical 128 mel bins)
            self.freq_masking = T.FrequencyMasking(freq_mask_param=15)
            # Masking up to 30 time frames
            self.time_masking = T.TimeMasking(time_mask_param=30)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        utt_id = str(row['Utterance_ID'])
        
        # Resolve Fine-Grained Label
        raw_emotion = str(row.get('Raw_Emotion', 'xxx')).lower()
        label = self.fine_grained_map.get(raw_emotion, -1)
        
        # Fetch P_neg from Stage 1 (Required for Soft Gating)
        if utt_id not in self.stage1_outputs:
            raise KeyError(f"[ERROR] Missing Stage 1 Output for {utt_id}. Run Stage 1 completely.")
        p_neg = self.stage1_outputs[utt_id]['p_neg']
        
        # 3. Read from RAM Cache or Load from Disk
        if utt_id in self.cache:
            spectrogram = self.cache[utt_id]
        else:
            spec_path = self.spec_dir / f"{utt_id}.pt"
            if not spec_path.exists():
                raise FileNotFoundError(f"[ERROR] Missing Dual-Band tensor: {spec_path}")
            spectrogram = torch.load(spec_path, weights_only=True)
            self.cache[utt_id] = spectrogram
        
        # CLONE the tensor before augmentation so we don't corrupt the cached original!
        spectrogram = spectrogram.clone().float()
        
        # 4. Apply SpecAugment on the fly (Training only)
        if self.is_train:
            spectrogram = self.freq_masking(spectrogram)
            spectrogram = self.time_masking(spectrogram)
        
        return {
            "utt_id": utt_id,
            "spectrogram": spectrogram,
            "p_neg": p_neg.float(),
            "label": torch.tensor(label, dtype=torch.long)
        }

def dualband_pad_collate_fn(batch):
    """
    Dynamically pads the Time_Frames (width) of the spectrograms to the maximum length in the batch.
    """
    utt_ids = [item["utt_id"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    p_negs = torch.stack([item["p_neg"] for item in batch])
    
    spectrograms = [item["spectrogram"] for item in batch]
    
    # Find the maximum time dimension (dim=2) in this specific batch
    max_time_frames = max([spec.shape[2] for spec in spectrograms])
    
    padded_spectrograms = []
    for spec in spectrograms:
        # F.pad format for 3D tensor (Channels, Height, Width):
        # We only pad the right side of the time axis (Width)
        pad_amount = max_time_frames - spec.shape[2]
        padded_spec = F.pad(spec, (0, pad_amount, 0, 0), mode='constant', value=0.0)
        padded_spectrograms.append(padded_spec)
        
    padded_spectrograms = torch.stack(padded_spectrograms)
    
    return {
        "utt_ids": utt_ids,
        "spectrograms": padded_spectrograms,
        "p_negs": p_negs,
        "labels": labels
    }