import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import torchaudio.transforms as T
import torchaudio.functional as AF
import pandas as pd

class DualBandResNetDataset(Dataset):
    def __init__(self, metadata_df, spectrogram_dir, stage1_outputs, is_train=True):
        """
        Multimodal Dataset for Stage 2 Dual-Band Spectrograms with RAM Caching & SpecAugment.
        Dynamically extracts Delta-Delta as the 3rd channel on the fly.
        """
        self.df = metadata_df.reset_index(drop=True)
        self.spec_dir = spectrogram_dir
        self.stage1_outputs = stage1_outputs
        self.is_train = is_train
        
        # RAM Cache to eliminate disk I/O bottlenecks after Epoch 1
        self.cache = {}

        # Fine-grained Negative Emotion Mapping (Target for Stage 2 classification)
        # Any emotion outside this dictionary will be mapped to -1 (Ignored during Loss calculation)
        self.fine_grained_map = {
            'ang': 0, # Anger
            'sad': 1, # Sadness
            'fru': 2, # Frustration
            'dis': 3, # Disgust
            'fea': 4  # Fear
        }
        
        # Initialize SpecAugment only for training to combat Overfitting
        if self.is_train:
            # Masking up to 15 frequency bins
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
            raise KeyError(f"[ERROR] Missing Stage 1 Output for {utt_id}.")
        p_neg = self.stage1_outputs[utt_id]['p_neg']
        
        # Read from RAM Cache or Load from Disk
        if utt_id in self.cache:
            spectrogram = self.cache[utt_id]
        else:
            spec_path = self.spec_dir / f"{utt_id}.pt"
            spectrogram = torch.load(spec_path, weights_only=True)
            self.cache[utt_id] = spectrogram
        
        # Clone tensor to avoid corrupting the cached original data in RAM
        spectrogram = spectrogram.clone().float()
        
        return {
            "utt_id": utt_id,
            "spectrogram": spectrogram,
            "p_neg": p_neg.detach().clone().float(),
            "label": torch.tensor(label, dtype=torch.long)
        }

def dualband_pad_collate_fn(batch):
    """
    Dynamically pads the Time_Frames (width) of the 3-channel spectrograms to the maximum length in the batch.
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