"""
IEMOCAP Conversational DataLoader Module
Provides sequence structuring and PyTorch Dataset class for Contextual Emotion Tracking (Goal 3).
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import traceback

# modes: 'flat8': for 8 emotions; 'stage1': for 3 sentiments; 'stage2': for 4 negative emotions
def build_conversational_sequences(metadata_path, window_size=3, mode='flat8'):
    """
    Reads metadata, groups utterances by Dialog_ID, and constructs 
    sliding window sequences with temporal zero-padding for initial turns.
    
    Args:
        metadata_path (str): Path to the parsed iemocap_metadata.csv.
        window_size (int): Size of the contextual sliding window (Default: 3).
        
    Returns:
        tuple: (sequences, targets_stage1, targets_stage2)
    """
    print("[INFO] Loading and sorting metadata...", flush=True) # flush=True forces immediate terminal output
    
    # ---------------------------------------------------------
    # AGGRESSIVE DEBUG BLOCK START
    # ---------------------------------------------------------
    try:
        print(f"  -> [DEBUG] Attempting to read CSV from: {metadata_path}", flush=True)
        df = pd.read_csv(metadata_path, engine='python')
        print(f"  -> [DEBUG] CSV loaded successfully! Shape: {df.shape}", flush=True)
        
        print("  -> [DEBUG] Attempting to sort values...", flush=True)
        df = df.sort_values(by=['Session', 'Dialog_ID', 'Turn_Order']).reset_index(drop=True)
        print("  -> [DEBUG] Sorting complete!", flush=True)
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR DURING PANDAS OPERATION]: {e}", flush=True)
        traceback.print_exc()
        import sys
        sys.exit(1)
    # ---------------------------------------------------------
    # AGGRESSIVE DEBUG BLOCK END
    # ---------------------------------------------------------
    
    # Ensure chronological order (Session -> Dialog -> Turn) to maintain temporal integrity
    df = df.sort_values(by=['Session', 'Dialog_ID', 'Turn_Order']).reset_index(drop=True)
    
    # ---------------------------------------------------------
    # DYNAMIC EMOTION TO INDEX MAPPING BASED ON EXPERIMENT MODE
    # 'dis' (Disgust) is dropped due to severe class imbalance (only 2 samples).
    # Unmapped labels like 'xxx' or 'oth' default to -1 (ignored during training).
    # ---------------------------------------------------------
    if mode == 'flat8':
        # 7 Target Classes: Disgust removed, Happiness and Excitement kept separate
        EMOTION_TO_IDX = {
            'neu': 0,
            'hap': 1,
            'exc': 2, 
            'sad': 3,
            'ang': 4,
            'fru': 5,
            'fea': 6,
            'sur': 7
        }
    elif mode == 'stage1':
        # 3 Macro-Sentiment Classes (0: Positive, 1: Neutral/Other, 2: Negative)
        EMOTION_TO_IDX = {
            'hap': 0, 'exc': 0,
            'neu': 1, 'sur': 1,
            'ang': 2, 'sad': 2, 'fea': 2, 'fru': 2
        }
    elif mode == 'stage2':
        # 4 Fine-grained Negative Classes (Disgust removed)
        EMOTION_TO_IDX = {
            'ang': 0,
            'sad': 1,
            'fea': 2,
            'fru': 3
        }
    else:
        raise ValueError("Invalid mode configuration. Choose among: 'flat8', 'stage1', 'stage2'")
        
    sequences = []
    targets = []
    
    print(f"[INFO] Applying sliding window (N={window_size}) for MODE: {mode.upper()}...")
    
    
    # Group by independent conversational sessions to avoid context leakage across dialogues
    for dialog_id, group in df.groupby('Dialog_ID'):
        utterances = group['Utterance_ID'].tolist()
        
        # Dynamically find the emotion column (handles both 'Raw_Emotion' and 'Emotion' namings)
        if 'Raw_Emotion' in group.columns:
            emotions = group['Raw_Emotion'].tolist()
        elif 'Emotion' in group.columns:
            emotions = group['Emotion'].tolist()
        else:
            raise ValueError("Could not find 'Raw_Emotion' or 'Emotion' column in metadata!")
            
        num_turns = len(utterances)
        
        for t in range(num_turns):
            current_utt = utterances[t]
            
            # Map emotion to integer ID
            raw_emo = str(emotions[t]).lower().strip()
            lbl = EMOTION_TO_IDX.get(raw_emo, -1)
            
            # ---------------------------------------------------
            # TEMPORAL ZERO-PADDING LOGIC (For N=3)
            # ---------------------------------------------------
            if t == 0:
                # First turn: No historical context. Pad with two None tokens.
                seq = [None, None, current_utt]
            elif t == 1:
                # Second turn: Only one historical utterance available. Pad with one None token.
                seq = [None, utterances[t-1], current_utt]
            else:
                # Third turn onwards: Full historical context available [U_{t-2}, U_{t-1}, U_t]
                seq = [utterances[t-2], utterances[t-1], current_utt]
                
            sequences.append(seq)
            targets.append(lbl)
            
    print(f"[SUCCESS] Generated {len(sequences)} sequence windows.")
    return sequences, targets


class IEMOCAPConversationalDataset(Dataset):
    """
    Custom PyTorch Dataset for Conversational Emotion Tracking.
    Dynamically maps Utterance IDs to their corresponding 768-D acoustic embeddings.
    """
    def __init__(self, metadata_path, embeddings_npy_path, mode='flat8'):
        """
        Initializes the Dataset by loading the embedding dictionary and building sequence structures.
        
        Args:
            metadata_path (str): Path to iemocap_metadata.csv.
            embeddings_npy_path (str): Path to the static embeddings dictionary (.npy).
        """
        super().__init__()
        
        print("[INFO] Loading 768-D acoustic embeddings into memory...")
        self.embeddings_dict = np.load(embeddings_npy_path, allow_pickle=True).item()
        
        # Build logical sequences and flat 8-class labels
        self.sequences, self.targets = build_conversational_sequences(metadata_path, mode=mode)
        
        # Define a zero-vector for padding missing historical contexts
        # Shape: (768,) matching the Wav2Vec2 output dimension
        self.zero_padding_vector = np.zeros(768, dtype=np.float32)

    def __len__(self):
        """Returns the total number of sliding window sequences."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves a single sequence batch (X) and its target label (y).
        """
        # Retrieve the sequence of Utterance IDs (length 3)
        seq_utt_ids = self.sequences[idx]
        
        # Target label is directly fetched from the flat 8-class list
        target_label = self.targets[idx]
        
        window_embeddings = []
        
        # Iterate through U_{t-2}, U_{t-1}, U_t
        for utt_id in seq_utt_ids:
            if utt_id is None:
                # Apply Zero-Padding for initial conversation turns
                window_embeddings.append(self.zero_padding_vector)
            else:
                # Fetch actual acoustic embedding. Fallback to zero-vector if ID is missing.
                embedding = self.embeddings_dict.get(utt_id, self.zero_padding_vector)
                window_embeddings.append(embedding)
                
        # Convert the list of 3 vectors into a PyTorch Tensor
        # Output shape: (3, 768)
        X_tensor = torch.tensor(np.array(window_embeddings), dtype=torch.float32)
        y_tensor = torch.tensor(target_label, dtype=torch.long)
        
        return X_tensor, y_tensor