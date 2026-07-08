import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class IEMOCAPMultimodalDataset(Dataset):
    """
    Dedicated Dataset for Stage 1: Multimodal Sentiment Classification.
    Loads and synchronizes static Wav2Vec2 (Audio) and RoBERTa (Text) vectors.
    """
    def __init__(self, metadata_path, audio_emb_path, text_emb_path):
        super().__init__()
        print("[INFO] Loading Metadata and 768-D Embeddings (Audio + Text)...")
        self.df = pd.read_csv(metadata_path)
        
        # Load pre-extracted static embeddings
        self.audio_dict = np.load(audio_emb_path, allow_pickle=True).item()
        self.text_dict = np.load(text_emb_path, allow_pickle=True).item()
        
        # Mapping 7 emotions to 3 Macro-Sentiments
        sentiment_map = {
            'hap': 0, 'exc': 0,  # Positive
            'neu': 1, 'sur': 1,  # Neutral/Ambiguous
            'ang': 2, 'sad': 2, 'fea': 2, 'fru': 2  # Negative
        }
        
        self.samples = []
        missing_count = 0
        
        for _, row in self.df.iterrows():
            utt_id = row['Utterance_ID']
            raw_emo = str(row['Raw_Emotion']).lower().strip()
            
            # Only include samples belonging to the 7 defined labels (exclude 'xxx', 'dis', 'oth')
            if raw_emo in sentiment_map:
                if utt_id in self.audio_dict and utt_id in self.text_dict:
                    self.samples.append({
                        'utt_id': utt_id,
                        'label': sentiment_map[raw_emo]
                    })
                else:
                    missing_count += 1
                    
        print(f"[SUCCESS] Multimodal Dataset Ready. Total valid samples: {len(self.samples)}")
        if missing_count > 0:
            print(f"[WARNING] Skipped {missing_count} utterances due to missing embedding vectors.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        utt_id = sample['utt_id']
        label = sample['label']
        
        audio_emb = self.audio_dict[utt_id]
        text_emb = self.text_dict[utt_id]
        
        # Convert to float32 Tensor for compatibility with nn.MultiheadAttention
        tensor_audio = torch.tensor(audio_emb, dtype=torch.float32)
        tensor_text = torch.tensor(text_emb, dtype=torch.float32)
        tensor_label = torch.tensor(label, dtype=torch.long)
        
        return tensor_audio, tensor_text, tensor_label