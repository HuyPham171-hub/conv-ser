import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

class DialogueSequenceDataset(Dataset):
    def __init__(self, metadata_df, stage1_outputs, stage2_outputs):
        """
        Dataset for Stage 3: Temporal Contextualization via Sequence Modeling.
        Groups isolated utterances into chronologically ordered dialogues, 
        concatenates Stage 1 (Audio/Text) and Stage 2 (Gated Visual) features,
        and applies zero-tensor masking for non-linguistic utterances.
        """
        self.df = metadata_df.copy()
        
        # 1. Dynamically extract Dialog_ID from Utterance_ID (e.g., 'Ses01F_impro01_M000' -> 'Ses01F_impro01')
        if 'Dialog_ID' not in self.df.columns:
            self.df['Dialog_ID'] = self.df['Utterance_ID'].apply(lambda x: "_".join(x.split('_')[:-1]))
        
        # 2. Map fine-grained emotions
        self.fine_grained_map = {'ang': 0, 'sad': 1, 'fru': 2, 'dis': 3, 'fea': 4}
        self.df['label'] = self.df['Raw_Emotion'].astype(str).str.lower().map(self.fine_grained_map).fillna(-1).astype(int)
        
        self.stage1_outputs = stage1_outputs
        self.stage2_outputs = stage2_outputs
        
        # 3. Group by Dialog_ID to form sequences
        self.dialogues = []
        grouped = self.df.groupby('Dialog_ID')
        
        for dialog_id, group in grouped:
            # Strictly sort by Utterance_ID to maintain chronological order
            sorted_group = group.sort_values(by='Utterance_ID')
            
            sequence_data = []
            for _, row in sorted_group.iterrows():
                utt_id = row['Utterance_ID']
                label = row['label']
                
                # Verify existence in both early-stage checkpoints
                if utt_id not in self.stage1_outputs or utt_id not in self.stage2_outputs:
                    continue 
                
                v_stage1 = self.stage1_outputs[utt_id]['v_stage1'].squeeze() # Adjust key based on your Stage 1 dict structure
                v_gated = self.stage2_outputs[utt_id]['v_gated'].squeeze()
                
                # Concatenate heterogeneous features: V_concat = [V_stage1, V_gated]
                v_concat = torch.cat([v_stage1, v_gated], dim=-1)
                
                # Apply Zero-Tensor for non-linguistic/garbage utterances (xxx, oth)
                # This explicitly forces the GRU to treat this timestep as a contextual bridge, not a feature source
                if label == -1:
                    v_concat = torch.zeros_like(v_concat)
                    valid_mask = 0.0
                else:
                    valid_mask = 1.0
                    
                sequence_data.append({
                    'utt_id': utt_id,
                    'v_concat': v_concat,
                    'label': label,
                    'valid_mask': valid_mask
                })
            
            # Only append dialogues that have valid extracted sequences
            if len(sequence_data) > 0:
                self.dialogues.append({
                    'dialog_id': dialog_id,
                    'sequence': sequence_data
                })

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialog = self.dialogues[idx]
        
        v_concats = torch.stack([item['v_concat'] for item in dialog['sequence']])
        labels = torch.tensor([item['label'] for item in dialog['sequence']], dtype=torch.long)
        valid_masks = torch.tensor([item['valid_mask'] for item in dialog['sequence']], dtype=torch.float32)
        utt_ids = [item['utt_id'] for item in dialog['sequence']]
        
        return {
            'dialog_id': dialog['dialog_id'],
            'v_concats': v_concats,          # Shape: (Sequence_Length, Feature_Dim)
            'labels': labels,                # Shape: (Sequence_Length,)
            'valid_masks': valid_masks,      # Shape: (Sequence_Length,)
            'utt_ids': utt_ids,              # List of strings
            'seq_len': len(dialog['sequence'])
        }

def sequence_collate_fn(batch):
    """
    Collate function to pad variable-length dialogue sequences.
    It sorts the batch by sequence length in descending order, 
    which is an optimized format for PyTorch's pack_padded_sequence.
    """
    # 1. Sort the batch by seq_len descending
    batch = sorted(batch, key=lambda x: x['seq_len'], reverse=True)
    
    dialog_ids = [item['dialog_id'] for item in batch]
    utt_ids_list = [item['utt_ids'] for item in batch]
    seq_lengths = torch.tensor([item['seq_len'] for item in batch], dtype=torch.long)
    
    # 2. Extract sequences
    v_concats_list = [item['v_concats'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    valid_masks_list = [item['valid_masks'] for item in batch]
    
    # 3. Apply Zero-Padding
    # batch_first=True returns shape (Batch_Size, Max_Seq_Len, Feature_Dim)
    padded_v_concats = pad_sequence(v_concats_list, batch_first=True, padding_value=0.0)
    
    # Pad labels with -1 (ignore_index)
    padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=-1)
    
    # Pad masks with 0.0 (ignored in loss computation)
    padded_valid_masks = pad_sequence(valid_masks_list, batch_first=True, padding_value=0.0)
    
    return {
        'dialog_ids': dialog_ids,
        'v_concats': padded_v_concats,
        'labels': padded_labels,
        'valid_masks': padded_valid_masks,
        'seq_lengths': seq_lengths,
        'utt_ids_list': utt_ids_list
    }