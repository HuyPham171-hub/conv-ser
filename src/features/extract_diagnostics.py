"""
Diagnostic Extraction Script
Extracts Bi-GRU hidden states and predictions across 'flat8', 'stage1', and 'stage2' modes.
Saves aligned results to a CSV file and compressed hidden states to a NumPy array for UMAP/t-SNE visualization.
"""

import torch
import numpy as np
import argparse
import pandas as pd
from torch.utils.data import DataLoader
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.baselines.iemocap_loader import IEMOCAPConversationalDataset
from src.baselines.bigru_stage3 import ConversationalBiGRU

# =====================================================================
# 1. PARAMETER PARSING & PATH CONFIGURATIONS
# =====================================================================
parser = argparse.ArgumentParser(description="Extract Diagnostics for Conversational Bi-GRU")
parser.add_argument('--mode', type=str, choices=['flat8', 'stage1', 'stage2'], required=True, help="Experiment track")
parser.add_argument('--fold', type=int, default=5, help="The specific fold checkpoint to load (1-5)")
args = parser.parse_args()

# set the number of classes based on the tracking mode
if args.mode == 'flat8':
    NUM_CLASSES = 8
elif args.mode == 'stage1':
    NUM_CLASSES = 3
elif args.mode == 'stage2':
    NUM_CLASSES = 4

EMBEDDINGS_PATH = r"d:\Resfes\Project\Ser\data\Embeddings\iemocap_wav2vec2_embeddings.npy"
METADATA_CSV_PATH = r"d:\Resfes\Project\Ser\data\DataFrames\iemocap_metadata.csv"
CHECKPOINT_PATH = f"d:\\Resfes\\Project\\Ser\\checkpoints\\{args.mode}_bigru\\{args.mode}_best_model_fold_{args.fold}.pth"

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Target model checkpoint does not exist at: {CHECKPOINT_PATH}")

# =====================================================================
# 2. DATASET & MODEL INITIALIZATION
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Initializing Evaluation Framework on: {device}")
print(f"[INFO] Track Mode: {args.mode.upper()} | Target Checkpoint: Fold {args.fold}")


dataset = IEMOCAPConversationalDataset(METADATA_CSV_PATH, EMBEDDINGS_PATH, mode=args.mode)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False) # MUST BE FALSE to keep order

# model = ConversationalBiGRU(num_classes=3).to(device)
model = ConversationalBiGRU(num_classes=NUM_CLASSES).to(device) #change num_class to 8 for flat8_classified
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# =====================================================================
# 3. EXTRACTION LOOP
# =====================================================================
all_preds = []
all_targets = []
all_hidden_states = []

print("[INFO] Extracting hidden states and predictions...")
with torch.no_grad():
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        
        # Call forward with return_hidden=True
        logits, hidden = model(batch_X, return_hidden=True)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())
        all_hidden_states.append(hidden.cpu().numpy())

final_hidden_states = np.vstack(all_hidden_states) # Shape: (10039, 512)
all_targets_arr = np.array(all_targets)            # Shape: (10039,)
all_preds_arr = np.array(all_preds)                # Shape: (10039,)

# =====================================================================
# 4. TEMPORAL ALIGNMENT & FILTERING MASK
# =====================================================================
# Create a boolean mask to drop masked labels (-1)
valid_mask = (all_targets_arr != -1)

# Apply the mask to EVERYTHING so they align perfectly
filtered_hidden_states = final_hidden_states[valid_mask]
filtered_targets = all_targets_arr[valid_mask]
filtered_preds = all_preds_arr[valid_mask]

# =====================================================================
# 5. EXPORT AND SERIALIZATION TO DISK
# =====================================================================
# Define dynamic output pathways based on the argument track mode
npy_output_path = f"d:\\Resfes\Project\\Ser\\data\\Embeddings\\{args.mode}_bigru_hidden_states.npy"
csv_output_path = f"d:\\Resfes\\Project\\Ser\\data\\DataFrames\\{args.mode}_evaluation_results.csv"

# Save compressed hidden states for t-SNE or UMAP analysis
np.save(npy_output_path, filtered_hidden_states)

# Read base metadata, apply the exact same alignment mask, and inject classifications
results_df = pd.read_csv(METADATA_CSV_PATH)
# Critical Check: Ensure the mask length matches metadata rows. 
# If metadata holds unmasked records (e.g. stage2 evaluation filtering), handle it via dataset indices.
if len(results_df) != len(valid_mask):
    print("[INFO] Metadata mismatch detected. Fetching exact indices mapped via the data loader...")
    # Map back to the active indices processed within the sequential loop
    valid_indices = [seq[-1] for idx, seq in enumerate(dataset.sequences) if valid_mask[idx]]
    results_df = results_df[results_df['Utterance_ID'].isin(valid_indices)].copy()
else:
    results_df = results_df[valid_mask].copy()

results_df['True_Label'] = filtered_targets
results_df['Pred_Label'] = filtered_preds

results_df.to_csv(csv_output_path, index=False)

print(f"\n{'=' * 60}")
print(f"🏆 DIAGNOSTIC EXTRACTION COMPLETED FOR: {args.mode.upper()}")
print(f"\n{'=' * 60}")
print(f"[SUCCESS] Extracted and aligned {len(results_df)} valid records.")
print(f"-> Saved Hidden States (.npy) : {npy_output_path} (Shape: {filtered_hidden_states.shape})")
print(f"-> Saved Evaluation Report (.csv): {csv_output_path}")
print(f"\n{'=' * 60}")