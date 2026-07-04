import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Ser.src.baselines.iemocap_loader import IEMOCAPConversationalDataset
from Ser.src.baselines.bigru_stage3 import ConversationalBiGRU

# 1. Paths
EMBEDDINGS_PATH = r"d:\Resfes\Project\Ser\data\Embeddings\iemocap_static_embeddings_step1.npy"
METADATA_CSV_PATH = r"d:\Resfes\Project\Ser\data\DataFrames\iemocap_metadata.csv"
CHECKPOINT_PATH = r"d:\Resfes\Project\Ser\checkpoints\stage3_bigru\flat_8class_best_model_fold_5.pth" # Pick your best fold

# 2. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset = IEMOCAPConversationalDataset(METADATA_CSV_PATH, EMBEDDINGS_PATH, target_stage=1)
dataset = IEMOCAPConversationalDataset(METADATA_CSV_PATH, EMBEDDINGS_PATH)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False) # MUST BE FALSE to keep order

# model = ConversationalBiGRU(num_classes=3).to(device)
model = ConversationalBiGRU(num_classes=8).to(device) #change num_class to 8 for flat8_classified
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

# 3. Extraction Arrays
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

# 4. FILTER AND SAVE TO DISK
final_hidden_states = np.vstack(all_hidden_states) # Shape: (10039, 512)
all_targets_arr = np.array(all_targets)            # Shape: (10039,)
all_preds_arr = np.array(all_preds)                # Shape: (10039,)

# Create a boolean mask to drop masked labels (-1)
valid_mask = (all_targets_arr != -1)

# Apply the mask to EVERYTHING so they align perfectly
filtered_hidden_states = final_hidden_states[valid_mask]
filtered_targets = all_targets_arr[valid_mask]
filtered_preds = all_preds_arr[valid_mask]
# np.save(r"d:\Resfes\Project\Ser\data\Embeddings\bigru_hidden_states.npy", filtered_hidden_states)
np.save(r"d:\Resfes\Project\Ser\data\Embeddings\flat8_bigru_hidden_states.npy", filtered_hidden_states)


# Save the perfectly aligned Predictions CSV
results_df = pd.read_csv(METADATA_CSV_PATH)
# Apply the exact same mask to the dataframe to drop the -1 rows
results_df = results_df[valid_mask].copy()

results_df['True_Label'] = filtered_targets
results_df['Pred_Label'] = filtered_preds

# results_df.to_csv(r"d:\Resfes\Project\Ser\data\DataFrames\evaluation_results.csv", index=False)
results_df.to_csv(r"d:\Resfes\Project\Ser\data\DataFrames\flat8_evaluation_results.csv", index=False)


print(f"[SUCCESS] Saved {len(filtered_targets)} valid samples.")
print(f"-> Hidden states shape: {filtered_hidden_states.shape}")
print(f"-> DataFrame length: {len(results_df)}")