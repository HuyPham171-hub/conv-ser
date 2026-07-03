"""
Conversational Bi-GRU Training Module
Implements Leave-One-Session-Out (LOSO) Cross-Validation and temporal masking.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score

# Add project root to sys.path to import local modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import custom dataset and model architectures
from src.data.iemocap_loader import IEMOCAPConversationalDataset
from src.models.bigru_stage3 import ConversationalBiGRU

# =====================================================================
# 1. HYPERPARAMETERS & PATHS CONFIGURATION
# =====================================================================
EMBEDDINGS_PATH = r"d:\Resfes\Project\Ser\data\Embeddings\iemocap_static_embeddings_step1.npy"
METADATA_CSV_PATH = r"d:\Resfes\Project\Ser\data\DataFrames\iemocap_metadata.csv"

parser = argparse.ArgumentParser(description="Train Conversational Bi-GRU")
parser.add_argument('--mode', type=str, choices=['flat8', 'stage1', 'stage2'], required=True, help="Experiment track")
args = parser.parse_args()

if args.mode == 'flat8':
    NUM_CLASSES = 8
elif args.mode == 'stage1':
    NUM_CLASSES = 3
elif args.mode == 'stage2':
    NUM_CLASSES = 4

CHECKPOINT_DIR = f"d:\\Resfes\\Project\\Ser\\checkpoints\\{args.mode}_bigru"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Initializing Training Pipeline on: {device}")

# =====================================================================
# 2. LOSO (LEAVE-ONE-SESSION-OUT) HELPER FUNCTION
# =====================================================================
def get_loso_indices(dataset, test_session):
    """
    Directly scans the sequences array inside the dataset.
    Ensures 100% index alignment even if the dataset was sliced/trimmed.
    """
    train_indices = []
    test_indices = []
    
    for idx, seq in enumerate(dataset.sequences):
        current_utt_id = seq[-1]  
        try:
            session_id = int(current_utt_id.split('_')[0][3:5])
        except Exception as e:
            raise ValueError(f"Failed to extract Session ID from {current_utt_id}. Error: {e}")
        
        if session_id == test_session:
            test_indices.append(idx)
        else:
            train_indices.append(idx)
            
    return train_indices, test_indices

# =====================================================================
# 3. CORE TRAINING & EVALUATION FUNCTIONS
# =====================================================================
def train_epoch(model, dataloader, criterion, optimizer):
    """ Executes one training epoch """
    model.train()
    total_loss = 0.0
    valid_batches = 0
    
    for batch_X, batch_y in dataloader:
        if (batch_y != -1).sum() == 0:
            continue
        # FILTER EMPTY BATCHES (STAGE 2 SHIELD)
        # Skip loop if the batch consists entirely of -1 labels
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        valid_batches += 1
        
    return total_loss / valid_batches if valid_batches > 0 else 0.0

def evaluate(model, dataloader, criterion):
    """ Evaluates the model and computes metrics (ignoring masked labels) """
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            # FILTER EMPTY BATCHES (STAGE 2 SHIELD)
            if (batch_y != -1).sum() == 0:
                continue

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            if torch.isnan(loss):
                continue
            total_loss += loss.item()
            valid_batches += 1

            preds = torch.argmax(logits, dim=1)
            
            # Memory optimization: Append tensors directly
            all_preds.append(preds.detach())
            all_targets.append(batch_y)

    if valid_batches == 0:
        return 0.0, 0.0, 0.0
            
    # Concatenate tensors on GPU first, then transfer to CPU numpy arrays
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()
    
    # Filter out samples where the true label is -1 (Unknown)
    valid_indices = all_targets != -1
    valid_preds = all_preds[valid_indices]
    valid_targets = all_targets[valid_indices]
    
    acc = accuracy_score(valid_targets, valid_preds)
    # Explicitly set zero_division=0.0 to prevent NaN returns when a class is totally missed
    f1 = f1_score(valid_targets, valid_preds, average='macro', zero_division=0.0)
    
    return total_loss / valid_batches if valid_batches > 0 else 0.0, acc, f1

# =====================================================================
# 4. MAIN LOSO TRAINING LOOP
# =====================================================================
def run_loso_pipeline():
    full_dataset = IEMOCAPConversationalDataset(METADATA_CSV_PATH, EMBEDDINGS_PATH, mode=args.mode)
    
    fold_f1_scores = []
    fold_acc_scores = []
    
    for test_session in range(1, 6):
        print("\n" + "="*50)
        print(f"🚀 STARTING LOSO FOLD: Testing on Session {test_session}")
        print("="*50)
        
        train_idx, test_idx = get_loso_indices(full_dataset, test_session)
        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, test_idx)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"[INFO] Train samples: {len(train_subset)} | Test samples: {len(test_subset)}")
        
        model = ConversationalBiGRU(num_classes=NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        # Fixed: Initialized best_acc to prevent potential NameError bugs
        best_f1 = 0.0
        best_acc = 0.0
        
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, val_f1 = evaluate(model, test_loader, criterion)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                # Save flat baseline model
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{args.mode}_best_model_fold_{test_session}.pth"))
                
        print(f"🎯 Fold {test_session} Best Results -> Acc: {best_acc:.4f} | F1: {best_f1:.4f}")
        fold_acc_scores.append(best_acc)
        fold_f1_scores.append(best_f1)
        
    # =====================================================================
    # 5. FINAL LOSO REPORT
    # =====================================================================
    print("\n" + "#"*50)
    print("🏆 FINAL LOSO CROSS-VALIDATION REPORT")
    print("#"*50)
    print(f"Average Accuracy : {np.mean(fold_acc_scores):.4f} ± {np.std(fold_acc_scores):.4f}")
    print(f"Average Macro F1 : {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
    print("Models successfully saved to Checkpoint directory.")

if __name__ == "__main__":
    run_loso_pipeline()