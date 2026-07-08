import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Import custom modules based on the new folder structure
from data_loaders.multimodal_loader import IEMOCAPMultimodalDataset
from models.stage1_multimodal import Stage1SentimentClassifier

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Resolve the path to the .env file located at the project root (2 levels up)
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

def get_required_path(env_name):
    """Fetches a directory path from the environment and expands user variables."""
    value = os.getenv(env_name)
    if not value:
        raise ValueError(f"[ERROR] {env_name} is not set in {ENV_PATH}")
    return Path(value).expanduser()

# Dynamically build paths using environment variables
DATAFRAMES_DIR = get_required_path("DATAFRAMES_DIR")
EMBEDDINGS_DIR = get_required_path("EMBEDDINGS_DIR")
CHECKPOINTS_BASE_DIR = get_required_path("CHECKPOINTS_DIR")

# Define target file pathways
METADATA_PATH = DATAFRAMES_DIR / "iemocap_metadata.csv"
AUDIO_EMB_PATH = EMBEDDINGS_DIR / "iemocap_wav2vec2_embeddings.npy"
TEXT_EMB_PATH = EMBEDDINGS_DIR / "iemocap_roberta_embeddings.npy"
CHECKPOINT_DIR = CHECKPOINTS_BASE_DIR / "multimodal_stage1"

# Ensure checkpoint directory exists
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-4

# Fetch device configuration
requested_device = os.getenv("DEVICE", "auto").lower()
if requested_device == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif requested_device == "cuda" and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ==========================================
# 2. TRAINING ROUTINE
# ==========================================
def train_loso():
    print(f"[INFO] Initializing Stage 1 Training Pipeline on: {DEVICE}")
    
    # Load the entire dataset once to save RAM
    full_dataset = IEMOCAPMultimodalDataset(
        metadata_path=METADATA_PATH,
        audio_emb_path=AUDIO_EMB_PATH,
        text_emb_path=TEXT_EMB_PATH
    )
    
    # Storage for cross-validation metrics
    fold_accuracies = []
    fold_f1_scores = []
    
    # Leave-One-Session-Out (LOSO) Cross-Validation (Sessions 1 to 5)
    for test_session in range(1, 6):
        print(f"\n{'='*50}")
        print(f"🚀 STARTING LOSO FOLD: Testing on Session {test_session}")
        print(f"{'='*50}")
        
        # 2.1 Split indices based on the Session number in utterance ID (e.g., 'Ses01')
        test_prefix = f"Ses0{test_session}"
        train_indices = []
        test_indices = []
        
        for idx, sample in enumerate(full_dataset.samples):
            if sample['utt_id'].startswith(test_prefix):
                test_indices.append(idx)
            else:
                train_indices.append(idx)
                
        train_subset = Subset(full_dataset, train_indices)
        test_subset = Subset(full_dataset, test_indices)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"[INFO] Train samples: {len(train_indices)} | Test samples: {len(test_indices)}")
        
        # 2.2 Initialize Model, Loss, and Optimizer
        model = Stage1SentimentClassifier(embed_dim=768, num_classes=3, dropout=0.3).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        
        # AdamW is generally better for Transformer-based embeddings
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        best_f1 = 0.0
        best_acc = 0.0
        
        # 2.3 Epoch Loop
        for epoch in range(1, EPOCHS + 1):
            # --- TRAIN PHASE ---
            model.train()
            train_loss = 0.0
            
            for batch_audio, batch_text, batch_labels in train_loader:
                batch_audio = batch_audio.to(DEVICE)
                batch_text = batch_text.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                
                optimizer.zero_grad()
                logits = model(audio_emb=batch_audio, text_emb=batch_text)
                loss = criterion(logits, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_audio.size(0)
                
            train_loss = train_loss / len(train_indices)
            
            # --- EVALUATION PHASE ---
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_audio, batch_text, batch_labels in test_loader:
                    batch_audio = batch_audio.to(DEVICE)
                    batch_text = batch_text.to(DEVICE)
                    batch_labels = batch_labels.to(DEVICE)
                    
                    logits = model(audio_emb=batch_audio, text_emb=batch_text)
                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item() * batch_audio.size(0)
                    
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
                    
            val_loss = val_loss / len(test_indices)
            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='macro')
            
            # Log metrics every 5 epochs or on the first epoch
            if epoch == 1 or epoch % 5 == 0:
                print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            # 2.4 Checkpoint Saving
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                checkpoint_path = CHECKPOINT_DIR / f"stage1_fold{test_session}_best.pth"
                torch.save(model.state_dict(), checkpoint_path)
                
        print(f"🎯 Fold {test_session} Best Results -> Acc: {best_acc:.4f} | F1: {best_f1:.4f}")
        fold_accuracies.append(best_acc)
        fold_f1_scores.append(best_f1)
        
    # ==========================================
    # 3. FINAL REPORT
    # ==========================================
    print(f"\n{'#'*50}")
    print("🏆 FINAL MULTIMODAL STAGE 1 LOSO REPORT")
    print(f"{'#'*50}")
    print(f"Average Accuracy : {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Average Macro F1 : {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
    print("[INFO] Model weights successfully saved to Checkpoint directory.")

if __name__ == "__main__":
    train_loso()