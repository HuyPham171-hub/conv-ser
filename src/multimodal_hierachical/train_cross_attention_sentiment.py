import os
import gc
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# ==========================================
# 1. ENVIRONMENT & PATH CONFIGURATION
# ==========================================
def load_flexible_env():
    current_dir = Path(__file__).resolve().parent
    for check_dir in [current_dir, current_dir.parent, current_dir.parent.parent]:
        env_file = check_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return env_file
    return None

env_resolved_path = load_flexible_env()

def get_required_path(env_name):
    value = os.getenv(env_name)
    if not value:
        raise ValueError(f"[ERROR] {env_name} is not set in environment.")
    return Path(value).expanduser()

DATAFRAMES_DIR = get_required_path("DATAFRAMES_DIR")
EMBEDDINGS_DIR = get_required_path("EMBEDDINGS_DIR")

# Input Paths 
CLEAN_CSV = DATAFRAMES_DIR / "iemocap_metadata.csv" 
RESCUED_CSV = Path(r"d:\Resfes\Project\Ser\data\DataFrames\iemocap_metadata_xxx_rescued.csv")
ACOUSTIC_DIR = EMBEDDINGS_DIR / "Acoustic"

OUTPUT_DIR = Path("./checkpoints/cross_attention_stage1").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. CUSTOM MULTIMODAL PYTORCH DATASET
# ==========================================
class MultimodalEmbeddingDataset(Dataset):
    def __init__(self, metadata_df, lexical_dict, acoustic_dict, is_train=True):
        self.df = metadata_df.reset_index(drop=True)
        self.lexical_dict = lexical_dict
        self.acoustic_dict = acoustic_dict
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        utt_id = row['Utterance_ID']
        
        if utt_id not in self.lexical_dict or utt_id not in self.acoustic_dict:
            raise KeyError(f"[ERROR] Missing Embedding for Utterance_ID: {utt_id}. Check Phase 0.")
        
        # Resolve dynamic task labels
        label = int(row.get('Stage1_Label', row.get('sentiment_label')))
        
        # Fetch pre-extracted static 768-D embeddings from Phase 0
        lex_emb = self.lexical_dict[utt_id].squeeze(0)   # Shape: (768,)
        acous_emb = self.acoustic_dict[utt_id].squeeze(0) # Shape: (768,)
        
        return {
            "utt_id": utt_id,
            "lexical": lex_emb.float(),
            "acoustic": acous_emb.float(),
            "label": torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 3. LATE-FUSION CROSS-ATTENTION NETWORK
# ==========================================
class CrossAttentionFusionNetwork(nn.Module):
    def __init__(self, embed_dim=768, num_classes=3, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, lexical, acoustic):
        q = lexical.unsqueeze(1)  # Shape: (Batch, 1, 768)
        k = acoustic.unsqueeze(1) # Shape: (Batch, 1, 768)
        v = acoustic.unsqueeze(1) # Shape: (Batch, 1, 768)
        
        attn_output, _ = self.cross_attention(q, k, v)
        attn_output = attn_output.squeeze(1)          
        
        fused_hidden_state = self.layer_norm(lexical + self.dropout(attn_output)) 
        logits = self.classifier(fused_hidden_state)
        
        return logits, fused_hidden_state

# ==========================================
# 4. METRICS & MLOPS SAFEGUARDS
# ==========================================
def compute_metrics(labels, predictions):
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    uar = recall_score(labels, predictions, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": macro_f1, "uar": uar}

def plot_curves(train_losses, val_losses, fold_dir, fold):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", color="#1f77b4")
    plt.plot(val_losses, label="Validation Loss", color="#d62728", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Cross-Attention Learning Curves - Fold {fold}")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.savefig(fold_dir / "learning_curves.png", dpi=300)
    plt.close()

def plot_confusion_matrix_heatmap(labels, predictions, fold_dir, fold):
    """Generates a comprehensive 3x3 confusion matrix heatmap for Macro-Sentiment."""
    # Define label names in the exact alignment of your mapping index (0: Neg, 1: Neu, 2: Pos)
    target_names = ["Negative", "Neutral", "Positive"]
    
    # Enforce a strict 3x3 matrix layout using predefined indices
    cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        annot_kws={"size": 12, "weight": "bold"}
    )
    
    plt.xlabel("Predicted Labels", fontweight='bold', fontsize=11)
    plt.ylabel("True Labels", fontweight='bold', fontsize=11)
    plt.title(f"Confusion Matrix (Stage 1) - Fold {fold}", fontweight='bold', fontsize=13)
    plt.tight_layout()
    
    # Save target artifact to the specified fold directory
    output_path = fold_dir / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion Matrix heatmap saved at: {output_path}")

# ==========================================
# 5. ENGINE CORE: RUN TRAINING PIPELINE
# ==========================================
def main():
    print("[INFO] Initializing Late-Fusion Cross-Attention Stage 1 Pipeline...")
    
    print(f"[INFO] Loading lexical embeddings from: {EMBEDDINGS_DIR / 'lexical_embeddings.pt'}")
    lexical_dict = torch.load(EMBEDDINGS_DIR / "lexical_embeddings.pt", weights_only=False)
    
    clean_df = pd.read_csv(CLEAN_CSV)
    emotion_map = {'ang':0, 'dis':1, 'exc':2, 'fea':3, 'fru':4, 'hap':5, 'neu':6, 'sad':7, 'sur':8}
    sentiment_map = {'ang':0, 'sad':0, 'fea':0, 'fru':0, 'dis':0, 'neu':1, 'sur':1, 'hap':2, 'exc':2}
    
    clean_df = clean_df[clean_df["Raw_Emotion"].isin(emotion_map.keys())].copy()
    clean_df["Stage1_Label"] = clean_df["Raw_Emotion"].map(sentiment_map)
    
    rescued_df = pd.read_csv(RESCUED_CSV)
    
    fold_results = []
    
    for test_session in range(1, 6):
        print(f"\n{'='*60}\n[INFO] STARTING CROSS-ATTENTION FOLD {test_session}\n{'='*60}")
        
        acoustic_file = ACOUSTIC_DIR / f"acoustic_embeddings_fold_{test_session}.pt"
        print(f"[INFO] Loading anchor acoustic weights from: {acoustic_file}")
        acoustic_dict = torch.load(acoustic_file, weights_only=False)
        
        train_clean = clean_df[clean_df["Session"] != test_session]
        eval_clean = clean_df[clean_df["Session"] == test_session]
        train_rescued = rescued_df[rescued_df["Session"] != test_session]
        
        train_metadata = pd.concat([train_clean, train_rescued]).sample(frac=1, random_state=42)
        eval_metadata = eval_clean.copy()
        
        print(f"  -> Training Size   : {len(train_metadata)} samples")
        print(f"  -> Validation Size : {len(eval_metadata)} (Pure Ground Truth)")
        
        train_dataset = MultimodalEmbeddingDataset(train_metadata, lexical_dict, acoustic_dict, is_train=True)
        eval_dataset = MultimodalEmbeddingDataset(eval_metadata, lexical_dict, acoustic_dict, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        # FIXED: Removed .view(pd.Series) to prevent Pandas attribute error
        train_labels = train_metadata['Stage1_Label']
        class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels.values)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
        
        model = CrossAttentionFusionNetwork().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        
        fold_dir = OUTPUT_DIR / f"fold_{test_session}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        best_uar = 0.0
        patience = 5
        patience_counter = 0
        train_loss_history, val_loss_history = [], []
        
        for epoch in range(1, 31):
            model.train()
            total_train_loss = 0.0
            
            for batch in train_loader:
                lex = batch["lexical"].to(DEVICE)
                acous = batch["acoustic"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                optimizer.zero_grad()
                logits, _ = model(lex, acous)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
            avg_train_loss = total_train_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)
            
            model.eval()
            total_val_loss = 0.0
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for batch in eval_loader:
                    lex = batch["lexical"].to(DEVICE)
                    acous = batch["acoustic"].to(DEVICE)
                    labels = batch["label"].to(DEVICE)
                    
                    logits, _ = model(lex, acous)
                    loss = loss_fn(logits, labels)
                    total_val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
                    
            avg_val_loss = total_val_loss / len(eval_loader)
            val_loss_history.append(avg_val_loss)
            
            metrics = compute_metrics(all_labels, all_preds)
            current_uar = metrics["uar"]
            
            print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val UAR: {current_uar:.4f}")
            
            if current_uar > best_uar:
                best_uar = current_uar
                patience_counter = 0
                torch.save(model.state_dict(), fold_dir / "best_model.pt")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.")
                break
                
        plot_curves(train_loss_history, val_loss_history, fold_dir, test_session)
        
        # ==========================================
        # FINAL VERIFICATION & TENSOR EXTRACTION 
        # ==========================================
        model.load_state_dict(torch.load(fold_dir / "best_model.pt"))
        model.eval()
        
        print(f"[INFO] Extracting Stage 1 Tensors for ENTIRE Fold {test_session}...")
        full_metadata = pd.concat([train_metadata, eval_metadata]).drop_duplicates(subset=['Utterance_ID'])
        full_dataset = MultimodalEmbeddingDataset(full_metadata, lexical_dict, acoustic_dict, is_train=False)
        full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=2)

        fold_stage1_outputs = {}

        with torch.no_grad():
            for batch in full_loader:
                utt_ids = batch["utt_id"]
                lex = batch["lexical"].to(DEVICE)
                acous = batch["acoustic"].to(DEVICE)

                logits, fused_hidden_state = model(lex, acous)
                probs = torch.softmax(logits, dim=1)

                for i, uid in enumerate(utt_ids):
                    fold_stage1_outputs[uid] = {
                        "p_neg": probs[i, 0].cpu(),   
                        "v_stage1": fused_hidden_state[i].cpu() 
                    }

        torch.save(fold_stage1_outputs, fold_dir / f"stage1_outputs_fold_{test_session}.pt")
        print(f"[SUCCESS] Saved Stage 1 Output Tensors for ENTIRE Fold {test_session}.")
        
        print(f"[INFO] Calculating evaluation metrics for Fold {test_session}...")
        final_preds, final_labels = [], []

        with torch.no_grad():
            for batch in eval_loader:
                lex = batch["lexical"].to(DEVICE)
                acous = batch["acoustic"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                
                logits, _ = model(lex, acous)
                
                final_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                final_labels.extend(labels.cpu().numpy())

        print(f"[INFO] Generating Confusion Matrix for Fold {test_session}...")
        plot_confusion_matrix_heatmap(final_labels, final_preds, fold_dir, test_session)
                
        final_metrics = compute_metrics(final_labels, final_preds)
        print(f"[RESULT] Fold {test_session} Best Evaluation UAR: {final_metrics['uar']:.4f}")
        
        fold_results.append({
            "fold": test_session,
            "uar": final_metrics["uar"],
            "macro_f1": final_metrics["macro_f1"],
            "accuracy": final_metrics["accuracy"]
        })
        
        del model, optimizer, acoustic_dict
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\n[SUCCESS] CROSS-ATTENTION 5-FOLD SER RUN COMPLETED\n{'='*60}")
    uars = [r["uar"] for r in fold_results]
    print(f"Final Aggregated UAR Metrics Score: {np.mean(uars):.4f} ± {np.std(uars):.4f}")
    
    with open(OUTPUT_DIR / "stage1_summary_report.json", "w") as f:
        json.dump(fold_results, f, indent=4)

if __name__ == "__main__":
    main()