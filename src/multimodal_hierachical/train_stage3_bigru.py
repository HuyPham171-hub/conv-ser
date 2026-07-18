import os
import gc
import json
import tarfile
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Import custom dataset and collate_fn for Stage 3
from data_loaders.stage3_sequence_loader import DialogueSequenceDataset, sequence_collate_fn

# ==========================================
# 1. HARDCODED CLOUD PATH CONFIGURATION
# ==========================================
BASE_CLOUD_DIR = Path("/workspace/conv-ser")
DUALBAND_DIR = BASE_CLOUD_DIR / "data" / "Visual" / "DualBand_Spectrograms"
CHECKPOINTS_DIR = BASE_CLOUD_DIR / "checkpoints"

CLEAN_CSV = DUALBAND_DIR / "iemocap_metadata.csv" 
RESCUED_CSV = DUALBAND_DIR / "iemocap_metadata_xxx_rescued.csv"

STAGE1_OUTPUTS_DIR = CHECKPOINTS_DIR / "cross_attention_stage1"
STAGE2_OUTPUTS_DIR = CHECKPOINTS_DIR / "resnet_stage2"
STAGE3_OUTPUT_DIR = CHECKPOINTS_DIR / "bigru_stage3"
STAGE3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. AUTO-DOWNLOAD DATASETS FROM HUGGING FACE
# ==========================================
def ensure_cloud_assets_exist(hf_token: str):
    """
    Synchronizes Stage 1, Stage 2 features, and Metadata from HF Hub.
    """
    # 1. Metadata (from dualband repo)
    if not CLEAN_CSV.exists():
        print("[INFO] Metadata missing. Downloading from HF Hub...")
        snapshot_download(
            repo_id="HuyPham171/iemocap-dualband-spectrograms", repo_type="dataset",
            local_dir=DUALBAND_DIR, token=hf_token, allow_patterns=["*.csv"]
        )
        
    # 2. Stage 1 Checkpoints
    if not (STAGE1_OUTPUTS_DIR.exists() and any(STAGE1_OUTPUTS_DIR.glob("**/*.pt"))):
        print("[INFO] Stage 1 checkpoints missing. Downloading from HF Hub...")
        STAGE1_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="HuyPham171/iemocap-stage1-checkpoints", repo_type="dataset",
            local_dir=STAGE1_OUTPUTS_DIR, token=hf_token
        )
        tar_file_s1 = STAGE1_OUTPUTS_DIR / "stage1_checkpoints.tar"
        if tar_file_s1.exists():
            with tarfile.open(tar_file_s1, "r") as tar:
                tar.extractall(path=STAGE1_OUTPUTS_DIR)
            tar_file_s1.unlink()
            
    # 3. Stage 2 Gated ResNet Outputs
    if not (STAGE2_OUTPUTS_DIR.exists() and any(STAGE2_OUTPUTS_DIR.glob("**/*.pt"))):
        print("[INFO] Stage 2 Gated outputs missing. Downloading from HF Hub...")
        STAGE2_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="HuyPham171/iemocap-stage2-resnet-gated", repo_type="dataset",
            local_dir=STAGE2_OUTPUTS_DIR, token=hf_token
        )
    print("[SUCCESS] All cloud assets are successfully synchronized for Stage 3.")

# ==========================================
# 3. CONTEXTUAL BiGRU ARCHITECTURE
# ==========================================
class ContextualBiGRU(nn.Module):
    def __init__(self, input_dim, proj_dim=1024, hidden_dim=512, num_classes=5, dropout_rate=0.3):
        """
        Sequence modeling architecture to resolve dimensional mismatches and 
        capture temporal conversational context using a Bidirectional GRU.
        """
        super().__init__()
        
        # Linear Projection to strictly fix static dimensions
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Bidirectional GRU for context aggregation (past and future)
        self.gru = nn.GRU(
            input_size=proj_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification Head (hidden_dim * 2 because of bidirectionality)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x, seq_lengths):
        """
        Args:
            x: Tensor of shape (Batch, Max_Seq_Len, Input_Dim)
            seq_lengths: Tensor of valid lengths for each dialogue in the batch
        """
        # 1. Project heterogeneous concatenated features
        x_proj = self.projection(x)
        
        # 2. Pack the padded sequence to bypass zero-padding during GRU computation
        # enforce_sorted=True as the sequence_collate_fn already sorted by length descending
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x_proj, seq_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # 3. Pass through biGRU
        packed_output, _ = self.gru(packed_input)
        
        # 4. Unpack back to padded sequence format (Batch, Max_Seq_Len, Hidden_Dim * 2)
        gru_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 5. Classify the smoothed temporal states
        out = self.dropout(gru_output)
        logits = self.classifier(out)
        
        return logits

# ==========================================
# 4. MASKED LOSS FUNCTION
# ==========================================
class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        """
        Computes Cross-Entropy loss strictly on valid timesteps.
        Ignores padding and non-linguistic 'xxx' masks to prevent gradient corruption.
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=-1, reduction='none')
        
    def forward(self, logits, labels, valid_masks):
        B, L, C = logits.shape
        logits_flat = logits.view(-1, C)
        labels_flat = labels.view(-1)
        masks_flat = valid_masks.view(-1)
        
        # Calculate raw loss (padding mapped to -1 is automatically output as 0.0)
        raw_loss = self.ce(logits_flat, labels_flat)
        
        # Apply the binary valid mask (zeros out the 'xxx' garbage utterances)
        masked_loss = raw_loss * masks_flat
        
        # Average loss solely over meaningful dialogue timesteps
        valid_count = masks_flat.sum()
        if valid_count > 0:
            return masked_loss.sum() / valid_count
        else:
            return masked_loss.sum()

# ==========================================
# 5. EVALUATION METRICS & ACADEMIC PLOTTING
# ==========================================
def compute_masked_metrics(labels, predictions, valid_masks):
    labels_flat = labels.flatten()
    preds_flat = predictions.flatten()
    masks_flat = valid_masks.flatten()
    
    # Strictly filter out padding (-1) and garbage utterance masks (0.0)
    valid_idx = (labels_flat != -1) & (masks_flat == 1.0)
    
    valid_labels = labels_flat[valid_idx]
    valid_preds = preds_flat[valid_idx]
    
    if len(valid_labels) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "uar": 0.0}
        
    acc = accuracy_score(valid_labels, valid_preds)
    macro_f1 = f1_score(valid_labels, valid_preds, average="macro", zero_division=0)
    uar = recall_score(valid_labels, valid_preds, average="macro", zero_division=0)
    
    return {"accuracy": acc, "macro_f1": macro_f1, "uar": uar}

def plot_curves(train_losses, val_losses, val_uars, fold_dir, fold):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('Masked Contextual Loss', color=color, fontweight='bold')
    ax1.plot(train_losses, label="Train Loss", color='tab:orange', linestyle='-', linewidth=2)
    ax1.plot(val_losses, label="Val Loss", color=color, linestyle='--', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.7)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Negative-UAR', color=color, fontweight='bold')
    ax2.plot(val_uars, label="Val UAR", color=color, marker='o', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title(f"Stage 3 (biGRU) Learning Dynamics - Fold {fold}", fontweight='bold')
    plt.savefig(fold_dir / f"learning_curves_fold_{fold}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_heatmap(labels, predictions, valid_masks, fold_dir, fold):
    labels_flat = labels.flatten()
    preds_flat = predictions.flatten()
    masks_flat = valid_masks.flatten()
    
    valid_idx = (labels_flat != -1) & (masks_flat == 1.0)
    valid_labels = labels_flat[valid_idx]
    valid_preds = preds_flat[valid_idx]
    
    target_names = ["Anger", "Sadness", "Frustration", "Disgust", "Fear"]
    cm = confusion_matrix(valid_labels, valid_preds, labels=[0, 1, 2, 3, 4])
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 12}, cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted Emotions", fontweight='bold', fontsize=12)
    plt.ylabel("True Emotions", fontweight='bold', fontsize=12)
    plt.title(f"Context-Aware Confusion Matrix (Stage 3) - Fold {fold}", fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(fold_dir / f"confusion_matrix_fold_{fold}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 6. ENGINE CORE
# ==========================================
def main():
    print(f"\n{'='*70}\n[INFO] INITIALIZING STAGE 3: TEMPORAL CONTEXTUALIZATION (BiGRU)\n{'='*70}")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("[ERROR] Valid HF_TOKEN environment variable must be provided.")
        
    ensure_cloud_assets_exist(hf_token)
    
    clean_df = pd.read_csv(CLEAN_CSV)
    rescued_df = pd.read_csv(RESCUED_CSV)
    full_metadata = pd.concat([clean_df, rescued_df]).drop_duplicates(subset=['Utterance_ID'])
    
    print("[INFO] Aggregating Global Stage 1 & Stage 2 outputs across all folds...")
    stage1_global, stage2_global = {}, {}
    
    for fold_idx in range(1, 6):
        # Stage 1
        s1_files = list(STAGE1_OUTPUTS_DIR.rglob(f"stage1_outputs_fold_{fold_idx}.pt"))
        if s1_files:
            stage1_global.update(torch.load(s1_files[0], map_location='cpu', weights_only=True))
            
        # Stage 2
        s2_files = list(STAGE2_OUTPUTS_DIR.rglob(f"stage2_outputs_fold_{fold_idx}.pt"))
        if s2_files:
            stage2_global.update(torch.load(s2_files[0], map_location='cpu', weights_only=True))
            
    if not stage1_global or not stage2_global:
        raise FileNotFoundError("[ERROR] Missing extracted feature tensors from prior stages.")
        
    print(f"[SUCCESS] Global mappings established. Stage 1: {len(stage1_global)} | Stage 2: {len(stage2_global)}")
    
    # Infer concatenated input dimension from the first valid utterance
    sample_utt = list(stage2_global.keys())[0]
    dim_s1 = stage1_global[sample_utt]['features'].shape[-1]
    dim_s2 = stage2_global[sample_utt]['v_gated'].shape[-1]
    dynamic_input_dim = dim_s1 + dim_s2
    print(f"[INFO] Inferred Heterogeneous Feature Dimension: {dim_s1} + {dim_s2} = {dynamic_input_dim}")
    
    fold_results = []
    
    # 5-Fold Cross Validation loop
    for test_session in range(1, 6):
        print(f"\n{'-'*60}\n[INFO] STARTING BiGRU FOLD {test_session}\n{'-'*60}")
        
        # Isolate sessions securely to prevent data leakage
        train_metadata = full_metadata[full_metadata["Session"] != test_session].copy()
        eval_metadata = full_metadata[full_metadata["Session"] == test_session].copy()
        
        # Initialize sequence datasets
        train_dataset = DialogueSequenceDataset(train_metadata, stage1_global, stage2_global)
        eval_dataset = DialogueSequenceDataset(eval_metadata, stage1_global, stage2_global)
        
        # Utilizing custom sequence_collate_fn for padding and pack_padded_sequence compatibility
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=sequence_collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, collate_fn=sequence_collate_fn)
        
        # Compute dynamic class weights excluding the padding/xxx classes
        train_labels = [item['label'] for dialog in train_dataset.dialogues for item in dialog['sequence']]
        valid_train_labels = [L for L in train_labels if L != -1]
        class_weights = compute_class_weight("balanced", classes=np.unique(valid_train_labels), y=valid_train_labels)
        
        model = ContextualBiGRU(input_dim=dynamic_input_dim, num_classes=5).to(DEVICE)
        loss_fn = MaskedCrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        
        fold_dir = STAGE3_OUTPUT_DIR / f"fold_{test_session}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        best_uar = 0.0
        patience, patience_counter = 7, 0
        train_losses, val_losses, val_uars = [], [], []
        
        for epoch in range(1, 41):
            model.train()
            total_train_loss, valid_train_batches = 0.0, 0
            
            for batch in train_loader:
                v_concats = batch['v_concats'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                valid_masks = batch['valid_masks'].to(DEVICE)
                seq_lengths = batch['seq_lengths']
                
                optimizer.zero_grad()
                logits = model(v_concats, seq_lengths)
                
                loss = loss_fn(logits, labels, valid_masks)
                if not torch.isnan(loss) and loss.item() > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                    total_train_loss += loss.item()
                    valid_train_batches += 1
                    
            avg_train_loss = total_train_loss / max(1, valid_train_batches)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            total_val_loss, valid_val_batches = 0.0, 0
            all_preds, all_labels, all_masks = [], [], []
            
            with torch.no_grad():
                for batch in eval_loader:
                    v_concats = batch['v_concats'].to(DEVICE)
                    labels = batch['labels'].to(DEVICE)
                    valid_masks = batch['valid_masks'].to(DEVICE)
                    seq_lengths = batch['seq_lengths']
                    
                    logits = model(v_concats, seq_lengths)
                    loss = loss_fn(logits, labels, valid_masks)
                    
                    if not torch.isnan(loss) and loss.item() > 0:
                        total_val_loss += loss.item()
                        valid_val_batches += 1
                        
                    preds = torch.argmax(logits, dim=-1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_masks.extend(valid_masks.cpu().numpy())
                    
            avg_val_loss = total_val_loss / max(1, valid_val_batches)
            val_losses.append(avg_val_loss)
            
            metrics = compute_masked_metrics(np.array(all_labels), np.array(all_preds), np.array(all_masks))
            current_uar = metrics["uar"]
            val_uars.append(current_uar)
            
            scheduler.step(current_uar)
            
            print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val UAR: {current_uar:.4f}")
            
            # Model Checkpointing & Early Stopping
            if current_uar > best_uar:
                best_uar = current_uar
                patience_counter = 0
                torch.save(model.state_dict(), fold_dir / "best_bigru_model.pt")
                best_preds, best_labels, best_masks = all_preds, all_labels, all_masks
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.")
                break
                
        # Post-fold reporting
        plot_curves(train_losses, val_losses, val_uars, fold_dir, test_session)
        plot_confusion_matrix_heatmap(np.array(best_labels), np.array(best_preds), np.array(best_masks), fold_dir, test_session)
        
        final_metrics = compute_masked_metrics(np.array(best_labels), np.array(best_preds), np.array(best_masks))
        print(f"[RESULT] Fold {test_session} Best Temporal Negative-UAR: {final_metrics['uar']:.4f}")
        
        fold_results.append({
            "fold": test_session, 
            "uar": final_metrics["uar"],
            "macro_f1": final_metrics["macro_f1"], 
            "accuracy": final_metrics["accuracy"]
        })
        
        del model, optimizer, train_loader, eval_loader
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*70}\n[SUCCESS] STAGE 3 SEQUENCE MODELING COMPLETED\n{'='*70}")
    uars = [r["uar"] for r in fold_results]
    print(f"Final Aggregated Contextual UAR Score: {np.mean(uars):.4f} ± {np.std(uars):.4f}")
    
    with open(STAGE3_OUTPUT_DIR / "stage3_summary_report.json", "w") as f:
        json.dump(fold_results, f, indent=4)

if __name__ == "__main__":
    main()