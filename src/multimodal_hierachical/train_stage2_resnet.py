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
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Import custom dataset and collate_fn
from data_loaders.stage2_multimodal_loader import DualBandResNetDataset, dualband_pad_collate_fn

# ==========================================
# 1. HARDCODED CLOUD PATH CONFIGURATION (NO .ENV)
# ==========================================
BASE_CLOUD_DIR = Path("/workspace/conv-ser")
DUALBAND_DIR = BASE_CLOUD_DIR / "data" / "Visual" / "DualBand_Spectrograms"
CHECKPOINTS_DIR = BASE_CLOUD_DIR / "checkpoints"
CLEAN_CSV = DUALBAND_DIR / "iemocap_metadata.csv" 
RESCUED_CSV = DUALBAND_DIR / "iemocap_metadata_xxx_rescued.csv"
STAGE1_OUTPUTS_DIR = CHECKPOINTS_DIR / "cross_attention_stage1"
OUTPUT_DIR = CHECKPOINTS_DIR / "resnet_stage2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. AUTO-DOWNLOAD & EXTRACT DATASETS FROM HF HUB
# ==========================================
def ensure_cloud_assets_exist(hf_token: str):
    if not (DUALBAND_DIR.exists() and any(DUALBAND_DIR.glob("*.pt"))):
        print(f"[INFO] Spectrograms missing or incomplete. Downloading from HF Hub...")
        DUALBAND_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="HuyPham171/iemocap-dualband-spectrograms", repo_type="dataset",
            local_dir=DUALBAND_DIR, token=hf_token, max_workers=8, ignore_patterns=[".gitattributes", "README.md"]
        )
        tar_file = DUALBAND_DIR / "spectrograms.tar"
        if tar_file.exists():
            print(f"[INFO] Extracting {tar_file.name} to {DUALBAND_DIR}...")
            with tarfile.open(tar_file, "r") as tar:
                tar.extractall(path=DUALBAND_DIR)
            tar_file.unlink()
        
    if not (STAGE1_OUTPUTS_DIR.exists() and any(STAGE1_OUTPUTS_DIR.glob("**/*.pt"))):
        print(f"[INFO] Stage 1 checkpoints missing. Downloading from HF Hub...")
        STAGE1_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="HuyPham171/iemocap-stage1-checkpoints", repo_type="dataset",
            local_dir=STAGE1_OUTPUTS_DIR, token=hf_token, max_workers=4
        )
        tar_file_s1 = STAGE1_OUTPUTS_DIR / "stage1_checkpoints.tar"
        if tar_file_s1.exists():
            print(f"[INFO] Extracting {tar_file_s1.name} to {STAGE1_OUTPUTS_DIR}...")
            with tarfile.open(tar_file_s1, "r") as tar:
                tar.extractall(path=STAGE1_OUTPUTS_DIR)
            tar_file_s1.unlink()
    print("[SUCCESS] All cloud assets are successfully synchronized and ready for training.")

# ==========================================
# 3. MODIFIED RESNET ARCHITECTURE (DUAL-BAND)
# ==========================================
class ModifiedDualBandResNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        old_weights = base_model.conv1.weight.data
        self.conv1.weight.data[:, 0, :, :] = old_weights.mean(dim=1)
        self.conv1.weight.data[:, 1, :, :] = old_weights.mean(dim=1)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        v_resnet = torch.flatten(x, 1) 
        logits = self.fc(v_resnet)
        return logits, v_resnet

# ==========================================
# 4. MASKED METRICS & PLOTTING
# ==========================================
def compute_masked_metrics(labels, predictions):
    valid_indices = labels != -1
    valid_labels = labels[valid_indices]
    valid_preds = predictions[valid_indices]
    if len(valid_labels) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "uar": 0.0}
    acc = accuracy_score(valid_labels, valid_preds)
    macro_f1 = f1_score(valid_labels, valid_preds, average="macro", zero_division=0)
    uar = recall_score(valid_labels, valid_preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": macro_f1, "uar": uar}

def plot_curves(train_losses, val_losses, val_uars, fold_dir, fold):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Masked Cross-Entropy Loss', color=color)
    ax1.plot(train_losses, label="Train Loss", color='tab:orange', linestyle='-')
    ax1.plot(val_losses, label="Val Loss", color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation UAR', color=color)
    ax2.plot(val_uars, label="Val UAR", color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.title(f"Stage 2 (ResNet) Learning Curves - Fold {fold}")
    plt.savefig(fold_dir / "learning_curves.png", dpi=300)
    plt.close()

def plot_confusion_matrix_heatmap(labels, predictions, fold_dir, fold):
    valid_indices = labels != -1
    valid_labels = labels[valid_indices]
    valid_preds = predictions[valid_indices]
    target_names = ["Anger", "Sadness", "Frustration", "Disgust", "Fear"]
    cm = confusion_matrix(valid_labels, valid_preds, labels=[0, 1, 2, 3, 4])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted Labels", fontweight='bold')
    plt.ylabel("True Labels", fontweight='bold')
    plt.title(f"Fine-Grained Confusion Matrix (Stage 2) - Fold {fold}", fontweight='bold')
    plt.tight_layout()
    plt.savefig(fold_dir / "confusion_matrix.png", dpi=300)
    plt.close()

# ==========================================
# 5. ENGINE CORE: RUN TRAINING PIPELINE
# ==========================================
def main():
    print("[INFO] Initializing Stage 2: Dual-Branch ResNet & Soft Gating Pipeline...")
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("[ERROR] Valid HF_TOKEN must be provided to fetch cloud data. Set it via 'export HF_TOKEN=\"your_token\"'.")
        
    ensure_cloud_assets_exist(hf_token)
    clean_df = pd.read_csv(CLEAN_CSV)
    rescued_df = pd.read_csv(RESCUED_CSV)
    
    # -----------------------------------------------------------------
    # IMPROVEMENT 1: Recursive Glob for Robust Stage 1 Loading
    # -----------------------------------------------------------------
    print("[INFO] Aggregating Stage 1 outputs across all folds to build global mapping...")
    stage1_outputs = {}
    for fold_idx in range(1, 6):
        target_filename = f"stage1_outputs_fold_{fold_idx}.pt"
        matched_files = list(STAGE1_OUTPUTS_DIR.rglob(target_filename))
        
        if matched_files:
            fold_data = torch.load(matched_files[0], weights_only=True)
            stage1_outputs.update(fold_data)
            print(f"  -> Loaded {len(fold_data)} items from {target_filename}")
        else:
            print(f"  -> [WARNING] {target_filename} is completely missing in {STAGE1_OUTPUTS_DIR}!")
            
    if not stage1_outputs:
        raise FileNotFoundError(f"[ERROR] No Stage 1 checkpoint files found in {STAGE1_OUTPUTS_DIR}.")
    
    print(f"[SUCCESS] Global Stage 1 mapping established with {len(stage1_outputs)} total items.")

    fold_results = []
    
    for test_session in range(1, 6):
        print(f"\n{'='*60}\n[INFO] STARTING RESNET FOLD {test_session}\n{'='*60}")
        
        train_clean = clean_df[clean_df["Session"] != test_session]
        eval_clean = clean_df[clean_df["Session"] == test_session]
        train_rescued = rescued_df[rescued_df["Session"] != test_session]
        
        train_metadata = pd.concat([train_clean, train_rescued]).sample(frac=1, random_state=42)
        eval_metadata = eval_clean.copy()
        
        # -----------------------------------------------------------------
        # IMPROVEMENT 2: Safeguard Intersection (Prevents KeyError)
        # -----------------------------------------------------------------
        valid_stage1_keys = set(stage1_outputs.keys())
        original_train_len = len(train_metadata)
        
        train_metadata = train_metadata[train_metadata['Utterance_ID'].isin(valid_stage1_keys)].copy()
        eval_metadata = eval_metadata[eval_metadata['Utterance_ID'].isin(valid_stage1_keys)].copy()
        
        dropped_samples = original_train_len - len(train_metadata)
        if dropped_samples > 0:
            print(f"[WARNING] Dropped {dropped_samples} training samples because they are missing from Stage 1 outputs.")
            
        if len(train_metadata) == 0:
            raise ValueError(f"[ERROR] Fold {test_session} training data is completely empty after intersection.")
        # -----------------------------------------------------------------
        
        train_dataset = DualBandResNetDataset(train_metadata, DUALBAND_DIR, stage1_outputs, is_train=True)
        eval_dataset = DualBandResNetDataset(eval_metadata, DUALBAND_DIR, stage1_outputs, is_train=False)
        
        num_workers_cfg = max(2, os.cpu_count() // 2)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers_cfg, collate_fn=dualband_pad_collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=num_workers_cfg, collate_fn=dualband_pad_collate_fn)
        
        fine_grained_map = {'ang': 0, 'sad': 1, 'fru': 2, 'dis': 3, 'fea': 4}
        train_mapped_series = train_metadata['Raw_Emotion'].astype(str).str.lower().map(fine_grained_map)
        valid_train_labels = train_mapped_series[train_mapped_series.notna() & (train_mapped_series != -1)].astype(int).values

        if len(valid_train_labels) == 0:
            raise ValueError(f"[ERROR] Fold {test_session} contains no valid negative fine-grained labels in its training split.")

        class_weights = compute_class_weight("balanced", classes=np.unique(valid_train_labels), y=valid_train_labels)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE), ignore_index=-1)
        
        model = ModifiedDualBandResNet(num_classes=5).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        
        fold_dir = OUTPUT_DIR / f"fold_{test_session}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        best_uar = 0.0
        patience = 5
        patience_counter = 0
        train_loss_history, val_loss_history, val_uar_history = [], [], []
        
        for epoch in range(1, 31):
            model.train()
            total_train_loss = 0.0
            valid_train_batches = 0
            
            for batch in train_loader:
                specs = batch["spectrograms"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                
                optimizer.zero_grad()
                logits, _ = model(specs)
                loss = loss_fn(logits, labels)
                
                if not torch.isnan(loss) and loss.item() > 0:
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
                    valid_train_batches += 1
                
            avg_train_loss = total_train_loss / max(1, valid_train_batches)
            train_loss_history.append(avg_train_loss)
            
            model.eval()
            total_val_loss = 0.0
            valid_val_batches = 0
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for batch in eval_loader:
                    specs = batch["spectrograms"].to(DEVICE)
                    labels = batch["labels"].to(DEVICE)
                    
                    logits, _ = model(specs)
                    loss = loss_fn(logits, labels)
                    
                    if not torch.isnan(loss) and loss.item() > 0:
                        total_val_loss += loss.item()
                        valid_val_batches += 1
                    
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
                    
            avg_val_loss = total_val_loss / max(1, valid_val_batches)
            val_loss_history.append(avg_val_loss)
            
            metrics = compute_masked_metrics(np.array(all_labels), np.array(all_preds))
            current_uar = metrics["uar"]
            val_uar_history.append(current_uar)
            
            print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Neg-UAR: {current_uar:.4f}")
            
            if current_uar > best_uar:
                best_uar = current_uar
                patience_counter = 0
                torch.save(model.state_dict(), fold_dir / "best_model.pt")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.")
                break
                
        plot_curves(train_loss_history, val_loss_history, val_uar_history, fold_dir, test_session)
        
        model.load_state_dict(torch.load(fold_dir / "best_model.pt"))
        model.eval()
        
        print(f"[INFO] Extracting Soft Gated Tensors (V_gated) for ENTIRE Fold {test_session}...")
        full_metadata = pd.concat([train_metadata, eval_metadata]).drop_duplicates(subset=['Utterance_ID'])
        full_dataset = DualBandResNetDataset(full_metadata, DUALBAND_DIR, stage1_outputs, is_train=False)
        full_loader = DataLoader(full_dataset, batch_size=16, shuffle=False, num_workers=num_workers_cfg, collate_fn=dualband_pad_collate_fn)

        fold_stage2_outputs = {}
        final_preds, final_labels = [], []

        with torch.no_grad():
            for batch in full_loader:
                utt_ids = batch["utt_ids"]
                specs = batch["spectrograms"].to(DEVICE)
                p_negs = batch["p_negs"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                logits, v_resnet = model(specs)
                v_gated = v_resnet * p_negs.unsqueeze(1)
                
                final_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                final_labels.extend(labels.cpu().numpy())

                for i, uid in enumerate(utt_ids):
                    fold_stage2_outputs[uid] = {"v_gated": v_gated[i].cpu()}

        torch.save(fold_stage2_outputs, fold_dir / f"stage2_outputs_fold_{test_session}.pt")
        print(f"[SUCCESS] Saved V_gated Tensors for Stage 3.")
        
        plot_confusion_matrix_heatmap(np.array(final_labels), np.array(final_preds), fold_dir, test_session)
        eval_metrics = compute_masked_metrics(np.array(final_labels), np.array(final_preds))
        print(f"[RESULT] Fold {test_session} Negative-UAR: {eval_metrics['uar']:.4f}")
        
        fold_results.append({
            "fold": test_session, "uar": eval_metrics["uar"],
            "macro_f1": eval_metrics["macro_f1"], "accuracy": eval_metrics["accuracy"]
        })
        
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}\n[SUCCESS] STAGE 2 (DUAL-BAND RESNET) 5-FOLD RUN COMPLETED\n{'='*60}")
    uars = [r["uar"] for r in fold_results]
    print(f"Final Aggregated Negative-UAR Metrics Score: {np.mean(uars):.4f} ± {np.std(uars):.4f}")
    
    with open(OUTPUT_DIR / "stage2_summary_report.json", "w") as f:
        json.dump(fold_results, f, indent=4)

    del stage1_outputs
    gc.collect()

if __name__ == "__main__":
    main()