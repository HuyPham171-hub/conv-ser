import os
import sys
import torch
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from scipy.special import softmax
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

# ==========================================
# 1. PATH RESOLUTION & CONFIGURATION
# ==========================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2] # Navigates up to project root

CLEAN_METADATA_PATH = PROJECT_ROOT / "data" / "DataFrames" / "iemocap_metadata_clean.csv"
XXX_METADATA_PATH = PROJECT_ROOT / "data" / "DataFrames" / "iemocap_metadata_xxx.csv"
PSEUDO_OUTPUT_PATH = PROJECT_ROOT / "data" / "DataFrames" / "iemocap_pseudo_labeled.csv"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "roberta_pseudo_labeler"
PLOTS_DIR = PROJECT_ROOT / "reports" / "plots"

# Ensure output directories exist
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROBERTA_MODEL_NAME = "roberta-base"

def main():
    print("[INFO] Starting Professional Ensemble Weight Optimization Pipeline...")
    print(f"[INFO] Compute Device: {DEVICE}")

    # ==========================================
    # 2. DATA LOADING & PREPARATION (LOSO SPLIT)
    # ==========================================
    if not CLEAN_METADATA_PATH.exists():
        raise FileNotFoundError(f"[ERROR] Metadata not found at: {CLEAN_METADATA_PATH}")

    df = pd.read_csv(CLEAN_METADATA_PATH)
    
    if 'Session' not in df.columns:
        raise ValueError("[ERROR] Column 'Session' does not exist in metadata.")
    
    # Filter valid emotions
    valid_emotions = ['hap', 'exc', 'neu', 'sur', 'ang', 'sad', 'fea', 'dis', 'fru']
    df = df[df['Raw_Emotion'].isin(valid_emotions)].copy()
    
    # Encode categorical labels to integers (0 to num_classes - 1)
    label_encoder = LabelEncoder()
    df['label_id'] = label_encoder.fit_transform(df['Raw_Emotion'])
    num_classes = len(label_encoder.classes_)
    print(f"[INFO] Detected {num_classes} emotion classes: {label_encoder.classes_}")

    # STRICT LOSO SPLIT: Train (Sessions 1-4) | Validation (Session 5)
    train_df = df[df['Session'] != 5].copy()
    val_df = df[df['Session'] == 5].copy()
    
    print(f"[INFO] LOSO Data Split: {len(train_df)} Train rows | {len(val_df)} Val rows")

    # ==========================================
    # 3. TRAIN PSYCHOLOGICAL ANCHOR (KNN on V-A-D)
    # ==========================================
    print("\n[INFO] --- Training KNN Predictor (Valence, Arousal, Dominance) ---")
    vad_features = ['Valence', 'Arousal', 'Dominance']
    
    # Standardize 3D coordinates to ensure equal distance weighting
    scaler = StandardScaler()
    X_train_vad = scaler.fit_transform(train_df[vad_features])
    X_val_vad = scaler.transform(val_df[vad_features])
    
    y_train = train_df['label_id'].values
    y_val = val_df['label_id'].values

    # Train K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
    knn.fit(X_train_vad, y_train)
    
    # Extract prediction probabilities (Shape: [num_val_samples, num_classes])
    knn_val_probs = knn.predict_proba(X_val_vad)
    print("[SUCCESS] KNN Training and Inference completed.")

    # ==========================================
    # 4. TRAIN LEXICAL ANCHOR (RoBERTa on Transcript)
    # ==========================================
    print("\n[INFO] --- Training RoBERTa Predictor (Transcripts) ---")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)

    # Convert Pandas DataFrame to Hugging Face Dataset
    train_ds = Dataset.from_pandas(train_df[['Transcript', 'label_id']])
    val_ds = Dataset.from_pandas(val_df[['Transcript', 'label_id']])

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['Transcript'], padding="max_length", truncation=True, max_length=128)

    train_ds = train_ds.map(tokenize_function, batched=True).rename_column("label_id", "labels")
    val_ds = val_ds.map(tokenize_function, batched=True).rename_column("label_id", "labels")

    # Initialize Model
    model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_MODEL_NAME, num_labels=num_classes
    ).to(DEVICE)

    # Define training arguments specifically optimized for evaluation and plotting
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        eval_strategy="steps",        # Evaluate periodically during training
        logging_strategy="steps",     # Log metrics periodically
        eval_steps=50,                # Evaluation interval
        logging_steps=50,             # Logging interval
        save_strategy="no",           # Do not save checkpoints to save disk space
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(), 
        report_to="none"              # Disable external loggers (e.g., WandB)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("[INFO] Starting RoBERTa fine-tuning...")
    trainer.train()

    # ---------------------------------------------------------
    # PLOT 1: RoBERTa Learning Curves (Train vs Eval Loss)
    # ---------------------------------------------------------
    print("[INFO] Generating Learning Curves...")
    log_history = trainer.state.log_history
    steps, train_loss, eval_loss = [], [], []
    
    for log in log_history:
        if "step" in log:
            if "loss" in log:
                steps.append(log["step"])
                train_loss.append(log["loss"])
            if "eval_loss" in log:
                # Sync evaluation logs with training steps
                if len(eval_loss) < len(train_loss):
                    eval_loss.append(log["eval_loss"])
                    
    plt.figure(figsize=(10, 6))
    plt.plot(steps[:len(train_loss)], train_loss, label="Training Loss", color='#1f77b4', linewidth=2)
    plt.plot(steps[:len(eval_loss)], eval_loss, label="Validation Loss", color='#d62728', linestyle='--', linewidth=2)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("RoBERTa Fine-Tuning Loss History", fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roberta_learning_curves.png", dpi=300)
    plt.close()

    # Extract raw logits and convert to probabilities
    print("[INFO] Running RoBERTa Inference on Validation set...")
    roberta_predictions = trainer.predict(val_ds)
    roberta_val_probs = softmax(roberta_predictions.predictions, axis=1) 
    print("[SUCCESS] RoBERTa Training and Inference completed.")

    # ==========================================
    # 5. ENSEMBLE WEIGHT OPTIMIZATION (GRID SEARCH)
    # ==========================================
    print("\n[INFO] --- Running Alpha Weight Optimization ---")
    best_alpha = 0.0
    best_f1 = 0.0
    alphas, f1_scores, accuracies = [], [], []
    results_log = []

    # Iterate alpha from 0.0 to 1.0 in steps of 0.05
    for alpha in np.arange(0.0, 1.05, 0.05):
        # Ensemble Probability Formula: P_ensemble = α * P_roberta + (1 - α) * P_knn
        ensemble_probs = (alpha * roberta_val_probs) + ((1.0 - alpha) * knn_val_probs)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        # Calculate evaluation metrics
        macro_f1 = f1_score(y_val, ensemble_preds, average='macro')
        acc = accuracy_score(y_val, ensemble_preds)
        
        alphas.append(alpha)
        f1_scores.append(macro_f1)
        accuracies.append(acc)
        results_log.append({'Alpha': alpha, 'Macro_F1': macro_f1, 'Accuracy': acc})
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_alpha = alpha

    results_df = pd.DataFrame(results_log)
    print("\n--- Grid Search Results ---")
    print(results_df.to_string(index=False))
    
    print("\n" + "="*60)
    print(f"[CONCLUSION] Optimal RoBERTa Weight (α) : {best_alpha:.2f}")
    print(f"[CONCLUSION] Optimal KNN Weight (1 - α) : {(1.0 - best_alpha):.2f}")
    print(f"[CONCLUSION] Best Macro F1 Score      : {best_f1:.4f}")
    print("="*60)

    # ---------------------------------------------------------
    # PLOT 2: Alpha Optimization Curve
    # ---------------------------------------------------------
    print("[INFO] Generating Alpha Optimization Curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, f1_scores, marker='o', label='Macro F1-Score', color='#ff7f0e', linewidth=2)
    plt.plot(alphas, accuracies, marker='s', label='Accuracy', color='#2ca02c', linestyle=':', linewidth=2)
    plt.axvline(x=best_alpha, color='#d62728', linestyle='--', linewidth=2, label=f'Optimal Alpha ({best_alpha:.2f})')
    plt.xlabel('Alpha (Lexical/RoBERTa Weight)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Ensemble Weight Optimization', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "alpha_optimization_curve.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 3: Confusion Matrix & Classification Report
    # ---------------------------------------------------------
    print("\n[INFO] Generating Confusion Matrix for Best Ensemble Model...")
    best_probs = (best_alpha * roberta_val_probs) + ((1.0 - best_alpha) * knn_val_probs)
    best_preds = np.argmax(best_probs, axis=1)
    
    print("\n--- Validation Classification Report ---")
    print(classification_report(
        y_val, 
        best_preds, 
        labels=range(num_classes),             # Ép sklearn nhận diện đủ 9 nhãn
        target_names=label_encoder.classes_, 
        zero_division=0                        # Tránh cảnh báo chia cho 0 với nhãn thiếu
    ))

    cm = confusion_matrix(y_val, best_preds, labels=range(num_classes))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_,
                annot_kws={"size": 11})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Ensemble Confusion Matrix (Alpha = {best_alpha:.2f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ensemble_confusion_matrix.png", dpi=300)
    plt.close()

    # ==========================================
    # 6. PHASE 1.5: RESCUE AMBIGUOUS DATA ('xxx')
    # ==========================================
    print("\n[INFO] --- Phase 1.5: Rescuing 'xxx' Data ---")
    
    if not XXX_METADATA_PATH.exists():
        print(f"[WARNING] Target data not found at: {XXX_METADATA_PATH}. Skipping Phase 1.5.")
        return

    df_xxx = pd.read_csv(XXX_METADATA_PATH)
    initial_xxx_count = len(df_xxx)
    
    # Clean structural noise and drop empty transcripts
    df_xxx["Transcript"] = df_xxx["Transcript"].fillna("").astype(str)
    df_xxx["Transcript"] = df_xxx["Transcript"].str.replace(r'\[.*?\]', '', regex=True)
    df_xxx["Transcript"] = df_xxx["Transcript"].str.replace(r'\(.*?\)', '', regex=True)
    df_xxx["Transcript"] = df_xxx["Transcript"].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    df_xxx = df_xxx[df_xxx["Transcript"] != ""].reset_index(drop=True)
    print(f"[INFO] Processing {len(df_xxx)} valid 'xxx' samples (Dropped {initial_xxx_count - len(df_xxx)} empty/noise transcripts).")

    # Predict with Psychological Anchor (KNN)
    X_xxx_vad = scaler.transform(df_xxx[vad_features])
    knn_xxx_probs = knn.predict_proba(X_xxx_vad)

    # Predict with Lexical Anchor (RoBERTa)
    xxx_ds = Dataset.from_pandas(df_xxx[['Transcript']])
    xxx_ds = xxx_ds.map(tokenize_function, batched=True)
    
    print("[INFO] Running RoBERTa Inference on ambiguous data...")
    roberta_xxx_preds = trainer.predict(xxx_ds)
    roberta_xxx_probs = softmax(roberta_xxx_preds.predictions, axis=1)

    # Apply Optimized Ensemble Weights
    ensemble_xxx_probs = (best_alpha * roberta_xxx_probs) + ((1.0 - best_alpha) * knn_xxx_probs)
    
    max_probs = np.max(ensemble_xxx_probs, axis=1)
    pred_indices = np.argmax(ensemble_xxx_probs, axis=1)

    # Apply Confidence Threshold Filter
    TAU = 0.85
    confident_mask = max_probs >= TAU
    
    df_rescued = df_xxx[confident_mask].copy()
    
    # Map indices back to original string labels and save
    df_rescued['Pseudo_Label'] = label_encoder.inverse_transform(pred_indices[confident_mask])
    df_rescued['Confidence_Score'] = max_probs[confident_mask]
    df_rescued['Raw_Emotion'] = df_rescued['Pseudo_Label'] 
    
    df_rescued.to_csv(PSEUDO_OUTPUT_PATH, index=False)

    # ---------------------------------------------------------
    # PLOT 4: Confidence Score Distribution
    # ---------------------------------------------------------
    print("[INFO] Generating Confidence Distribution Histogram...")
    plt.figure(figsize=(10, 6))
    sns.histplot(max_probs, bins=40, kde=True, color='#9467bd', edgecolor='black', alpha=0.7)
    plt.axvline(x=TAU, color='#d62728', linestyle='--', linewidth=2.5, label=f'Selection Threshold ($\tau$ = {TAU})')
    
    # Highlight the accepted region
    plt.axvspan(TAU, 1.0, color='#2ca02c', alpha=0.15, label='Rescued Region')
    
    plt.xlabel('Ensemble Confidence Score (Max Probability)', fontsize=12)
    plt.ylabel('Frequency (Utterances)', fontsize=12)
    plt.title('Confidence Distribution on Ambiguous Data (xxx)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "xxx_confidence_distribution.png", dpi=300)
    plt.close()
    
    print("\n" + "="*60)
    print(f"[VICTORY] Successfully rescued {len(df_rescued)} out of {len(df_xxx)} ambiguous samples.")
    print(f"[OUTPUT] Rescued dataset saved to: {PSEUDO_OUTPUT_PATH}")
    print(f"[OUTPUT] All evaluation plots securely saved to: {PLOTS_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()