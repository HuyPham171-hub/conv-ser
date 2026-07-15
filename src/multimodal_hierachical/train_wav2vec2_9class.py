import os
# Enable fast transfers for Hugging Face Hub (crucial for Vast.ai cloud instances)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import json
import gc
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. ENVIRONMENT & MLOPS CONFIGURATION
# ==========================================
def load_flexible_env():
    """Locates and loads the .env file recursively from parent directories."""
    current_dir = Path(__file__).resolve().parent
    for check_dir in [current_dir, current_dir.parent, current_dir.parent.parent, current_dir.parent.parent.parent]:
        env_file = check_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return env_file
    return None

env_resolved_path = load_flexible_env()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("[ERROR] HF_TOKEN is not found in environment or .env file.")

# Set to True to run a fast 1-epoch test on subset data before full training
DUMMY_RUN = False  

# TARGET HF REPOSITORY: 9-class emotions
DATASET_REPO = "HuyPham171/iemocap-emotion-clean"
MODEL_ID = "facebook/wav2vec2-base"

# Ensure output directory exists in the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3] 
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "wav2vec2_9class_stage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. CUSTOM TRAINER FOR WEIGHTED LOSS
# ==========================================
# class WeightedTrainer(Trainer):
#     """
#     Custom Trainer that injects class weights into the CrossEntropyLoss function 
#     to handle the severe class imbalance in IEMOCAP 9-class data.
#     """
#     def __init__(self, class_weights, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#         loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss

class FocalLossTrainer(Trainer):
    def __init__(self, class_weights, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Calculate standard cross entropy without reduction
        ce_loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        ce_loss = ce_loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        # Calculate pt (probability of correct class)
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss extension
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        loss = focal_loss.mean()
        
        return (loss, outputs) if return_outputs else loss

# ==========================================
# 3. METRICS & PLOTTING FUNCTIONS
# ==========================================
def compute_metrics(eval_pred):
    """Computes academic metrics with zero_division safeguards for minority classes."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    
    acc = accuracy_score(labels, predictions)
    # Use zero_division=0 to prevent crashes if 'dis' or 'fea' are not predicted
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    uar = recall_score(labels, predictions, average="macro", zero_division=0)
    
    return {"accuracy": acc, "macro_f1": macro_f1, "uar": uar}

def plot_learning_curves(log_history, output_dir, fold):
    """Plots training vs validation loss across epochs."""
    train_loss = [x["loss"] for x in log_history if "loss" in x]
    eval_loss = [x["eval_loss"] for x in log_history if "eval_loss" in x]
    
    epochs_train = [x["epoch"] for x in log_history if "loss" in x]
    epochs_eval = [x["epoch"] for x in log_history if "eval_loss" in x]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_train, train_loss, label="Train Loss", marker="o", color='#1f77b4')
    plt.plot(epochs_eval, eval_loss, label="Validation Loss", marker="x", color='#d62728', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Focal Loss")
    plt.title(f"Wav2Vec2 Learning Curves - Fold {fold}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(output_dir / "learning_curves.png", dpi=300)
    plt.close()

def plot_confusion_matrix_heatmap(labels, predictions, id2label, output_dir, fold):
    """Generates a comprehensive 9x9 confusion matrix."""
    # Enforce a 9x9 matrix even if some classes are missing in the validation set
    cm = confusion_matrix(labels, predictions, labels=list(id2label.keys()))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(id2label.values()),
                yticklabels=list(id2label.values()))
    plt.xlabel("Predicted Labels", fontweight='bold')
    plt.ylabel("True Labels", fontweight='bold')
    plt.title(f"Confusion Matrix (9 Classes) - Fold {fold}", fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()

# ==========================================
# 4. DATA COLLATOR
# ==========================================
@dataclass
class DataCollatorAudioWithPadding:
    """Dynamically pads audio sequences to the length of the longest sample in the batch."""
    feature_extractor: AutoFeatureExtractor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["label"] for feature in features]
        batch = self.feature_extractor.pad(input_features, padding=self.padding, return_tensors="pt")
        batch["labels"] = torch.tensor(label_features, dtype=torch.long)
        return batch

# ==========================================
# 5. MAIN TRAINING PIPELINE (LOSO CV)
# ==========================================
def main():
    print(f"[INFO] Initializing Cloud 9-Class Training Pipeline...")
    print(f"[INFO] Dummy Run Status: {'ACTIVE (Cloud Test)' if DUMMY_RUN else 'INACTIVE (Production Run)'}")
    print(f"[INFO] Target Device: {DEVICE}")

    # Initialize Feature Extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, token=HF_TOKEN)
    
    # 9-Class Label Mapping (Ensure alphabetical mapping matches your dataset encoding)
    label2id = {
        "ang": 0, "dis": 1, "exc": 2, "fea": 3, "fru": 4, 
        "hap": 5, "neu": 6, "sad": 7, "sur": 8
    }
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)
    
    config = AutoConfig.from_pretrained(
        MODEL_ID, num_labels=num_classes, label2id=label2id, id2label=id2label,
        finetuning_task="audio-classification", token=HF_TOKEN
    )
    
    print(f"[INFO] Connecting to Hugging Face Hub to load: {DATASET_REPO}")
    dataset = load_dataset(DATASET_REPO, token=HF_TOKEN)
    
    # Filter safety net: Ensure 'xxx' and 'oth' are completely dropped if they exist in HF Repo
    print("[INFO] Enforcing strict 9-class filtering...")
    valid_labels = list(label2id.values())
    dataset = dataset.filter(lambda x: x["label"] in valid_labels)

    if "audio" not in dataset["train"].column_names:
        dataset = dataset.rename_column("file_name", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def preprocess_function(batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        # Cap max audio length at 15 seconds to prevent OOM errors on the GPU
        inputs = feature_extractor(
            audio_arrays, sampling_rate=feature_extractor.sampling_rate,
            truncation=True, max_length=16000 * 15 
        )
        return inputs

    fold_results = []
    
    for test_session in range(1, 6):
        print(f"\n{'='*60}")
        print(f"[INFO] STARTING FOLD {test_session} (Validation on Session: {test_session})")
        print(f"{'='*60}")
        
        train_ds = dataset["train"].filter(lambda x: x["Session"] != test_session)
        eval_ds  = dataset["train"].filter(lambda x: x["Session"] == test_session)
        
        if DUMMY_RUN:
            train_ds = train_ds.select(range(40))
            eval_ds = eval_ds.select(range(10))

        # Dynamic Class Weighting for the current Fold
        print("[INFO] Computing dynamic class weights for 9 classes...")
        train_labels = train_ds["label"]
        class_weights = compute_class_weight(
            class_weight="balanced", 
            classes=np.unique(train_labels), 
            y=train_labels
        )
        
        # In case a minority class is completely missing from the train fold, pad weights
        full_weights = np.ones(num_classes, dtype=np.float32)
        for idx, cls in enumerate(np.unique(train_labels)):
            full_weights[cls] = class_weights[idx]
        
        print(f"[INFO] Applied Class Weights: {np.round(full_weights, 3)}")

        print("[INFO] Extracting raw audio arrays into Wav2Vec2 input vectors...")
        train_remove_cols = [col for col in train_ds.column_names if col != "label"]
        eval_remove_cols = [col for col in eval_ds.column_names if col != "label"]
        
        train_encoded = train_ds.map(preprocess_function, remove_columns=train_remove_cols, batched=True, batch_size=4)
        eval_encoded  = eval_ds.map(preprocess_function, remove_columns=eval_remove_cols, batched=True, batch_size=4)

        # Initialize the fresh model for the current fold
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            MODEL_ID, config=config, ignore_mismatched_sizes=True, token=HF_TOKEN
        )
        
        # Freezing the CNN feature extractor to save VRAM and avoid overfitting
        model.freeze_feature_encoder()

        fold_output_dir = OUTPUT_DIR / f"fold_{test_session}"
        epochs = 1 if DUMMY_RUN else 15
        
        training_args = TrainingArguments(
            output_dir=str(fold_output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=3e-5,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=4,
            num_train_epochs=epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="uar",
            greater_is_better=True,
            push_to_hub=False,
            fp16=torch.cuda.is_available(), 
            dataloader_num_workers=4 if not DUMMY_RUN else 0,
            report_to="none" 
        )

        trainer = FocalLossTrainer(
            class_weights=full_weights,
            gamma=2.0,
            model=model,
            args=training_args,
            train_dataset=train_encoded,
            eval_dataset=eval_encoded,
            processing_class=feature_extractor,
            data_collator=DataCollatorAudioWithPadding(feature_extractor=feature_extractor),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        print(f"[INFO] Training Fold {test_session} on GPU...")
        trainer.train()
        
        print(f"[INFO] Plotting Learning Curves...")
        plot_learning_curves(trainer.state.log_history, fold_output_dir, test_session)

        print(f"[INFO] Evaluating optimal model for Fold {test_session}...")
        eval_metrics = trainer.evaluate()
        
        # ==========================================
        # SAVE MODEL & TRAINER STATE
        # ==========================================
        print(f"[INFO] Saving best model and trainer state for Fold {test_session}...")
        best_model_path = fold_output_dir / "best_model"
        trainer.save_model(str(best_model_path))
        
        print(f"[INFO] Generating Confusion Matrix...")
        predictions_output = trainer.predict(eval_encoded)
        predicted_labels = np.argmax(predictions_output.predictions, axis=1)
        true_labels = predictions_output.label_ids
        plot_confusion_matrix_heatmap(true_labels, predicted_labels, id2label, fold_output_dir, test_session)

        # -----------------------------------------------------------------
        # DYNAMIC CLEANUP BLOCK: Purge intermediate heavy checkpoints
        # -----------------------------------------------------------------
        import shutil
        print(f"[INFO] Fold {test_session} finished. Purging intermediate checkpoints to save cloud disk space...")
        # Scan and destroy all temporary directories prefixed with "checkpoint-*"
        for checkpoint_dir in fold_output_dir.glob("checkpoint-*"):
            if checkpoint_dir.is_dir():
                shutil.rmtree(checkpoint_dir)
        print(f"[SUCCESS] Cleaned up intermediate checkpoints for Fold {test_session}.")
        # -----------------------------------------------------------------

        print(f"[RESULT] Fold {test_session} Metrics: {eval_metrics}")
        
        fold_results.append({
            "fold": test_session,
            "uar": eval_metrics["eval_uar"],
            "macro_f1": eval_metrics["eval_macro_f1"],
            "accuracy": eval_metrics["eval_accuracy"]
        })
        
        # MLOps Memory Management: Clear GPU VRAM before starting the next fold
        del model, trainer, train_encoded, eval_encoded
        gc.collect()
        torch.cuda.empty_cache()
        
        if DUMMY_RUN and test_session == 2:
            break

    # -----------------------------------------------------------------
    # FINAL RESEARCH METRICS AGGREGATION
    # -----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[SUCCESS] 5-FOLD LOSO CROSS-VALIDATION COMPLETED")
    print(f"{'='*60}")
    
    uars = [res["uar"] for res in fold_results]
    f1s = [res["macro_f1"] for res in fold_results]
    accs = [res["accuracy"] for res in fold_results]
    
    print("Fold-by-Fold Unweighted Average Recall (UAR):", [f"{u:.4f}" for u in uars])
    print(f"Final UAR:      {np.mean(uars):.4f} ± {np.std(uars):.4f}")
    print(f"Final Macro F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Final Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    final_report = {
        "fold_results": fold_results,
        "summary": {
            "mean_uar": float(np.mean(uars)), "std_uar": float(np.std(uars)),
            "mean_macro_f1": float(np.mean(f1s)), "std_macro_f1": float(np.std(f1s)),
            "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs))
        }
    }
    
    report_path = OUTPUT_DIR / "loso_9class_final_metrics.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=4)
        
    print(f"[INFO] All metrics and plots saved at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()