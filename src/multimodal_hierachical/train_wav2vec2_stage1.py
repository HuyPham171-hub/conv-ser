import os
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
    current_dir = Path(__file__).resolve().parent
    for check_dir in [current_dir, current_dir.parent, current_dir.parent.parent]:
        env_file = check_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return env_file
    return None

env_resolved_path = load_flexible_env()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(f"[ERROR] HF_TOKEN is not found in environment or .env file.")

DUMMY_RUN = False  

DATASET_REPO = "HuyPham171/iemocap-sentiment-clean"
MODEL_ID = "facebook/wav2vec2-base"

OUTPUT_DIR = Path("./checkpoints/wav2vec2_stage1").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. CUSTOM TRAINER FOR WEIGHTED LOSS
# ==========================================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Chuyển class_weights vào GPU cùng với model
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ==========================================
# 3. METRICS & PLOTTING FUNCTIONS
# ==========================================
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    uar = recall_score(labels, predictions, average="macro")
    
    return {"accuracy": acc, "macro_f1": macro_f1, "uar": uar}

def plot_learning_curves(log_history, output_dir, fold):
    train_loss = [x["loss"] for x in log_history if "loss" in x]
    eval_loss = [x["eval_loss"] for x in log_history if "eval_loss" in x]
    
    # Bước nhảy xấp xỉ theo số epoch
    epochs_train = [x["epoch"] for x in log_history if "loss" in x]
    epochs_eval = [x["epoch"] for x in log_history if "eval_loss" in x]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_train, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs_eval, eval_loss, label="Validation Loss", marker="x")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Learning Curves - Fold {fold}")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "learning_curves.png")
    plt.close()

def plot_confusion_matrix_heatmap(labels, predictions, id2label, output_dir, fold):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[id2label[i] for i in range(len(id2label))],
                yticklabels=[id2label[i] for i in range(len(id2label))])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

# ==========================================
# 4. DATA COLLATOR
# ==========================================
@dataclass
class DataCollatorAudioWithPadding:
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
    print(f"[INFO] Initializing Cloud Training Pipeline...")
    print(f"[INFO] Dummy Run Status: {'ACTIVE (Cloud Pipeline Test)' if DUMMY_RUN else 'INACTIVE (Production Run)'}")
    print(f"[INFO] Target Device: {DEVICE}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, token=HF_TOKEN)
    label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}
    id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    config = AutoConfig.from_pretrained(
        MODEL_ID, num_labels=3, label2id=label2id, id2label=id2label,
        finetuning_task="audio-classification", token=HF_TOKEN
    )
    
    print(f"[INFO] Connecting to Hugging Face Hub to load: {DATASET_REPO}")
    dataset = load_dataset(DATASET_REPO, token=HF_TOKEN)
    
    if "audio" not in dataset["train"].column_names:
        dataset = dataset.rename_column("file_name", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def preprocess_function(batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        inputs = feature_extractor(
            audio_arrays, sampling_rate=feature_extractor.sampling_rate,
            truncation=True, max_length=16000 * 15 
        )
        return inputs

    fold_results = []
    
    for test_session in range(1, 6):
        print(f"\n{'='*50}")
        print(f"[INFO] STARTING FOLD {test_session} (Test Session: {test_session})")
        print(f"{'='*50}")
        
        train_ds = dataset["train"].filter(lambda x: x["Session"] != test_session)
        eval_ds  = dataset["train"].filter(lambda x: x["Session"] == test_session)
        
        if DUMMY_RUN:
            train_ds = train_ds.select(range(40))
            eval_ds = eval_ds.select(range(10))

        # Tính toán Class Weights động cho Fold hiện tại
        print("[INFO] Computing dynamic class weights to handle imbalance...")
        train_labels = train_ds["label"]
        class_weights = compute_class_weight(
            class_weight="balanced", 
            classes=np.unique(train_labels), 
            y=train_labels
        )
        print(f"[INFO] Applied Class Weights: {class_weights}")

        print("[INFO] Extracting raw audio arrays into Wav2Vec2 input vectors...")
        train_remove_cols = [col for col in train_ds.column_names if col != "label"]
        eval_remove_cols = [col for col in eval_ds.column_names if col != "label"]
        
        train_encoded = train_ds.map(preprocess_function, remove_columns=train_remove_cols, batched=True, batch_size=4)
        eval_encoded  = eval_ds.map(preprocess_function, remove_columns=eval_remove_cols, batched=True, batch_size=4)

        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            MODEL_ID, config=config, ignore_mismatched_sizes=True, token=HF_TOKEN
        )
        model.freeze_feature_encoder()

        fold_output_dir = OUTPUT_DIR / f"fold_{test_session}"
        epochs = 1 if DUMMY_RUN else 15
        
        training_args = TrainingArguments(
            output_dir=str(fold_output_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
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
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_encoded,
            eval_dataset=eval_encoded,
            processing_class=feature_extractor,
            data_collator=DataCollatorAudioWithPadding(feature_extractor=feature_extractor),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Early Stopping kích hoạt
        )

        print(f"[INFO] Training Fold {test_session}...")
        trainer.train()
        
        print(f"[INFO] Plotting Learning Curves...")
        plot_learning_curves(trainer.state.log_history, fold_output_dir, test_session)

        print(f"[INFO] Evaluating optimal model for Fold {test_session}...")
        eval_metrics = trainer.evaluate()
        
        print(f"[INFO] Generating Confusion Matrix...")
        predictions_output = trainer.predict(eval_encoded)
        predicted_labels = np.argmax(predictions_output.predictions, axis=1)
        true_labels = predictions_output.label_ids
        plot_confusion_matrix_heatmap(true_labels, predicted_labels, id2label, fold_output_dir, test_session)

        print(f"[RESULT] Fold {test_session} Metrics: {eval_metrics}")
        
        fold_results.append({
            "fold": test_session,
            "uar": eval_metrics["eval_uar"],
            "macro_f1": eval_metrics["eval_macro_f1"],
            "accuracy": eval_metrics["eval_accuracy"]
        })
        
        del model, trainer, train_encoded, eval_encoded
        gc.collect()
        torch.cuda.empty_cache()
        
        if DUMMY_RUN and test_session == 2:
            break

    # -----------------------------------------------------------------
    # FINAL RESEARCH METRICS AGGREGATION
    # -----------------------------------------------------------------
    print(f"\n{'='*50}")
    print("[SUCCESS] 5-FOLD LOSO CROSS-VALIDATION COMPLETED")
    print(f"{'='*50}")
    
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
    
    report_path = OUTPUT_DIR / "loso_final_metrics.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=4)
        
    print(f"[INFO] All results saved at: {report_path}")

if __name__ == "__main__":
    main()