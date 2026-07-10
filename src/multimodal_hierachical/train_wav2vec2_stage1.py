import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import json
import gc
import torch
import numpy as np
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
    Trainer
)
from sklearn.metrics import accuracy_score, recall_score, f1_score

# ==========================================
# 1. ENVIRONMENT & MLOPS CONFIGURATION
# ==========================================
def load_flexible_env():
    """Robustly searches for .env file in current, parent, and grandparent directories."""
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

# ------------------------------------------
# CRITICAL: DUMMY RUN TOGGLE
# True  = Vast.ai Testing (Hits HF Hub, subsamples for quick validation)
# False = Full Cloud Training (Full LOSO Cross-Validation)
# ------------------------------------------
DUMMY_RUN = False  

DATASET_REPO = "HuyPham171/iemocap-sentiment-clean"
MODEL_ID = "facebook/wav2vec2-base"

OUTPUT_DIR = Path("./checkpoints/wav2vec2_stage1").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. METRICS COMPUTATION
# ==========================================
def compute_metrics(eval_pred):
    """Computes Accuracy, Macro F1, and Unweighted Average Recall (UAR)."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    uar = recall_score(labels, predictions, average="macro")
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "uar": uar
    }

# ==========================================
# 3. DATA COLLATOR FOR PADDING
# ==========================================
@dataclass
class DataCollatorAudioWithPadding:
    """Dynamically pads the audio input sequences to the maximum length in a batch."""
    feature_extractor: AutoFeatureExtractor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["label"] for feature in features]

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(label_features, dtype=torch.long)
        return batch

# ==========================================
# 4. MAIN TRAINING PIPELINE (LOSO CV)
# ==========================================
def main():
    print(f"[INFO] Initializing Cloud Training Pipeline...")
    print(f"[INFO] Dummy Run Status: {'ACTIVE (Cloud Pipeline Test)' if DUMMY_RUN else 'INACTIVE (Production Run)'}")
    print(f"[INFO] Target Device: {DEVICE}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, token=HF_TOKEN)
    label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}
    id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        num_labels=3,
        label2id=label2id,
        id2label=id2label,
        finetuning_task="audio-classification",
        token=HF_TOKEN
    )
    
    print(f"[INFO] Connecting to Hugging Face Hub to load: {DATASET_REPO}")
    dataset = load_dataset(DATASET_REPO, token=HF_TOKEN)
    
    if "audio" not in dataset["train"].column_names:
        dataset = dataset.rename_column("file_name", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def preprocess_function(batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            truncation=True,
            max_length=16000 * 15  # Capped at 15 seconds to cover >98% of IEMOCAP durations safely
        )
        return inputs

    # Track metrics across all 5 folds
    fold_results = []
    
    # -----------------------------------------------------------------
    # 5-FOLD LEAVE-ONE-SESSION-OUT (LOSO) CROSS-VALIDATION LOOP
    # -----------------------------------------------------------------
    for test_session in range(1, 6):
        print(f"\n{'='*50}")
        print(f"[INFO] STARTING FOLD {test_session} (Test Session: {test_session})")
        print(f"{'='*50}")
        
        train_ds = dataset["train"].filter(lambda x: x["Session"] != test_session)
        eval_ds  = dataset["train"].filter(lambda x: x["Session"] == test_session)
        
        if DUMMY_RUN:
            print("[WARNING] Subsampling fold data for Dummy Run...")
            train_ds = train_ds.select(range(40))
            eval_ds = eval_ds.select(range(10))

        print("[INFO] Extracting raw audio arrays into Wav2Vec2 input vectors...")
        train_remove_cols = [col for col in train_ds.column_names if col != "label"]
        eval_remove_cols = [col for col in eval_ds.column_names if col != "label"]
        
        train_encoded = train_ds.map(preprocess_function, remove_columns=train_remove_cols, batched=True, batch_size=4)
        eval_encoded  = eval_ds.map(preprocess_function, remove_columns=eval_remove_cols, batched=True, batch_size=4)

        # Re-initialize model inside the loop to reset weights for each fold
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            MODEL_ID,
            config=config,
            ignore_mismatched_sizes=True,
            token=HF_TOKEN
        )
        model.freeze_feature_encoder()

        fold_output_dir = OUTPUT_DIR / f"fold_{test_session}"
        epochs = 1 if DUMMY_RUN else 5
        
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

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_encoded,
            eval_dataset=eval_encoded,
            processing_class=feature_extractor,
            data_collator=DataCollatorAudioWithPadding(feature_extractor=feature_extractor),
            compute_metrics=compute_metrics
        )

        print(f"[INFO] Training Fold {test_session}...")
        trainer.train()
        
        print(f"[INFO] Evaluating optimal model for Fold {test_session}...")
        eval_metrics = trainer.evaluate()
        print(f"[RESULT] Fold {test_session} Metrics: {eval_metrics}")
        
        fold_results.append({
            "fold": test_session,
            "uar": eval_metrics["eval_uar"],
            "macro_f1": eval_metrics["eval_macro_f1"],
            "accuracy": eval_metrics["eval_accuracy"]
        })
        
        # Free GPU memory before starting the next fold to prevent OOM
        del model, trainer, train_encoded, eval_encoded
        gc.collect()
        torch.cuda.empty_cache()
        
        if DUMMY_RUN and test_session == 2:
            print("[INFO] Dummy Run mode limits execution to 2 folds. Exiting loop.")
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