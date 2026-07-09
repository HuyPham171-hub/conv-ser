import os
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
# True  = Vast.ai Testing (Hits HF Hub, subsamples 50 files, 1 Epoch)
# False = Full Cloud Training (Full data, Full Epochs)
# ------------------------------------------
DUMMY_RUN = True  

# Repository configurations
DATASET_REPO = "HuyPham171/iemocap-sentiment-clean"
MODEL_ID = "facebook/wav2vec2-base"

# Cloud-safe relative directory for checkpoints
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
class DataCollatorCTCWithPadding:
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
# 4. MAIN TRAINING PIPELINE
# ==========================================
def main():
    print(f"[INFO] Initializing Cloud Training Pipeline...")
    print(f"[INFO] Dummy Run Status: {'ACTIVE (Cloud Pipeline Test)' if DUMMY_RUN else 'INACTIVE (Production Run)'}")
    print(f"[INFO] Target Device: {DEVICE}")

    # 4.1 Load Feature Extractor & Architecture Config
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
    
    # 4.2 Unified Dataset Loading via Hugging Face Hub
    print(f"[INFO] Connecting to Hugging Face Hub to load: {DATASET_REPO}")
    dataset = load_dataset(DATASET_REPO, token=HF_TOKEN)
    
    # Standardize column structure for AudioFolder format
    if "audio" not in dataset["train"].column_names:
        dataset = dataset.rename_column("file_name", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # -----------------------------------------------------------------
    # CLOUD DUMMY RUN LOGIC: Validates Hub Token & Pipeline without heavy downloading
    # -----------------------------------------------------------------
    if DUMMY_RUN:
        print("[WARNING] Dummy Run Active on Cloud! Subsampling 50 items from downloaded Hub data...")
        dummy_ds = dataset["train"].select(range(50))
        train_ds = dummy_ds.select(range(0, 40))
        eval_ds = dummy_ds.select(range(40, 50))
    else:
        # Standard Leave-One-Session-Out (LOSO) Split Strategy for Full Production
        print("[INFO] Applying LOSO Strategy (Train=Sessions 1-4, Eval=Session 5)...")
        train_ds = dataset["train"].filter(lambda x: x["Session"] != 5)
        eval_ds  = dataset["train"].filter(lambda x: x["Session"] == 5)
    # -----------------------------------------------------------------

    # 4.3 Preprocessing Map Function
    def preprocess_function(batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            truncation=True,
            max_length=16000 * 10  # Cap at 10 seconds to protect VRAM
        )
        return inputs

    print("[INFO] Extracting raw audio arrays into Wav2Vec2 input vectors...")
    # Isolate non-tensor columns dynamically to keep the dataset clean
    train_remove_cols = [col for col in train_ds.column_names if col != "label"]
    eval_remove_cols = [col for col in eval_ds.column_names if col != "label"]
    
    train_ds = train_ds.map(preprocess_function, remove_columns=train_remove_cols, batched=True, batch_size=4)
    eval_ds  = eval_ds.map(preprocess_function, remove_columns=eval_remove_cols, batched=True, batch_size=4)

    # 4.4 Model Initialization
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID,
        config=config,
        ignore_mismatched_sizes=True,
        token=HF_TOKEN
    )
    
    # Freeze CNN layers for pure acoustic feature extraction
    model.freeze_feature_encoder()

    # 4.5 Training Arguments
    epochs = 1 if DUMMY_RUN else 5
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
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
        dataloader_num_workers=4 if not DUMMY_RUN else 0, # Multi-processing enabled for Linux production
    )

    # 4.6 Trainer Construction
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=feature_extractor,
        data_collator=DataCollatorCTCWithPadding(feature_extractor=feature_extractor),
        compute_metrics=compute_metrics
    )

    # 4.7 Execution
    print(f"\n[INFO] Starting Trainer...")
    trainer.train()
    
    print("\n[INFO] Evaluating optimal model...")
    eval_results = trainer.evaluate()
    print(f"[SUCCESS] Cloud Pipeline Finished Successfully. Eval Results: {eval_results}")

if __name__ == "__main__":
    main()