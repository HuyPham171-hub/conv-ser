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
# Resolve .env path (Assuming script is in src/multimodal_hierarchical/)
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(f"[ERROR] HF_TOKEN is not set in {ENV_PATH}")

# ------------------------------------------
# CRITICAL: DUMMY RUN TOGGLE
# True  = Local testing (Subsamples data, 1 Epoch, prevents crashes)
# False = Cloud deployment (Full data, Full Epochs)
# ------------------------------------------
DUMMY_RUN = True  

# Repository configurations
DATASET_REPO = "HuyPham171/iemocap-sentiment-clean" # Update with your HF username
MODEL_ID = "facebook/wav2vec2-base"
OUTPUT_DIR = Path("../../checkpoints/wav2vec2_stage1").resolve()

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
    print(f"[INFO] Initializing Training Pipeline...")
    print(f"[INFO] Dummy Run Status: {'ACTIVE (Local Testing)' if DUMMY_RUN else 'INACTIVE (Full Training)'}")
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
    
    # 4.2 Load Dataset from Hugging Face Hub
    print(f"[INFO] Downloading/Loading dataset from Hub: {DATASET_REPO}")
    dataset = load_dataset(DATASET_REPO, token=HF_TOKEN)
    
    # HF datasets automatically detects the 'audio' column. We enforce 16kHz sampling rate.
    if "audio" not in dataset["train"].column_names:
        dataset = dataset.rename_column("file_name", "audio")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Leave-One-Session-Out (LOSO) Split Strategy: Fold 1 (Session 5 as Test)
    print("[INFO] Splitting dataset (LOSO Fold 1: Train=Ses1-4, Eval=Ses5)...")
    train_ds = dataset["train"].filter(lambda x: x["Session"] != 5)
    eval_ds  = dataset["train"].filter(lambda x: x["Session"] == 5)

    # ------------------------------------------
    # EXECUTE DUMMY RUN SUBSAMPLING
    # ------------------------------------------
    if DUMMY_RUN:
        train_ds = train_ds.select(range(min(40, len(train_ds))))
        eval_ds = eval_ds.select(range(min(10, len(eval_ds))))
        print(f"[WARNING] Subsampled for Dummy Run: {len(train_ds)} train | {len(eval_ds)} eval")

    # 4.3 Preprocessing Map Function
    def preprocess_function(batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            truncation=True,
            max_length=16000 * 10  # Truncate at 10 seconds to prevent OOM
        )
        return inputs

    print("[INFO] Extracting raw audio arrays into Wav2Vec2 input vectors...")
    train_ds = train_ds.map(preprocess_function, remove_columns=["audio", "Utterance_ID", "Session"], batched=True, batch_size=4)
    eval_ds  = eval_ds.map(preprocess_function, remove_columns=["audio", "Utterance_ID", "Session"], batched=True, batch_size=4)

    # 4.4 Model Initialization
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID,
        config=config,
        ignore_mismatched_sizes=True,
        token=HF_TOKEN
    )
    
    # Freeze CNN feature extractor layers (Static Acoustic Features)
    model.freeze_feature_extractor()

    # 4.5 Training Arguments
    epochs = 1 if DUMMY_RUN else 5  # Switch epochs dynamically
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        evaluation_strategy="epoch",
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
        fp16=torch.cuda.is_available(), # Mixed precision if GPU is present
        dataloader_num_workers=4 if not DUMMY_RUN else 0, # Windows local needs 0
    )

    # 4.6 Trainer Construction
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=feature_extractor,
        data_collator=DataCollatorCTCWithPadding(feature_extractor=feature_extractor),
        compute_metrics=compute_metrics
    )

    # 4.7 Execution
    print(f"\n[INFO] Starting Trainer...")
    trainer.train()
    
    print("\n[INFO] Evaluating optimal model...")
    eval_results = trainer.evaluate()
    print(f"[SUCCESS] Pipeline Finished. Eval Results: {eval_results}")

if __name__ == "__main__":
    main()