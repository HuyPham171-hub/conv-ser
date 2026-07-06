"""
Separate scr_noise BiGRU baselines.

This script does not touch the existing clean precomputed-embedding baseline in:
    src/training/train_bigru.py

It trains BiGRU from raw IEMOCAP audio sequences:

raw audio sequence
-> DataLoader
-> optional random AWGN during training only
-> Wav2Vec2 extracts embeddings on the fly
-> BiGRU

Examples:
    python src/training/scr_noise/train_bigru.py --mode flat8 --baseline clean
    python src/training/scr_noise/train_bigru.py --mode stage1 --baseline clean
    python src/training/scr_noise/train_bigru.py --mode stage2 --baseline clean

    python src/training/scr_noise/train_bigru.py --mode flat8 --baseline dynamic-awgn
    python src/training/scr_noise/train_bigru.py --mode stage1 --baseline dynamic-awgn
    python src/training/scr_noise/train_bigru.py --mode stage2 --baseline dynamic-awgn

    python src/training/scr_noise/train_bigru.py --run-all-baselines
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bigru_stage3 import ConversationalBiGRU
from src.training.scr_noise.dataset import (
    DEFAULT_FLAT8_LABELS,
    IEMOCAPRawSequenceDataset,
    raw_sequence_collate,
)


ENV_PATH = PROJECT_ROOT / "src" / ".env"
load_dotenv(ENV_PATH)


def parse_csv_floats(value):
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_csv_strings(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def get_device(requested_device):
    requested_device = requested_device.lower()

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE=cuda was requested, but PyTorch cannot see a CUDA GPU."
            )
        return torch.device("cuda")

    if requested_device == "cpu":
        return torch.device("cpu")

    raise ValueError("device must be one of: auto, cuda, cpu")


def load_wav2vec2_encoder(model_dir, device):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"WAV2VEC2 model directory does not exist: {model_dir}")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    config_path = model_dir / "config.json"
    if config_path.exists():
        config = Wav2Vec2Config.from_pretrained(str(model_dir), local_files_only=True)
    else:
        print(f"[WARNING] No config.json found in {model_dir}; using default Wav2Vec2Config.")
        config = Wav2Vec2Config()

    wav2vec2 = Wav2Vec2Model(config)

    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path))
        print("[INFO] Loading Wav2Vec2 weights from model.safetensors")
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
        print("[INFO] Loading Wav2Vec2 weights from pytorch_model.bin")
    else:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin found in {model_dir}"
        )

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("base.wav2vec2."):
            cleaned_state_dict[key.replace("base.wav2vec2.", "")] = value
        elif key.startswith("base.") and not any(
            head in key for head in ["classifier", "projector"]
        ):
            cleaned_state_dict[key.replace("base.", "")] = value
        elif key.startswith("wav2vec2."):
            cleaned_state_dict[key.replace("wav2vec2.", "")] = value
        elif any(head in key for head in ["sent_head", "classifier", "projector"]):
            continue
        else:
            cleaned_state_dict[key] = value

    msg = wav2vec2.load_state_dict(cleaned_state_dict, strict=False)
    print(f"[INFO] Wav2Vec2 load status: {msg}")

    wav2vec2 = wav2vec2.to(device)
    wav2vec2.eval()
    for param in wav2vec2.parameters():
        param.requires_grad = False

    return processor, wav2vec2


def encode_batch_waveforms(batch, processor, wav2vec2, device, embedding_dim=768):
    """
    Convert a collated raw waveform batch into BiGRU input:
        [batch_size, sequence_len=3, embedding_dim=768]
    """
    batch_size = len(batch["waveforms"])
    seq_len = len(batch["waveforms"][0])

    embeddings = torch.zeros(batch_size, seq_len, embedding_dim, dtype=torch.float32)
    flat_waveforms = []
    flat_positions = []

    for batch_idx, window in enumerate(batch["waveforms"]):
        for seq_idx, waveform in enumerate(window):
            if waveform is None:
                continue

            flat_positions.append((batch_idx, seq_idx))
            flat_waveforms.append(waveform)

    if not flat_waveforms:
        return embeddings

    inputs = processor(
        flat_waveforms,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = wav2vec2(input_values)

    pooled = outputs.last_hidden_state.mean(dim=1).detach().cpu()

    for (batch_idx, seq_idx), embedding in zip(flat_positions, pooled):
        embeddings[batch_idx, seq_idx] = embedding

    return embeddings


def get_loso_indices(dataset, test_session):
    train_indices = []
    test_indices = []

    for idx, sample in enumerate(dataset.samples):
        if sample["session"] == test_session:
            test_indices.append(idx)
        else:
            train_indices.append(idx)

    return train_indices, test_indices


def train_epoch(model, dataloader, criterion, optimizer, processor, wav2vec2, device):
    model.train()
    total_loss = 0.0
    valid_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch_x = encode_batch_waveforms(batch, processor, wav2vec2, device).to(device)
        batch_y = torch.tensor(batch["labels"], dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1

    return total_loss / valid_batches if valid_batches else 0.0


def evaluate(model, dataloader, criterion, processor, wav2vec2, device):
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch_x = encode_batch_waveforms(batch, processor, wav2vec2, device).to(device)
            batch_y = torch.tensor(batch["labels"], dtype=torch.long, device=device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item()
            valid_batches += 1
            all_preds.append(torch.argmax(logits, dim=1).detach().cpu())
            all_targets.append(batch_y.detach().cpu())

    if not all_targets:
        return 0.0, 0.0, 0.0

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    return (
        total_loss / valid_batches if valid_batches else 0.0,
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
    )


def run_single_baseline(args, mode, baseline):
    device = get_device(args.device)
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    metadata_csv_path = Path(args.metadata_csv_path)
    iemocap_root_dir = Path(args.iemocap_root_dir)
    checkpoint_dir = Path(args.checkpoint_dir) / baseline / mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    awgn_enabled = baseline == "dynamic-awgn"

    print("\n" + "=" * 70)
    print(f"[BASELINE] {baseline} | [MODE] {mode}")
    print(f"[INFO] Metadata: {metadata_csv_path}")
    print(f"[INFO] IEMOCAP root: {iemocap_root_dir}")
    if awgn_enabled:
        print(
            f"[INFO] Dynamic AWGN during training only | "
            f"prob={args.awgn_prob} | snr_choices={args.awgn_snr_choices}"
        )
    else:
        print("[INFO] Clean raw-audio baseline: no AWGN")

    processor, wav2vec2 = load_wav2vec2_encoder(args.wav2vec2_model_dir, device)

    train_dataset = IEMOCAPRawSequenceDataset(
        metadata_csv_path=metadata_csv_path,
        iemocap_root_dir=iemocap_root_dir,
        mode=mode,
        training=True,
        awgn_enabled=awgn_enabled,
        awgn_prob=args.awgn_prob,
        awgn_snr_choices=args.awgn_snr_choices,
        flat8_labels=args.flat8_labels,
    )
    eval_dataset = IEMOCAPRawSequenceDataset(
        metadata_csv_path=metadata_csv_path,
        iemocap_root_dir=iemocap_root_dir,
        mode=mode,
        training=False,
        awgn_enabled=False,
        flat8_labels=args.flat8_labels,
    )

    print(f"[INFO] Label names: {train_dataset.label_names}")
    print(f"[INFO] Samples: {len(train_dataset)}")

    fold_acc_scores = []
    fold_f1_scores = []

    for test_session in range(1, 6):
        print("\n" + "-" * 70)
        print(f"[LOSO] Fold {test_session}: test Session {test_session}")

        train_idx, test_idx = get_loso_indices(eval_dataset, test_session)
        train_subset = Subset(train_dataset, train_idx)
        test_subset = Subset(eval_dataset, test_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=raw_sequence_collate,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=raw_sequence_collate,
            num_workers=args.num_workers,
        )

        print(f"[INFO] Train samples: {len(train_subset)} | Test samples: {len(test_subset)}")

        model = ConversationalBiGRU(num_classes=train_dataset.num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        best_f1 = 0.0
        best_acc = 0.0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, processor, wav2vec2, device
            )
            val_loss, val_acc, val_f1 = evaluate(
                model, test_loader, criterion, processor, wav2vec2, device
            )

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                checkpoint_path = checkpoint_dir / f"{mode}_{baseline}_fold_{test_session}.pth"
                torch.save(model.state_dict(), checkpoint_path)

        print(
            f"[FOLD {test_session}] Best Acc: {best_acc:.4f} | "
            f"Best Macro F1: {best_f1:.4f}"
        )
        fold_acc_scores.append(best_acc)
        fold_f1_scores.append(best_f1)

    print("\n" + "#" * 70)
    print(f"[FINAL] {baseline} | {mode}")
    print(f"Average Accuracy : {np.mean(fold_acc_scores):.4f} ± {np.std(fold_acc_scores):.4f}")
    print(f"Average Macro F1 : {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
    print(f"[CHECKPOINTS] {checkpoint_dir}")
    print("#" * 70)


def build_arg_parser():
    metadata_default = Path(os.getenv("OUTPUT_DIR", "")) / "iemocap_metadata.csv"

    parser = argparse.ArgumentParser(description="Separate scr_noise BiGRU baselines")
    parser.add_argument("--mode", choices=["flat8", "stage1", "stage2"], default="stage1")
    parser.add_argument("--baseline", choices=["clean", "dynamic-awgn"], default="clean")
    parser.add_argument(
        "--run-all-baselines",
        action="store_true",
        help="Run clean and dynamic-AWGN for flat8, stage1, and stage2.",
    )

    parser.add_argument("--iemocap-root-dir", default=os.getenv("IEMOCAP_ROOT_DIR"))
    parser.add_argument("--metadata-csv-path", default=str(metadata_default))
    parser.add_argument("--wav2vec2-model-dir", default=os.getenv("WAV2VEC2_MODEL_DIR"))
    parser.add_argument(
        "--checkpoint-dir",
        default=os.getenv(
            "SCR_NOISE_CHECKPOINT_DIR",
            str(PROJECT_ROOT / "checkpoints" / "scr_noise_bigru"),
        ),
    )
    parser.add_argument("--device", default=os.getenv("DEVICE", "auto"))

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--awgn-prob",
        type=float,
        default=float(os.getenv("DYNAMIC_AWGN_PROB", "0.5")),
    )
    parser.add_argument(
        "--awgn-snr-choices",
        type=parse_csv_floats,
        default=parse_csv_floats(os.getenv("DYNAMIC_AWGN_SNR_CHOICES", "10,15,20")),
    )
    parser.add_argument(
        "--flat8-labels",
        type=parse_csv_strings,
        default=parse_csv_strings(
            os.getenv("FLAT8_LABELS", ",".join(DEFAULT_FLAT8_LABELS))
        ),
        help=(
            "Comma-separated Raw_Emotion labels used for flat8. "
            "Default: ang,sad,hap,exc,neu,fru,fea,sur"
        ),
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.iemocap_root_dir:
        raise ValueError("Missing --iemocap-root-dir or IEMOCAP_ROOT_DIR in src/.env")
    if not args.wav2vec2_model_dir:
        raise ValueError("Missing --wav2vec2-model-dir or WAV2VEC2_MODEL_DIR in src/.env")

    if args.run_all_baselines:
        for mode in ("flat8", "stage1", "stage2"):
            for baseline in ("clean", "dynamic-awgn"):
                run_single_baseline(args, mode=mode, baseline=baseline)
        return

    run_single_baseline(args, mode=args.mode, baseline=args.baseline)


if __name__ == "__main__":
    main()
