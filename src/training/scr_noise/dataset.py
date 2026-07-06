"""
Raw-audio conversational IEMOCAP dataset for the separate scr_noise baseline.

This dataset is for:

raw audio sequence
-> DataLoader
-> random AWGN during training only
-> Wav2Vec2 embeddings extracted on the fly
-> BiGRU

It intentionally does not use precomputed .npy embeddings.
"""

import random
from pathlib import Path

import librosa
import pandas as pd
from torch.utils.data import Dataset

from src.data.augmentation import AudioAugmentor


DEFAULT_FLAT8_LABELS = ("ang", "sad", "hap", "exc", "neu", "fru", "fea", "sur")


def build_iemocap_wav_index(iemocap_root_dir):
    iemocap_root_dir = Path(iemocap_root_dir)
    return {
        wav_path.stem: wav_path
        for wav_path in iemocap_root_dir.rglob("sentences/wav/*/*.wav")
    }


def resolve_mode_config(mode, flat8_labels=DEFAULT_FLAT8_LABELS):
    mode = mode.lower()

    if mode == "flat8":
        label_to_id = {label: idx for idx, label in enumerate(flat8_labels)}
        return {
            "mode": mode,
            "num_classes": len(label_to_id),
            "label_names": list(flat8_labels),
            "label_to_id": label_to_id,
        }

    if mode == "stage1":
        return {
            "mode": mode,
            "num_classes": 3,
            "label_names": ["negative", "neutral", "positive"],
            "label_column": "Stage1_Label",
        }

    if mode == "stage2":
        return {
            "mode": mode,
            "num_classes": 5,
            "label_names": ["ang", "sad", "fea", "dis", "fru"],
            "label_column": "Stage2_Label",
        }

    raise ValueError("mode must be one of: flat8, stage1, stage2")


class IEMOCAPRawSequenceDataset(Dataset):
    """
    Builds conversational windows from raw IEMOCAP audio.

    For each sample, returns a 3-turn waveform sequence:
        [U_t-2, U_t-1, U_t]

    Missing context at the start of a dialogue is padded with None. The training
    loop turns those None values into zero 768-D embeddings after Wav2Vec2.
    """

    def __init__(
        self,
        metadata_csv_path,
        iemocap_root_dir,
        mode,
        training=False,
        awgn_enabled=False,
        awgn_prob=0.5,
        awgn_snr_choices=(10, 15, 20),
        sr=16000,
        flat8_labels=DEFAULT_FLAT8_LABELS,
    ):
        super().__init__()
        self.metadata_csv_path = Path(metadata_csv_path)
        self.iemocap_root_dir = Path(iemocap_root_dir)
        self.mode = mode.lower()
        self.training = training
        self.awgn_enabled = awgn_enabled
        self.awgn_prob = awgn_prob
        self.awgn_snr_choices = tuple(float(value) for value in awgn_snr_choices)
        self.sr = sr
        self.augmentor = AudioAugmentor(sr=sr)
        self.mode_config = resolve_mode_config(self.mode, flat8_labels=flat8_labels)

        df = pd.read_csv(self.metadata_csv_path)
        df = df.sort_values(by=["Session", "Dialog_ID", "Turn_Order"]).reset_index(drop=True)

        wav_index = build_iemocap_wav_index(self.iemocap_root_dir)
        df["wav_path"] = df["Utterance_ID"].map(wav_index)
        df = df.dropna(subset=["wav_path"]).reset_index(drop=True)

        self.samples = self._build_samples(df)
        if not self.samples:
            raise ValueError(
                "No usable samples found. Check metadata path, IEMOCAP root, and mode labels."
            )

    @property
    def num_classes(self):
        return self.mode_config["num_classes"]

    @property
    def label_names(self):
        return self.mode_config["label_names"]

    def _label_for_row(self, row):
        if self.mode == "flat8":
            return self.mode_config["label_to_id"].get(row["Raw_Emotion"], -1)

        return int(row[self.mode_config["label_column"]])

    def _build_samples(self, df):
        samples = []

        for _, group in df.groupby("Dialog_ID", sort=False):
            group = group.sort_values("Turn_Order").reset_index(drop=True)

            for idx, row in group.iterrows():
                target_label = self._label_for_row(row)
                if target_label == -1:
                    continue

                window_rows = []
                for prev_idx in (idx - 2, idx - 1, idx):
                    if prev_idx < 0:
                        window_rows.append(None)
                    else:
                        window_rows.append(group.iloc[prev_idx])

                samples.append(
                    {
                        "utterance_id": row["Utterance_ID"],
                        "session": int(row["Session"]),
                        "label": target_label,
                        "window_rows": window_rows,
                    }
                )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        waveforms = []
        awgn_mask = []
        snr_values = []

        for row in sample["window_rows"]:
            if row is None:
                waveforms.append(None)
                awgn_mask.append(False)
                snr_values.append(None)
                continue

            waveform, _ = librosa.load(row["wav_path"], sr=self.sr)
            should_apply_awgn = (
                self.training
                and self.awgn_enabled
                and random.random() < self.awgn_prob
            )

            if should_apply_awgn:
                snr_db = random.choice(self.awgn_snr_choices)
                waveform = self.augmentor.add_awgn(waveform, snr_db=snr_db)
                awgn_mask.append(True)
                snr_values.append(snr_db)
            else:
                awgn_mask.append(False)
                snr_values.append(None)

            waveforms.append(waveform)

        return {
            "utterance_id": sample["utterance_id"],
            "session": sample["session"],
            "label": sample["label"],
            "waveforms": waveforms,
            "awgn_mask": awgn_mask,
            "snr_values": snr_values,
        }


def raw_sequence_collate(batch):
    return {
        "utterance_ids": [item["utterance_id"] for item in batch],
        "sessions": [item["session"] for item in batch],
        "labels": [item["label"] for item in batch],
        "waveforms": [item["waveforms"] for item in batch],
        "awgn_mask": [item["awgn_mask"] for item in batch],
        "snr_values": [item["snr_values"] for item in batch],
    }
