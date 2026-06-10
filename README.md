# Speech Emotion & Sentiment Recognition (SER) workspace

End-to-end pipeline for:
- 8-way Speech Emotion Recognition (SER) with Wav2Vec2 and CNN baselines.
- 3-way Sentiment (Negative, Neutral, Positive) using a sentiment head on top of the SER model and via emotion→sentiment mapping.

Key notebooks and scripts:
- Data conversion and preprocessing: [to_ravdess_format.ipynb](to_ravdess_format.ipynb), [preprocessing.ipynb](preprocessing.ipynb)
- Feature extraction (MFCC, Log-Mel, spectrogram images): [extract_features.ipynb](extract_features.ipynb)
- Wav2Vec2 training + sentiment head: [wav2vec2_emo.ipynb](wav2vec2_emo.ipynb)
- CNN baselines: [Scipts/cnn_overlap.ipynb](Scipts/cnn_overlap.ipynb), [Scipts/cnn_rnn_emos.ipynb](Scipts/cnn_rnn_emos.ipynb)
- Simple UI tests (voice wake word demo): [test.py](test.py), [test2.py](test2.py)

Models are saved to:
- 8-emotion Wav2Vec2: [Models/wav2vec2/final-model](Models/wav2vec2/final-model)
- 3-sentiment head weights: [Models/wav2vec2_sentiment/final/model.safetensors](Models/wav2vec2_sentiment/final/model.safetensors)

Data lives under:
- Original/converted audio: [Data/Dataset/](Data/Dataset/)
- Preprocessed audio: [Data/Preprocessed_data/](Data/Preprocessed_data/)
- DataFrames/CSVs: [Data/DataFrame/](Data/DataFrame/)
- Spectrogram metadata: [Data/spectrogram_metadata.csv](Data/spectrogram_metadata.csv)

## Setup

- Python 3.12 (see [requirements.txt](requirements.txt))

```sh
pip install -r requirements.txt
```

Recommended: run notebooks in VS Code and use the provided cells.

## Pipeline overview

1) Convert external datasets to RAVDESS-like structure
- Use [to_ravdess_format.ipynb](to_ravdess_format.ipynb) to normalize TESS, SAVEE, CREMA-D into RAVDESS-style filenames/folders under [Data/Dataset/](Data/Dataset/).

2) Preprocess audio (trim, denoise, normalize)
- Run [preprocessing.ipynb](preprocessing.ipynb) to create cleaned wavs in [Data/Preprocessed_data/](Data/Preprocessed_data/).

3) Build a master dataframe and speaker-independent splits
- In [wav2vec2_emo.ipynb](wav2vec2_emo.ipynb):
  - Combine sources → [Data/DataFrame/preprocessed_df.csv](Data/DataFrame/preprocessed_df.csv)
  - Split by actor per dataset → [Data/DataFrame/train_ser.csv](Data/DataFrame/train_ser.csv), [Data/DataFrame/val_ser.csv](Data/DataFrame/val_ser.csv), [Data/DataFrame/test_ser.csv](Data/DataFrame/test_ser.csv)

4) Feature extraction (optional for CNN baselines)
- Use [extract_features.ipynb](extract_features.ipynb) to create:
  - MFCC/Log-Mel arrays under [Features/](Features/)
  - Full-resolution spectrogram PNGs and [Data/DataFrame/spectrogram_metadata.csv](Data/DataFrame/spectrogram_metadata.csv)

## Training and evaluation

### A) Wav2Vec2 — 8 emotions

- In [wav2vec2_emo.ipynb](wav2vec2_emo.ipynb):
  - Load splits into Hugging Face Datasets
  - Tokenize with `AutoFeatureExtractor("facebook/wav2vec2-base")`
  - Train `AutoModelForAudioClassification` and save to [Models/wav2vec2/final-model](Models/wav2vec2/final-model)
  - Evaluate on test split with classification report and confusion matrix

Inference example (single .wav → emotion):

```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import librosa, torch, numpy as np

model_id = "Models/wav2vec2/final-model"
extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = AutoModelForAudioClassification.from_pretrained(model_id).eval()

def predict_emotion(wav_path):
    audio, _ = librosa.load(wav_path, sr=extractor.sampling_rate)
    inputs = extractor(audio, sampling_rate=extractor.sampling_rate, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=int(extractor.sampling_rate*4.5))
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = int(torch.argmax(logits, dim=-1))
    return model.config.id2label[pred]
```

### B) Sentiment head (Negative/Neutral/Positive)

- Dataset: [Data/DataFrame/sentiment_balanced.csv](Data/DataFrame/sentiment_balanced.csv) → mapped to labels {0,1,2}
- In [wav2vec2_emo.ipynb](wav2vec2_emo.ipynb):
  - Replace raw paths with preprocessed paths
  - Build HF Dataset from `path,label`
  - Define `Wav2Vec2Sentiment` wrapping the frozen base’s `wav2vec2` encoder and a linear head
  - Train/evaluate with `Trainer`
  - Save weights to [Models/wav2vec2_sentiment/final/model.safetensors](Models/wav2vec2_sentiment/final/model.safetensors)

Evaluate-only (loads frozen base + sentiment head) is included at the end of [wav2vec2_emo.ipynb](wav2vec2_emo.ipynb) and prints metrics plus a confusion matrix.

### C) Sentiment via emotion grouping

- In [wav2vec2_emo.ipynb](wav2vec2_emo.ipynb), two analyses exist:
  1) Direct 3-way sentiment model evaluation (from the trained head).
  2) 8-emotion predictions mapped to sentiment groups:
     - Positive: happy, surprise
     - Neutral: neutral, calm
     - Negative: angry, sad, fear, disgust
  - Both produce classification reports and confusion matrices.

## CNN baselines (spectrograms)

- Pure CNN with overlapping time windows: [Scipts/cnn_overlap.ipynb](Scipts/cnn_overlap.ipynb)
- CNN + BiLSTM (time-distributed features): [Scipts/cnn_rnn_emos.ipynb](Scipts/cnn_rnn_emos.ipynb)
- Models and histories saved under [Models/logmel/](Models/logmel/)

These rely on spectrogram PNGs and metadata produced by [extract_features.ipynb](extract_features.ipynb).

## Data artifacts

- Master and split CSVs:
  - [Data/DataFrame/combined_df.csv](Data/DataFrame/combined_df.csv)
  - [Data/DataFrame/preprocessed_df.csv](Data/DataFrame/preprocessed_df.csv)
  - [Data/DataFrame/train_ser.csv](Data/DataFrame/train_ser.csv)
  - [Data/DataFrame/val_ser.csv](Data/DataFrame/val_ser.csv)
  - [Data/DataFrame/test_ser.csv](Data/DataFrame/test_ser.csv)
  - [Data/DataFrame/sentiment_balanced.csv](Data/DataFrame/sentiment_balanced.csv)

- Preprocessed audio: [Data/Preprocessed_data/](Data/Preprocessed_data/)
- Spectrogram metadata: [Data/DataFrame/spectrogram_metadata.csv](Data/DataFrame/spectrogram_metadata.csv)

## Notes and tips

- Ensure sampling rate consistency (Wav2Vec2 expects 16k by default for facebook/wav2vec2-base; code resamples as needed).
- Audio is padded/truncated to ~4–5 seconds for stable batching.
- Actor-based splits prevent speaker leakage across train/val/test.
- For Windows paths in CSVs, normalization is applied in notebooks.

## Folder layout

```
.
├─ wav2vec2_emo.ipynb
├─ preprocessing.ipynb
├─ extract_features.ipynb
├─ to_ravdess_format.ipynb
├─ Scipts/
│  ├─ cnn_overlap.ipynb
│  └─ cnn_rnn_emos.ipynb
├─ Models/
│  ├─ wav2vec2/final-model/
│  └─ wav2vec2_sentiment/final/model.safetensors
├─ Data/
│  ├─ Preprocessed_data/
│  ├─ Dataset/
│  └─ DataFrame/
└─ Features/
```