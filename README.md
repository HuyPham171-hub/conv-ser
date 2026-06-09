# Speech Emotion Recognition Workspace

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN%20Baselines-orange.svg)

This repository contains a speech emotion recognition workspace focused on two main tasks:

- 8-way Speech Emotion Recognition (SER) from raw audio.
- 3-way sentiment classification derived from emotion labels and balanced sentiment datasets.

The project is organized around preprocessing, Wav2Vec2 fine-tuning, CNN baselines, cross-dataset experiments, and sentiment grouping. The main working area is the [Ser/](Ser/) folder.

## Key Capabilities

- Audio preprocessing with silence trimming, Wiener denoising, and amplitude normalization.
- Emotion grouping from 8 SER classes into Negative, Moderate, and Positive sentiment labels.
- Wav2Vec2-based SER and sentiment experiments.
- CNN baselines built on spectrogram and full-frequency image features.
- Dataset and metadata preparation for both raw audio and spectrogram-based workflows.

## Main Project Layout

- [Ser/](Ser/): Primary SER workspace with notebooks, scripts, processed CSVs, and audio artifacts.
- [IEMOCAP_full_release/](IEMOCAP_full_release/): Raw IEMOCAP corpus and documentation.
- [Chatbot/](Chatbot/): Reference workspace with related SER/Sentiment notebooks and notes.

Inside [Ser/](Ser/) you will find:

- [preprocessing.ipynb](Ser/preprocessing.ipynb): Audio trimming, denoising, and normalization.
- [group_wav2vec2.ipynb](Ser/group_wav2vec2.ipynb) and [group_wav2vec2_v2.ipynb](Ser/group_wav2vec2_v2.ipynb): Emotion-to-sentiment mapping, balancing, and Wav2Vec2 sentiment experiments.
- [cross_wav2vec2.ipynb](Ser/cross_wav2vec2.ipynb): Cross-dataset Wav2Vec2 workflow.
- [goup_cnn.ipynb](Ser/goup_cnn.ipynb): CNN baseline training.
- [augmentation.py](Ser/augmentation.py): Audio augmentation helpers used by the notebooks.
- [Script/](Ser/Script/): Additional notebook variants for cross-domain and grouped training runs.

## Data Artifacts

- [Ser/Data/](Ser/Data/): Dataset folders, preprocessed audio, spectrogram images, and metadata.
- [Ser/DataFrame/](Ser/DataFrame/): Master and processed CSV files.

Notable files:

- [Ser/DataFrame/combined_df.csv](Ser/DataFrame/combined_df.csv)
- [Ser/DataFrame/processed_df.csv](Ser/DataFrame/processed_df.csv)
- [Ser/DataFrame/processed_sentiment_df.csv](Ser/DataFrame/processed_sentiment_df.csv)
- [Ser/DataFrame/processed_sentiment_balanced.csv](Ser/DataFrame/processed_sentiment_balanced.csv)
- [Ser/Data/spectrogram_metadata.csv](Ser/Data/spectrogram_metadata.csv)

## Setup

The Ser workspace targets Python 3.12 and uses the dependencies listed in [Ser/requirements.txt](Ser/requirements.txt).

```sh
pip install -r Ser/requirements.txt
```

Recommended: open the notebooks in VS Code and run them in the provided order so the CSV and audio artifacts are generated before training.

## Workflow

1. Preprocess audio
- Run [Ser/preprocessing.ipynb](Ser/preprocessing.ipynb) to trim silence, denoise, normalize, and save cleaned WAV files under [Ser/Data/Preprocessed_data/](Ser/Data/Preprocessed_data/).

2. Build master metadata tables
- Use the notebooks in [Ser/](Ser/) and [Ser/Script/](Ser/Script/) to combine sources into [Ser/DataFrame/combined_df.csv](Ser/DataFrame/combined_df.csv) and the processed SER/sentiment CSVs.

3. Train Wav2Vec2 models
- Use [Ser/group_wav2vec2.ipynb](Ser/group_wav2vec2.ipynb), [Ser/group_wav2vec2_v2.ipynb](Ser/group_wav2vec2_v2.ipynb), and [Ser/cross_wav2vec2.ipynb](Ser/cross_wav2vec2.ipynb) for SER and sentiment experiments.

4. Train CNN baselines
- Use [Ser/goup_cnn.ipynb](Ser/goup_cnn.ipynb) for spectrogram-based CNN experiments.

5. Use generated features
- Spectrograms and full-frequency images are stored under [Ser/Data/Spectrograms/](Ser/Data/Spectrograms/), [Ser/Data/Spectrograms_filter/](Ser/Data/Spectrograms_filter/), and [Ser/Data/Spectrogram_fullfreq/](Ser/Data/Spectrogram_fullfreq/).

## Notes

- The project uses speaker- and dataset-aware splits in the notebooks to reduce leakage.
- Sentiment labels are derived by grouping emotions into Negative, Moderate, and Positive classes.
- The exact preprocessing and training settings may differ slightly between notebook versions, so treat the notebooks as the source of truth.

## Directory Snapshot

```text
.
├─ README.md
├─ IEMOCAP_full_release/
├─ Chatbot/
└─ Ser/
   ├─ augmentation.py
   ├─ preprocessing.ipynb
   ├─ group_wav2vec2.ipynb
   ├─ group_wav2vec2_v2.ipynb
   ├─ cross_wav2vec2.ipynb
   ├─ goup_cnn.ipynb
   ├─ Script/
   ├─ Data/
   └─ DataFrame/
```