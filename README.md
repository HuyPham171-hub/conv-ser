---

# 🎙️ Speech Emotion Recognition (SER) - IEMOCAP Dataset

This repository focuses on **Speech Emotion Recognition (SER)** using the **IEMOCAP** dataset. The pipeline leverages pre-extracted acoustic embeddings paired with a sequential deep learning architecture (Bidirectional GRU) to effectively model conversational context.

---

## 🗂️ Project Structure & Component Roles

The repository is modularly organized into distinct pipelines:

### 1. `src/data/` (Data Preprocessing)

* **`build_iemocap_metadata.py`**: Parses raw IEMOCAP text and transcription logs to generate the consolidated `iemocap_metadata.csv` (containing utterance IDs, emotion labels, and Session splits).
* **`iemocap_loader.py`**: Implements the custom PyTorch `Dataset` and `DataLoader` classes. It handles loading the feature vectors (`.npy`), sequence grouping via a conversational sliding window, and batching for model training.

### 2. `src/features/` (Feature Extraction)

* **`extract_iemocap_embeddings.py`**: Utilizes a pre-trained speech model (e.g., Wav2Vec2) to convert raw `.wav` audio files into high-dimensional acoustic vectors, saved at `data/Embeddings/iemocap_static_embeddings_step1.npy`.
* **`extract_biometrics.py` & `extract_diagnostics.py**`: Utility scripts for extracting auxiliary acoustic features (such as Spectrograms) and running deep signal analysis.

### 3. `src/models/` & `src/training/` (Modeling & Training)

* **`bigru_stage3.py`**: Defines the neural network architecture. It deploys a multi-layer Bidirectional Gated Recurrent Unit (Bi-GRU) to process chronological acoustic sequences, capturing the temporal dynamics of the dialogue.
* **`train_bigru.py`**: The core execution script containing the model's training loop. It evaluates model performance using **Leave-One-Session-Out (LOSO)** cross-validation to guarantee unbiased and subject-independent generalizability.

### 4. `notebooks/` (Experimentation & Analytics)

* **`eda_iemocap.ipynb`**: A Jupyter Notebook dedicated to Exploratory Data Analysis (EDA). It includes label distribution charts, data imbalance analysis, and interactive data visualization.
* **`test_temporal_pipeline.ipynb`**: A sandbox environment used to validate `DataLoader` sequence alignment and verify the temporal matrix flow prior to official script deployment.

---

## 🚀 Execution Workflow

To replicate the experimental pipeline from scratch, execute the following modules sequentially:

1. **Generate Metadata:**
```bash
python src/data/build_iemocap_metadata.py

```


*Creates the primary database mapping for all dataset utterances.*
2. **Extract Acoustic Embeddings:**
```bash
python src/features/extract_iemocap_embeddings.py

```


*Converts raw audio waves into static NumPy array dictionaries (`.npy`).*
3. **Train the Network:**
```bash
python src/training/train_bigru.py

```


*Initializes `iemocap_loader.py` for context batching, fetches the Bi-GRU core from `bigru_stage3.py`, and triggers the cross-validation loops across 5 Folds.*
4. **Evaluate & Diagnose Results:**
Open `notebooks/eda_iemocap.ipynb` to evaluate the classification reports and generated confusion matrices (`confusion_matrix_foldX.png` located inside `data/DataFrames/`).

---

## 🛠️ Environment Setup

Install the exact package dependencies into your virtual environment:

```bash
pip install -r requirements.txt

```