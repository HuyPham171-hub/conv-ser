# 🎙️ Speech Emotion Recognition (SER) - IEMOCAP Dataset

This repository focuses on **Speech Emotion Recognition (SER)** using the **IEMOCAP** dataset. The pipeline leverages pre-extracted acoustic embeddings paired with a sequential deep learning architecture (Bidirectional GRU) to effectively model conversational context.

---

## 🗂️ Project Structure & Component Roles

The repository is modularly organized into distinct pipelines:

### 1. `src/data/` (Data Preprocessing)
* **`build_iemocap_metadata.py`**: Parses raw IEMOCAP text and transcription logs to generate the consolidated `iemocap_metadata.csv` (containing utterance IDs, emotion labels, and Session splits).
* **`iemocap_loader.py`**: Implements the custom PyTorch `Dataset` and `DataLoader` classes. It dynamically adapts to three evaluation tracks (`flat8`, `stage1`, `stage2`) using custom mapping dictionaries while grouping sequences via a conversational sliding window.

### 2. `src/features/` (Feature Extraction)
* **`extract_iemocap_embeddings.py`**: Utilizes a pre-trained speech model (e.g., Wav2Vec2) to convert raw `.wav` audio files into high-dimensional acoustic vectors, saved at `data/Embeddings/iemocap_static_embeddings_step1.npy`.
* **`extract_biometrics.py` & `extract_diagnostics.py`**: Utility scripts for extracting auxiliary acoustic features (such as Spectrograms) and running evaluation autopsies to extract aligned prediction CSV files.

### 3. `src/models/` & `src/training/` (Modeling & Training)
* **`bigru_stage3.py`**: Defines the neural network architecture. It deploys a multi-layer Bidirectional Gated Recurrent Unit (Bi-GRU) with a dynamic classification head to process chronological acoustic sequences based on the runtime task.
* **`train_bigru.py`**: The parameter-driven core execution script containing the model's training loop. It evaluates model performance using **Leave-One-Session-Out (LOSO)** cross-validation to guarantee unbiased and subject-independent generalizability.

### 4. `notebooks/` (Experimentation & Analytics)
* **`eda_iemocap.ipynb`**: A Jupyter Notebook dedicated to Exploratory Data Analysis (EDA). It includes label distribution charts, diagnostic autopsies, and confusion matrix visualizations.
* **`test_temporal_pipeline.ipynb`**: A sandbox environment used to validate `DataLoader` sequence alignment and verify the temporal matrix flow prior to official script deployment.

---

## 🚀 Execution Workflow

To replicate the experimental pipeline from scratch, execute the following modules sequentially:

**1. Generate Metadata:** `python src/data/build_iemocap_metadata.py` 
**2. Extract Acoustic Embeddings:** `python src/features/extract_iemocap_embeddings.py`
**3. Train the Network (Parameter-Driven):** The system explicitly drops the highly unbalanced 'Disgust' label.
* *Flat 8-Class Baseline:* `python src/training/train_bigru.py --mode flat8`
* *Hierarchical Stage 1:* `python src/training/train_bigru.py --mode stage1`
* *Hierarchical Stage 2:* `python src/training/train_bigru.py --mode stage2`
**4. Extract Diagnostic Results:** `python src/features/extract_diagnostics.py` (Then open `notebooks/eda_iemocap.ipynb` for final analysis).

---

## 🛠️ Environment Setup
Install the streamlined package dependencies: `pip install -r requirements.txt`
