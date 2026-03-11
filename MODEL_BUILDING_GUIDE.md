# 🌳 Forest Guardian AI — Model Building Guide

A complete, step-by-step walkthrough of how the audio classification model was built from scratch.

---

## 📋 Table of Contents
1. [Problem Definition](#1-problem-definition)
2. [Tech Stack](#2-tech-stack)
3. [Dataset Collection & Classes](#3-dataset-collection--classes)
4. [Data Cleaning & Preparation](#4-data-cleaning--preparation)
5. [Feature Extraction — Mel Spectrograms](#5-feature-extraction--mel-spectrograms)
6. [Model Evolution](#6-model-evolution)
7. [CNN Architecture (Final — v2)](#7-cnn-architecture-final--v2)
8. [Training Strategy](#8-training-strategy)
9. [Normalization & Saving Stats](#9-normalization--saving-stats)
10. [Evaluation & Results](#10-evaluation--results)
11. [Streamlit Web App](#11-streamlit-web-app)
12. [Deployment](#12-deployment)

---

## 1. Problem Definition

**Goal:** Build an AI system that listens to forest sounds and automatically detects illegal human activity (chainsaw, gunshot, heavy machinery) vs. normal forest ambience.

**Why AI?** Manual forest monitoring is expensive and impractical at scale. An AI model can process audio in real-time and trigger alerts automatically.

---

## 2. Tech Stack

| Tool | Purpose |
| :--- | :--- |
| **Python 3.10+** | Core language |
| **TensorFlow / Keras** | Deep Learning framework |
| **Librosa** | Audio loading & Mel-Spectrogram extraction |
| **NumPy / Scikit-learn** | Data handling, normalization, metrics |
| **Matplotlib** | Visualization |
| **Streamlit** | Web dashboard / UI |
| **tqdm** | Progress bars during training |

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 3. Dataset Collection & Classes

The model was trained on **4 core sound classes**:

| Class | Description | Threat Level |
| :--- | :--- | :--- |
| `chainsaw` | Electric/petrol chainsaw sounds | 🔴 High |
| `gunshot` | Single or burst gunfire sounds | 🔴 High |
| `heavy_machine` | Bulldozers, excavators, trucks | 🔴 High |
| `normal` | Ambient forest sounds (birds, wind, rain) | 🟢 Safe |

**Dataset Sources explored:**
- ESC-50 Environmental Sound Dataset
- UrbanSound8K
- Custom-recorded / curated `.wav` files
- Augmented copies of minority classes to balance the dataset

**Dataset expansion considered** (for future versions):
- `Dangerous_Animals` — predator calls
- `Wildlife_Large` — elephant/rhino movement
- `Mistake_Mimicry` — sounds that mimic threats but aren't
- `Human_Activity` — footsteps, voices (non-illegal)

---

## 4. Data Cleaning & Preparation

All raw audio files were cleaned and standardized using a preprocessing script before training.

**Standardization rules applied:**
- Format: **WAV only** (MP3 / other formats converted using `librosa` or `ffmpeg`)
- Duration: **Exactly 3 seconds**
  - Files shorter than 3s → zero-padded at the end
  - Files longer than 3s → trimmed from the start
- Sample Rate: **22,050 Hz** (standard for audio ML)
- Channels: **Mono** (stereo files mixed down)

**Result:** All files in `cleaned_data/` are uniform 3-second `.wav` mono files, verified by a validation script.

---

## 5. Feature Extraction — Mel Spectrograms

Raw waveforms are not fed into the CNN directly. Instead, each audio clip is converted into a **Mel Spectrogram** — a 2D image representation of sound.

**Why Mel Spectrograms?**
- They mimic how the human ear perceives frequency (logarithmic scale)
- They turn audio into an image → CNN can treat it like a visual pattern recognition problem

**Parameters used:**

```python
SAMPLE_RATE = 22050   # Hz
DURATION    = 3       # seconds
N_MELS      = 64      # number of Mel filter banks (height of image)
MAX_LEN     = 130     # fixed time-axis width (columns)
```

**Extraction pipeline:**

```python
import librosa
import numpy as np

def extract_mel(file_path):
    audio, sr = librosa.load(file_path, sr=22050, duration=3)

    # Pad or trim to exact 3 seconds
    target_len = 22050 * 3
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # Compute Mel Spectrogram and convert to dB scale
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Fix width to 130 columns
    if mel_db.shape[1] < 130:
        mel_db = np.pad(mel_db, ((0,0),(0, 130 - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :130]

    return mel_db  # Shape: (64, 130)
```

Each sample becomes a **(64 × 130)** grayscale image fed into the CNN.

---

## 6. Model Evolution

The model went through several iterations to improve accuracy:

| Version | Approach | Accuracy |
| :--- | :--- | :--- |
| **Baseline** | Logistic Regression on raw MFCC features | ~60–65% |
| **v1 (Traditional ML)** | Decision Tree + Random Forest comparison | ~75–80% |
| **v2 (CNN v1)** | 3-block CNN + Flatten + Dense layers | ~84.9% |
| **v2 (CNN final)** | 4-block CNN + GlobalAveragePooling2D + AdamW + SpecAugment | **97.0%** |

Key improvements in the final version:
- Replaced `Flatten` with `GlobalAveragePooling2D` → reduced overfitting
- Added a **4th convolutional block** for deeper feature learning
- Applied **SpecAugment** (masking random time/frequency bands during training)
- Expanded **data augmentation** to the `heavy_machine` class as well
- Used **AdamW optimizer** (Adam with weight decay) instead of plain Adam
- Used **label smoothing** (factor 0.1) in the loss function to prevent overconfidence

---

## 7. CNN Architecture (Final — v2)

```
Input: (64, 130, 1)  ← Mel Spectrogram as grayscale image

Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
Conv2D(128, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
Conv2D(256, 3×3, ReLU) → BatchNorm → GlobalAveragePooling2D

Dense(256, ReLU) → Dropout(0.5)
Dense(128, ReLU) → Dropout(0.3)
Dense(4, Softmax)  ← Output: probability for each of 4 classes
```

**Key design choices:**
- `BatchNormalization` after every conv layer → stable and faster training
- `GlobalAveragePooling2D` instead of Flatten → reduces parameters, prevents overfitting
- Progressive filter doubling (32 → 64 → 128 → 256) → learns from simple to complex patterns
- `Dropout` at both conv and dense layers → strong regularization

---

## 8. Training Strategy

**Data Augmentation** (applied on-the-fly during training):
- Time stretching (±10%)
- Pitch shifting (±2 semitones)
- Adding Gaussian noise
- SpecAugment: random frequency masking + time masking

**Training configuration:**

```python
optimizer  = AdamW(learning_rate=1e-3, weight_decay=1e-4)
loss       = CategoricalCrossentropy(label_smoothing=0.1)
epochs     = 50
batch_size = 32
val_split  = 0.2   # 80% train / 20% validation
```

**Callbacks used:**
- `EarlyStopping(patience=10, restore_best_weights=True)` — stops if val_loss stops improving
- `ReduceLROnPlateau(factor=0.5, patience=5)` — halves LR when val_loss plateaus
- `ModelCheckpoint` — saves the best model weights automatically

**Dataset balance:** Classes with fewer samples were upsampled via augmentation to match the largest class (avoiding bias toward the majority class).

---

## 9. Normalization & Saving Stats

To ensure consistent input to the model at inference time, the training data mean and standard deviation are computed **once** and saved to `norm_stats.npy`.

```python
# During training
mean = X_train.mean()
std  = X_train.std()
np.save('norm_stats.npy', {'mean': mean, 'std': std})

# Normalization applied to train, val, and inference
X_normalized = (X - mean) / std
```

At inference time (`app.py`), the same stats are loaded and applied before prediction — this is critical for model correctness.

---

## 10. Evaluation & Results

The final model was evaluated on a held-out test set of original (non-augmented) forest audio files.

| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | **94.4.0%** |
| **Weighted F1-Score** | **0.94.4** |
| **Chainsaw Recall** | 95% |
| **Gunshot Recall** | 95% |
| **Heavy Machine Recall** | 93% |
| **Normal Forest Recall** | 95% |

**Confidence Threshold:** A prediction is only accepted if the model's top class probability is ≥ **65%**. Otherwise the audio is flagged as `OTHER / AMBIENT` to prevent false alarms from unknown sounds.

---

## 11. Streamlit Web App

The trained model is served through a Streamlit web dashboard (`app.py`).

**User flow:**
1. User uploads a `.wav` file
2. App extracts the Mel Spectrogram
3. Spectrogram is normalized using saved `norm_stats.npy`
4. CNN model predicts class probabilities
5. Result displayed with:
   - 🔴 **Red alert** for chainsaw / gunshot / heavy machine
   - 🟢 **Green** for normal forest sounds
   - ⚫ **Gray** for unknown / low-confidence sounds
6. Full probability distribution shown as a progress bar chart

**Run locally:**
```bash
streamlit run app.py
```

---

## 12. Deployment

The app is deployed on **Streamlit Community Cloud** (free tier):

1. Push the repository to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub
3. Select `app.py` as the entry point
4. Streamlit Cloud reads `requirements.txt` (Python packages) and `packages.txt` (Linux system libraries, e.g. `libsndfile1` for audio support)
5. App is live at a public URL — no server management needed

**Key files for deployment:**

| File | Purpose |
| :--- | :--- |
| `app.py` | Entry point for Streamlit |
| `forest_sound_model_v2.h5` | Trained model weights |
| `norm_stats.npy` | Normalization statistics |
| `requirements.txt` | Python dependencies |
| `packages.txt` | Linux system dependencies |

---

*Built for forest conservation and environmental protection. 🌿*
