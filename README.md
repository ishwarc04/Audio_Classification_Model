<div align="center">

# 🌳 Forest Guardian AI

### Real-time Forest Sound & Image Threat Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Gemini-AI%20Vision-blueviolet?logo=google)](https://ai.google.dev)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> An AI-powered forest monitoring system that detects illegal activities — chainsaw sounds, gunshots, and heavy machinery — in real time using deep learning audio classification and Gemini-powered visual analysis.

</div>

---

## 📋 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [System Architecture](#-system-architecture)
3. [Dataset](#-dataset)
4. [Classes Predicted](#-classes-predicted)
5. [Features Used](#-features-used)
6. [Algorithms & Model Evolution](#-algorithms--model-evolution)
7. [CNN Architecture (Final)](#-cnn-architecture-final)
8. [Training Strategy](#-training-strategy)
9. [Results & Accuracy](#-results--accuracy)
10. [Project Structure](#-project-structure)
11. [How to Run](#-how-to-run)
12. [Streamlit Web App](#-streamlit-web-app)
13. [Deployment](#-deployment)
14. [Tech Stack](#-tech-stack)

---

## 🚨 Problem Statement

Illegal deforestation, poaching, and unlawful forest activity cause irreversible environmental damage. Manual forest patrols are:
- **Expensive** — require large teams across vast areas
- **Reactive** — threats are detected too late
- **Impractical at scale** — forests can span thousands of kilometres

**Forest Guardian AI** solves this by providing an automated, AI-powered monitoring system that:
- 🎙️ Listens to forest audio and flags **chainsaw sounds, gunshots, and heavy machinery**
- 🖼️ Analyzes camera trap images for **visual signs of illegal activity**
- ⚡ Works in **real-time** with a confidence threshold to reduce false alarms

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Forest Guardian AI                      │
│                    (Streamlit App)                       │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
┌─────────────────┐       ┌──────────────────────┐
│  AUDIO MODULE   │       │    IMAGE MODULE       │
│ audio_model/    │       │   image_model/        │
│                 │       │                       │
│ 1. Upload .wav  │       │ 1. Upload image       │
│ 2. Extract Mel  │       │ 2. Send to Gemini API │
│    Spectrogram  │       │ 3. Parse JSON threat  │
│ 3. Normalize    │       │    assessment         │
│ 4. CNN Predict  │       │ 4. Display result     │
│ 5. Show result  │       └──────────────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CNN MODEL      │    Input: (64 × 130 × 1) Mel Spectrogram
│  4 Conv Blocks  │    Output: 4-class softmax
│  ~4.5M params   │    Threshold: ≥ 65% confidence
└─────────────────┘
```

**Data Flow (Audio):**
```
WAV File → Librosa Load → Mel Spectrogram (64×130) → Normalize → CNN → Softmax Probabilities → Prediction
```

**Data Flow (Image):**
```
Image Upload → PIL → Gemini 2.0 Flash API → JSON Response → Threat / Safe UI
```

---

## 📦 Dataset

| Property | Detail |
| :--- | :--- |
| **Format** | WAV, mono, 22,050 Hz, 3 seconds |
| **Total classes** | 4 core + 4 extended (future) |
| **Balancing** | Augmentation to equalise minority classes |

**Sources explored:**
- 🔊 [ESC-50 Environmental Sound Dataset](https://github.com/karolpiczak/ESC-50)
- 🔊 [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
- 🔊 Custom-curated & recorded `.wav` files
- 🔊 Augmented copies of minority classes

**Data Cleaning Rules:**
- Format: WAV only (MP3 converted via `ffmpeg` / `librosa`)
- Duration: exactly **3 seconds** (padded or trimmed)
- Sample rate: **22,050 Hz**
- Channels: **Mono** (stereo mixed down)

---

## 🎯 Classes Predicted

### Core Classes (trained)

| Class | Description | Threat Level |
| :--- | :--- | :---: |
| 🪚 `chainsaw` | Electric / petrol chainsaw sounds | 🔴 HIGH |
| 🔫 `gunshot` | Single or burst gunfire | 🔴 HIGH |
| 🚜 `heavy_machine` | Bulldozers, excavators, trucks | 🔴 HIGH |
| 🍃 `normal` | Ambient forest — birds, wind, rain | 🟢 SAFE |

> Predictions below **65% confidence** are flagged as `OTHER / AMBIENT` to prevent false alarms.

### Extended Classes (dataset collected, future training)

| Class | Description |
| :--- | :--- |
| `dangerous_animals` | Predator / dangerous wildlife calls |
| `wildlife_large` | Elephant, rhino, large animal movement |
| `mistake_mimicry` | Sounds that mimic threats but are not |
| `human_activity` | Footsteps, voices (non-illegal) |

---

## 🔬 Features Used

### Primary: Mel Spectrogram

Raw audio waveforms are converted into **Mel Spectrograms** — 2D image representations of sound that mimic how the human ear perceives frequency (logarithmic scale). This allows the CNN to treat audio classification as a visual pattern recognition problem.

```python
SAMPLE_RATE = 22050   # Hz
DURATION    = 3       # seconds
N_MELS      = 64      # Mel filter banks  → image height
MAX_LEN     = 130     # time-axis columns → image width
```

Each audio clip becomes a **(64 × 130)** grayscale image fed into the CNN.

### Why Mel Spectrograms over raw MFCCs?
| Feature | MFCC | Mel Spectrogram |
| :--- | :--- | :--- |
| Information retained | Compressed | Full spectral detail |
| CNN suitability | Moderate | ✅ Excellent (image-like) |
| Noise robustness | Good | Very good |
| Used in final model | Baseline only | ✅ Yes |

### Data Augmentation (applied during training)
| Technique | Details |
| :--- | :--- |
| **Time stretching** | ±10% speed change |
| **Pitch shifting** | ±2 semitones |
| **Gaussian noise** | Small random noise injection |
| **Time shift** | Circular roll of waveform ±20% |
| **SpecAugment** | Random frequency + time band masking |

---

## 🤖 Algorithms & Model Evolution

The model went through **4 major iterations**:

| Version | Algorithm | Feature | Accuracy |
| :--- | :--- | :--- | :---: |
| **Baseline** | Logistic Regression | Raw MFCC | ~62% |
| **v1** | Decision Tree | MFCC | ~72% |
| **v1** | Random Forest | MFCC | ~78% |
| **v2 (CNN v1)** | 3-block CNN + Flatten | Mel Spectrogram | ~84.9% |
| **v2 (Final)** | 4-block CNN + GAP + AdamW + SpecAugment | Mel Spectrogram | **94.4%** |

### Key Improvements in Final CNN

| Change | Impact |
| :--- | :--- |
| `Flatten` → `GlobalAveragePooling2D` | Reduced overfitting significantly |
| Added 4th Conv block (256 filters) | Deeper feature learning |
| **SpecAugment** during training | Better generalisation to real-world noise |
| **AdamW** optimizer (weight decay) | Prevented weight explosion |
| **Label smoothing** (0.1) | Prevented overconfident predictions |

---

## 🧠 CNN Architecture (Final)

```
Input: (64, 130, 1)  ←  Mel Spectrogram as grayscale image
│
├── Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
│
├── Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
│
├── Conv2D(128, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
│
├── Conv2D(256, 3×3, ReLU) → BatchNorm → GlobalAveragePooling2D
│
├── Dense(256, ReLU) → Dropout(0.5)
├── Dense(128, ReLU) → Dropout(0.3)
│
└── Dense(4, Softmax)  ←  Output: probability per class
```

**Model stats:**
- Total parameters: **~4.5 million**
- Model size: **~52 MB** (`.h5` format)
- Input shape: `(64, 130, 1)`
- Output: 4-class softmax

---

## ⚙️ Training Strategy

```python
optimizer  = AdamW(learning_rate=1e-3, weight_decay=1e-4)
loss       = CategoricalCrossentropy(label_smoothing=0.1)
epochs     = 50          # with early stopping
batch_size = 32
val_split  = 0.20        # 80% train / 20% validation
```

**Callbacks:**
| Callback | Config |
| :--- | :--- |
| `EarlyStopping` | patience=10, restore best weights |
| `ReduceLROnPlateau` | factor=0.5, patience=5, min_lr=1e-6 |
| `ModelCheckpoint` | saves best val_accuracy model |

---

## 📊 Results & Accuracy

Evaluated on a **held-out test set of original (non-augmented)** forest audio files:

| Metric | Value |
| :--- | :---: |
| **Overall Accuracy** | **94.4%** |
| **Weighted F1-Score** | **0.944** |
| Chainsaw Recall | 95% |
| Gunshot Recall | 95% |
| Heavy Machine Recall | 93% |
| Normal Forest Recall | 95% |

### Confidence Threshold
Predictions below **65%** confidence are shown as `OTHER / AMBIENT` — this prevents the model from making overconfident wrong calls on unknown sounds.

| Confidence | Label shown | Colour |
| :--- | :--- | :--- |
| ≥ 65% + threat class | Class name (e.g. CHAINSAW) | 🔴 Red |
| ≥ 65% + normal | NORMAL | 🟢 Green |
| < 65% | OTHER / AMBIENT | ⚫ Grey |

---

## 📁 Project Structure

```
Audio_Classification_Model/
│
├── audio_model/                  # CNN audio classifier
│   ├── __init__.py
│   ├── classifier.py             # extract_mel(), predict(), constants
│   ├── forest_sound_model_v2.h5  # trained model weights (~52 MB, Git LFS)
│   └── norm_stats.npy            # mean & std from training data
│
├── image_model/                  # Gemini visual analyzer
│   ├── __init__.py
│   └── analyzer.py               # analyze_image() via Gemini API
│
├── app.py                        # Streamlit entry-point (UI only)
├── requirements.txt              # Python dependencies
├── packages.txt                  # Linux system libs (Streamlit Cloud)
├── .gitignore                    # excludes .env, __pycache__, etc.
├── .gitattributes                # Git LFS config for model files
├── model_summary.md              # Full technical walkthrough
└── README.md
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- A Google Gemini API key → [get one free at aistudio.google.com](https://aistudio.google.com)

### 1. Clone the repository
```bash
git clone https://github.com/ishwarc04/Audio_Classification_Model.git
cd Audio_Classification_Model
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🖥️ Streamlit Web App

The app has two tabs:

### 🎙️ Audio Monitoring Tab
1. Upload a `.wav` file
2. App extracts the Mel Spectrogram
3. Spectrogram is normalized using saved `norm_stats.npy`
4. CNN predicts class probabilities
5. Results shown with colour-coded alert + full probability bars

### 🖼️ Image Monitoring Tab
1. Upload a forest image (`.jpg`, `.png`)
2. Image is sent to **Gemini 2.0 Flash**
3. AI returns a structured JSON threat assessment
4. Results shown with confidence score + threat/safe status
5. Adjustable confidence threshold slider

---

## ☁️ Deployment

Deployed on **Streamlit Community Cloud** (free tier).

### Steps to deploy your own
1. Push repo to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub
3. Select `app.py` as the entry point
4. Under **App Settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   ```
5. App is live at a public URL — no server management needed

**Key deployment files:**

| File | Purpose |
| :--- | :--- |
| `app.py` | Streamlit entry point |
| `audio_model/forest_sound_model_v2.h5` | Trained model (Git LFS) |
| `audio_model/norm_stats.npy` | Normalization stats (Git LFS) |
| `requirements.txt` | Python packages |
| `packages.txt` | Linux libs (`libsndfile1` for audio) |

---

## 🛠 Tech Stack

| Tool | Purpose |
| :--- | :--- |
| **Python 3.10+** | Core language |
| **TensorFlow / Keras** | CNN training & inference |
| **Librosa** | Audio loading & Mel Spectrogram extraction |
| **NumPy / Scikit-learn** | Data handling, metrics, class weights |
| **Matplotlib** | Spectrogram visualization |
| **Streamlit** | Web dashboard |
| **Google Gemini 2.0 Flash** | Image threat analysis (Vision AI) |
| **Pillow** | Image handling |
| **python-dotenv** | Local environment variable management |
| **Git LFS** | Large file storage for model weights |

---

<div align="center">

**Built for forest conservation and environmental protection. 🌿**

*If this project helped you, please ⭐ star the repository!*

</div>
