# 🌳 Forest Guardian AI

**Forest Guardian AI** is an AI-powered audio classification system designed to protect forest ecosystems from illegal logging and poaching. It uses a Deep Learning (CNN) model to analyze sound waves and detect specific threat signatures in real-time.

---

## 🚀 Performance Metrics
The model has been rigorously tested on original forest audio data and achieved a high degree of sensitivity and precision.

| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | **97.0%** |
| **Chainsaw Recall** | 100% |
| **Gunshot Recall** | 100% |
| **F1-Score (Weighted)**| 0.97 |

### Classification Breakdown
*   **Chainsaw**: High precision (95%) and perfect recall (100%).
*   **Gunshot**: Perfect precision and recall (100%).
*   **Heavy Machine**: Strong detection (93% precision, 97% recall).
*   **Normal Forest**: High accuracy in filtering ambient noise (95% recall).

---

## ✨ Features
*   **Real-Time Detection**: Web interface for instant audio analysis via Streamlit.
*   **Mel-Spectrogram Visualization**: Converts audio into "visual heatmaps" for the AI to analyze.
*   **Smart "Other" Categorization**: Uses a 65% confidence threshold to prevent false alarms from unknown sounds.
*   **Color-Coded Alerts**: Visual warnings (Red/Green) based on the threat level detected.

---

## 🛠️ Tech Stack
*   **Framework**: TensorFlow / Keras (Deep Learning)
*   **Audio**: Librosa (Feature extraction via Mel-Spectrograms)
*   **Frontend**: Streamlit (Web Dashboard)
*   **Visualization**: Matplotlib
*   **Data Handling**: NumPy, Scikit-learn

---

## 📂 Project Structure
```text
├── Audio_Model_CNN/
│   ├── app.py                  # Main Streamlit Application
│   ├── forest_sound_model_v2.h5 # Trained CNN Model
│   ├── norm_stats.npy           # Normalization Statistics
│   └── train_model_v2.py        # Model Architecture & Training
├── requirements.txt             # Python Dependencies
├── packages.txt                 # Linux System Dependencies
└── README.md                    # This File
```

---

## ⚙️ Installation & Usage

### 1. Local Setup
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run Audio_Model_CNN/app.py
   ```

### 2. Live Deployment
The project is configured for **Streamlit Community Cloud**. Simply push to a public GitHub repo and connect your account at [share.streamlit.io](https://share.streamlit.io).

---

## 📝 License
This project is dedicated to forest conservation and environmental protection.
