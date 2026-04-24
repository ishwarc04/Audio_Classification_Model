"""
audio_model/classifier.py
Audio classification logic — loads the CNN model, extracts Mel Spectrograms,
and returns class predictions.
"""

import os
import numpy as np
import librosa
import streamlit as st
from tensorflow.keras.models import load_model

# ── Paths ──────────────────────────────────────────────────────────────────────
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(_MODULE_DIR, "forest_sound_model_v2.h5")
NORM_STATS_PATH = os.path.join(_MODULE_DIR, "norm_stats.npy")

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE          = 22050
DURATION             = 3
N_MELS               = 64
MAX_LEN              = 130
CLASSES              = ['chainsaw', 'gunshot', 'heavy_machine', 'normal']
CONFIDENCE_THRESHOLD = 0.65   # below this → "OTHER / AMBIENT"

THREAT_CLASSES = {'CHAINSAW', 'GUNSHOT', 'HEAVY_MACHINE'}


# ── Model / stats loading (cached by Streamlit) ────────────────────────────────
@st.cache_resource
def load_audio_model():
    """Load the CNN model for inference (no optimizer compilation needed)."""
    return load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_norm_stats():
    """Return (mean, std) normalization stats saved during training."""
    stats = np.load(NORM_STATS_PATH, allow_pickle=True).item()
    return stats['mean'], stats['std']


# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_mel(audio_bytes):
    """
    Convert uploaded audio bytes → (mel_db array, raw audio waveform).
    Returns (None, None) on failure.
    """
    try:
        audio, sr = librosa.load(audio_bytes, sr=SAMPLE_RATE, duration=DURATION)

        # Pad / trim to exactly DURATION seconds
        target_len = SAMPLE_RATE * DURATION
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        audio = audio[:target_len]

        # Mel Spectrogram → dB scale
        mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Fix width to MAX_LEN columns
        if mel_db.shape[1] < MAX_LEN:
            mel_db = np.pad(mel_db, ((0, 0), (0, MAX_LEN - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :MAX_LEN]

        return mel_db, audio

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(mel_db):
    """
    Run model inference on a mel spectrogram.

    Returns:
        label      (str)  – class name or "OTHER / AMBIENT"
        confidence (float) – top class probability
        probs      (np.ndarray) – full probability vector
        color      (str)  – hex colour for the UI
    """
    model       = load_audio_model()
    mean, std   = load_norm_stats()

    X     = np.array([mel_db])[..., np.newaxis]
    X     = (X - mean) / std
    probs = model.predict(X, verbose=0)[0]

    max_idx  = int(np.argmax(probs))
    max_conf = float(probs[max_idx])

    if max_conf < CONFIDENCE_THRESHOLD:
        label = "OTHER / AMBIENT"
        color = "#6c757d"
    else:
        label = CLASSES[max_idx].upper()
        color = "#ff4b4b" if label in THREAT_CLASSES else "#28a745"

    return label, max_conf, probs, color
