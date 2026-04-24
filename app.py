"""
app.py — Forest Guardian AI
Streamlit entry-point.

Model logic is split into two packages:
  audio_model/  — CNN audio classifier (forest threat detection)
  image_model/  — Gemini image analyzer (visual threat detection)
"""

import os

# ── Environment setup ──────────────────────────────────────────────────────────
# Local dev: put GOOGLE_API_KEY=your_key in a .env file (never commit it).
# Production (Streamlit Cloud): add the key under App Settings → Secrets.
try:
    from dotenv import load_dotenv
    load_dotenv()          # no-op if .env doesn't exist
except ImportError:
    pass

# Surface a clear error early if the key is missing entirely.
if not os.environ.get("GOOGLE_API_KEY"):
    import streamlit as st
    st.error(
        "**GOOGLE_API_KEY is not set.**\n\n"
        "- **Local:** create a `.env` file with `GOOGLE_API_KEY=your_key`\n"
        "- **Streamlit Cloud:** go to *App Settings → Secrets* and add:\n"
        "  ```toml\n  GOOGLE_API_KEY = \"your_key\"\n  ```"
    )
    st.stop()

# Suppress TensorFlow / oneDNN noise.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import json
import logging
import streamlit as st
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ── Local modules ──────────────────────────────────────────────────────────────
from audio_model.classifier import (
    extract_mel,
    predict as audio_predict,
    CLASSES,
    CONFIDENCE_THRESHOLD,
)
from image_model.analyzer import analyze_image

try:
    from PIL import Image
except ImportError:
    Image = None

# ── Helpers ────────────────────────────────────────────────────────────────────
def plot_spectrogram(mel_db):
    from audio_model.classifier import SAMPLE_RATE
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=SAMPLE_RATE, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    return fig


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Forest Guardian AI", page_icon="🌳", layout="centered")

st.title("🌳 Forest Guardian AI")
st.markdown(
    "Monitor forest sounds and imagery for illegal activities like logging and poaching.  \n"
    "Upload a `.wav` audio file **or** an image to detect suspicious activity."
)

tab_audio, tab_image = st.tabs(["🎙️ Audio Monitoring", "🖼️ Image Monitoring"])

# ══════════════════════════════════════════════════════════════════════════════
# AUDIO TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_audio:
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

    if uploaded_file is not None:
        st.subheader("Analysis Results")

        with st.spinner("Analyzing sound waves..."):
            mel, audio = extract_mel(uploaded_file)

        if mel is not None:
            col1, col2 = st.columns([1.5, 1])

            with col1:
                st.pyplot(plot_spectrogram(mel))
                st.audio(uploaded_file, format='audio/wav')

            with col2:
                label, max_conf, probs, color = audio_predict(mel)

                st.markdown("### Predicted Class")
                st.markdown(
                    f"<h2 style='color:{color}; text-align:center;'>{label}</h2>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Confidence Score:** {max_conf * 100:.2f}%")

                if max_conf < CONFIDENCE_THRESHOLD:
                    st.warning(
                        f"Confidence below {CONFIDENCE_THRESHOLD * 100:.0f}%. "
                        "Sound categorized as unknown/other."
                    )

                st.divider()
                st.write("Full Probability Distribution:")
                for i, cls in enumerate(CLASSES):
                    st.text(cls.capitalize())
                    st.progress(float(probs[i]))

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_image:
    st.subheader("Image Monitoring (AI Agent)")
    st.markdown(
        """
Upload an image of the forest. The AI agent will look for signs of illegal activity
(e.g., tree cutting, hunting, poaching).

**Requirements:**
- Set `GOOGLE_API_KEY` in your environment, Streamlit secrets, or in a `.env` file.
  Example: `GOOGLE_API_KEY=your_key_here`
"""
    )

    confidence_threshold = st.slider(
        "Threat Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Only treat the result as a threat if Gemini's reported confidence meets this value.",
    )

    if Image is None:
        st.error("Pillow is not installed. Run `pip install pillow` to enable image uploads.")
    else:
        image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if image_file is not None:
            try:
                img = Image.open(image_file)
                st.image(img, caption="Uploaded image", width=700)
                st.markdown("---")

                with st.spinner("Analyzing image with Gemini..."):
                    result_text = analyze_image(img)

                st.subheader("AI Agent Response")

                try:
                    parsed = json.loads(result_text)
                except Exception:
                    st.warning("Gemini response is not valid JSON; showing raw response.")
                    st.write(result_text)
                else:
                    threat_flag  = bool(parsed.get("threat_found", False))
                    confidence   = parsed.get("confidence")
                    message      = parsed.get("message", "")
                    details      = parsed.get("details", "")

                    threat = threat_flag and (
                        confidence is not None and confidence >= confidence_threshold
                    )

                    color = "#ff4b4b" if threat else "#28a745"
                    title = "THREAT DETECTED" if threat else "No Threat Detected"

                    st.markdown(
                        f"<div style='border:2px solid {color}; padding:12px; border-radius:10px;'>"
                        f"<h3 style='margin:0; color:{color};'>{title}</h3>"
                        f"<p style='margin:0;'><strong>Confidence:</strong> {confidence}</p>"
                        f"<p style='margin:0;'><strong>Message:</strong> {message}</p>"
                        f"<p style='margin:0;'><strong>Threshold:</strong> {confidence_threshold}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    if details:
                        st.markdown("**Details:**")
                        st.write(details)

                    st.markdown("---")
                    st.markdown("**Raw JSON Output:**")
                    st.json(parsed)

            except Exception as e:
                st.error(f"Failed to process the image: {e}")

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.info(
    """
**Recognized Classes (Audio):**
1. 🪚 Chainsaw
2. 🔫 Gunshot
3. 🚜 Heavy Machine
4. 🍃 Normal (Forest)
"""
)
