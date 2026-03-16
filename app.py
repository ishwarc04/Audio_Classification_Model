import os

# Load environment variables from a `.env` file if present.
# Place a line like `GOOGLE_API_KEY=your-key` in .env to set the key automatically.
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

# NOTE: The app requires a Gemini API key. A key can be provided via env var, .env, or hard-coded below.
# WARNING: Hard-coding keys in source is not recommended for production.
# The user requested a direct key assignment, so we set it here for now.
os.environ.setdefault(
    "GOOGLE_API_KEY",
    "AIzaSyAjjSsE3MpR0oet7Aex5kncoolxrJ12Vdw",
)

# Silence TensorFlow/absl logging (these warnings are expected in many local setups)
# - TF_CPP_MIN_LOG_LEVEL: 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
# - ABSL_CPP_MIN_LOG_LEVEL: controls absl logging
# - TF_ENABLE_ONEDNN_OPTS: set to 0 to disable oneDNN optimizations logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import json
import logging
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

try:
    from google import genai
except ImportError:
    genai = None

try:
    from PIL import Image
except ImportError:
    Image = None

from tensorflow.keras.models import load_model

# Reduce TensorFlow logger verbosity
logging.getLogger("tensorflow").setLevel(logging.ERROR)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "forest_sound_model_v2.h5")
NORM_STATS_PATH = os.path.join(SCRIPT_DIR, "norm_stats.npy")

SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 64
MAX_LEN = 130
CLASSES = ['chainsaw', 'gunshot', 'heavy_machine', 'normal']
CONFIDENCE_THRESHOLD = 0.65  # 65% Threshold for "Other"

@st.cache_resource
def load_audio_model():
    # Load model for inference only (no optimizer/metrics compilation)
    return load_model(MODEL_PATH, compile=False)

@st.cache_data
def load_norm_stats():
    stats = np.load(NORM_STATS_PATH, allow_pickle=True).item()
    return stats['mean'], stats['std']

def extract_mel(audio_bytes):
    try:
        # Load from memory
        audio, sr = librosa.load(audio_bytes, sr=SAMPLE_RATE, duration=DURATION)
        
        # Padding
        if len(audio) < SAMPLE_RATE * DURATION:
            padding = SAMPLE_RATE * DURATION - len(audio)
            audio = np.pad(audio, (0, padding))
        audio = audio[:SAMPLE_RATE * DURATION]
        
        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Fixed Width
        if mel_db.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0,0),(0,pad_width)))
        else:
            mel_db = mel_db[:, :MAX_LEN]
            
        return mel_db, audio
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None

def plot_spectrogram(mel_db):
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=SAMPLE_RATE, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    return fig


def analyze_image_with_gemini(image: "Image.Image") -> str:
    """Send the uploaded image to Gemini and return the assistant's text response."""

    if genai is None:
        return (
            "google-genai is not installed. Install it with `pip install google-genai` "
            "and restart the app."
        )
    if Image is None:
        return (
            "Pillow is not installed. Install it with `pip install pillow` "
            "and restart the app."
        )

    # NOTE: Ensure GOOGLE_API_KEY is set in the environment or Streamlit secrets.
    # Example: set GOOGLE_API_KEY=YOUR_KEY
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                "You are a forest patrol assistant. Analyze the image and answer in JSON format with the following schema:\n"
                "{\n"
                "  \"threat_found\": <true|false>,\n"
                "  \"confidence\": <0-1 float>,\n"
                "  \"message\": <short summary>,\n"
                "  \"details\": <optional longer explanation>\n"
                "}\n"
                "If the image shows obvious illegal activity (logging, hunting, poaching, trespassing), set threat_found to true. Otherwise set it to false."
                "\nOnly output valid JSON; do not include extra text.\n",
                image,
            ],
        )
        return response.text
    except Exception as e:
        return f"Error querying Gemini: {e}"


st.set_page_config(page_title="Forest Guardian AI", page_icon="🌳", layout="centered")

st.title("🌳 Forest Guardian AI")
st.markdown("""
Monitor forest sounds and imagery for illegal activities like logging and poaching.
Upload a `.wav` audio file or an image to detect suspicious activity.
""")

# Split the UI into two tabs: Audio (existing) and Image (new)
tab_audio, tab_image = st.tabs(["Audio Monitoring", "Image Monitoring"])

with tab_audio:
    try:
        model = load_audio_model()
        mean, std = load_norm_stats()
    except Exception as e:
        st.error(f"Failed to load model or stats: {e}")
        st.stop()

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
                    X = np.array([mel])[..., np.newaxis]
                    X = (X - mean) / std

                    preds = model.predict(X, verbose=0)[0]
                    max_idx = np.argmax(preds)
                    max_conf = preds[max_idx]

                    # Confidence Threshold Logic
                    if max_conf < CONFIDENCE_THRESHOLD:
                        label = "OTHER / AMBIENT"
                        color = "#6c757d"
                    else:
                        label = CLASSES[max_idx].upper()

                        if label in ['CHAINSAW', 'GUNSHOT', 'HEAVY_MACHINE']:
                            color = "#ff4b4b" 
                        else:
                            color = "#28a745"

                    st.markdown(f"### Predicted Class")
                    st.markdown(
                        f"<h2 style='color: {color}; text-align: center;'>{label}</h2>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Confidence Score:** {max_conf*100:.2f}%")

                    if max_conf < CONFIDENCE_THRESHOLD:
                        st.warning(
                            f"Note: Confidence is below {CONFIDENCE_THRESHOLD*100:.0f}%. Sound categorized as unknown/other."
                        )

                    st.divider()
                    st.write("Full Probability Distribution:")
                    for i, cls in enumerate(CLASSES):
                        st.text(f"{cls.capitalize()}")
                        st.progress(float(preds[i]))

with tab_image:
    st.subheader("Image Monitoring (AI Agent)")
    st.markdown(
        """
Upload an image of the forest. The AI agent will look for signs of illegal activity (e.g., tree cutting, hunting, poaching).

**Requirements:**
- Set `GOOGLE_API_KEY` in your environment, Streamlit secrets, or in a `.env` file in this project root.
  Example `.env` line: `GOOGLE_API_KEY=your_key_here`
"""
    )

    # Allow users to control how confident Gemini must be before we treat a scenario as a threat.
    confidence_threshold = st.slider(
        "Threat Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Only treat Gemini output as a threat if its reported confidence is at or above this value.",
    )

    if Image is None:
        st.error(
            "Pillow is not installed. Please install it with `pip install pillow` to enable image uploads."
        )
    else:
        image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if image_file is not None:
            try:
                img = Image.open(image_file)
                st.image(img, caption="Uploaded image", width=700)
                st.markdown("---")

                with st.spinner("Analyzing image with Gemini..."):
                    result_text = analyze_image_with_gemini(img)

                st.subheader("AI Agent Response")

                # Try to parse structured JSON output for colored threat UI.
                try:
                    parsed = json.loads(result_text)
                except Exception:
                    st.warning("Gemini response is not valid JSON; showing raw response.")
                    st.write(result_text)
                else:
                    threat_flag = bool(parsed.get("threat_found", False))
                    confidence = parsed.get("confidence")
                    message = parsed.get("message", "")
                    details = parsed.get("details", "")

                    # Accept threat only if model says threat and confidence meets threshold
                    threat = threat_flag and (confidence is not None and confidence >= confidence_threshold)

                    color = "#ff4b4b" if threat else "#28a745"
                    title = "THREAT DETECTED" if threat else "No Threat Detected"

                    st.markdown(
                        f"<div style='border: 2px solid {color}; padding: 12px; border-radius: 10px;'>"
                        f"<h3 style='margin: 0; color: {color};'>{title}</h3>"
                        f"<p style='margin: 0;'><strong>Confidence:</strong> {confidence}</p>"
                        f"<p style='margin: 0;'><strong>Message:</strong> {message}</p>"
                        f"<p style='margin: 0;'><strong>Threshold:</strong> {confidence_threshold}</p>"
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

st.sidebar.info("""
**Recognized Classes (Audio):**
1. 🪚 Chainsaw
2. 🔫 Gunshot
3. 🚜 Heavy Machine
4. 🍃 Normal (Forest)
""")
