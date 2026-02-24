import streamlit as st
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --- SETTINGS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "forest_sound_model_v2.h5")
NORM_STATS_PATH = os.path.join(SCRIPT_DIR, "norm_stats.npy")

SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 64
MAX_LEN = 130
CLASSES = ['chainsaw', 'gunshot', 'heavy_machine', 'normal']
CONFIDENCE_THRESHOLD = 0.65  # 65% Threshold for "Other"

# --- CACHED RESOURCES ---
@st.cache_resource
def load_audio_model():
    return load_model(MODEL_PATH)

@st.cache_data
def load_norm_stats():
    stats = np.load(NORM_STATS_PATH, allow_pickle=True).item()
    return stats['mean'], stats['std']

# --- HELPER FUNCTIONS ---
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

# --- UI LAYOUT ---
st.set_page_config(page_title="Forest Guardian AI", page_icon="🌳", layout="centered")

st.title("🌳 Forest Guardian AI")
st.markdown("""
Monitor forest sounds for illegal activities like logging and poaching. 
Upload a `.wav` file to detect specific sound signatures.
""")

# Load resources
try:
    model = load_audio_model()
    mean, std = load_norm_stats()
except Exception as e:
    st.error(f"Failed to load model or stats: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.subheader("Analysis Results")
    
    # Process audio
    with st.spinner("Analyzing sound waves..."):
        mel, audio = extract_mel(uploaded_file)
        
        if mel is not None:
            # UI Columns
            col1, col2 = st.columns([1.5, 1])
            
            with col1:
                st.pyplot(plot_spectrogram(mel))
                st.audio(uploaded_file, format='audio/wav')
                
            with col2:
                # Prepare for model
                X = np.array([mel])[..., np.newaxis]
                X = (X - mean) / std
                
                # Prediction
                preds = model.predict(X, verbose=0)[0]
                max_idx = np.argmax(preds)
                max_conf = preds[max_idx]
                
                # Confidence Threshold Logic
                if max_conf < CONFIDENCE_THRESHOLD:
                    label = "OTHER / AMBIENT"
                    color = "#6c757d" # Gray
                else:
                    label = CLASSES[max_idx].upper()
                    # Color coding based on alarm level
                    if label in ['CHAINSAW', 'GUNSHOT', 'HEAVY_MACHINE']:
                        color = "#ff4b4b" # Red (Danger)
                    else:
                        color = "#28a745" # Green (Safe)
                
                st.markdown(f"### Predicted Class")
                st.markdown(f"<h2 style='color: {color}; text-align: center;'>{label}</h2>", unsafe_allow_html=True)
                st.markdown(f"**Confidence Score:** {max_conf*100:.2f}%")
                
                if max_conf < CONFIDENCE_THRESHOLD:
                    st.warning(f"Note: Confidence is below {CONFIDENCE_THRESHOLD*100:.0f}%. Sound categorized as unknown/other.")
                
                st.divider()
                st.write("Full Probability Distribution:")
                for i, cls in enumerate(CLASSES):
                    st.text(f"{cls.capitalize()}")
                    st.progress(float(preds[i]))

st.sidebar.info("""
**Recognized Classes:**
1. 🪚 Chainsaw
2. 🔫 Gunshot
3. 🚜 Heavy Machine
4. 🍃 Normal (Forest)
""")
