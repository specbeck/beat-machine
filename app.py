import streamlit as st
import streamlit.components.v1 as components
import librosa
import numpy as np
import pandas as pd
import noisereduce as nr
import joblib
import altair as alt
import tempfile
import os
import shutil

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Beatmachine // Decoder",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');

        /* GLOBAL THEME */
        .stApp {
            background-color: #050505;
            background-image: radial-gradient(circle at 50% 50%, #1a1a1a 0%, #000000 100%);
            color: #E0E0E0;
        }

        /* TYPOGRAPHY */
        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif !important;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        h1 {
            background: linear-gradient(90deg, #00FF94, #00B8FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0px 0px 20px rgba(0, 255, 148, 0.3);
            font-weight: 900;
            font-size: 3.5rem !important;
            text-align: center;
            margin-bottom: 0px;
        }

        p, div, label {
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.1rem;
        }

        /* RECORD WIDGET STYLING */
        .stAudioInput, .stFileUploader {
            border: 2px solid #333;
            border-radius: 15px;
            background: #111;
            padding: 20px;
            box-shadow: 0 0 15px rgba(0, 255, 148, 0.1);
            transition: all 0.3s ease;
        }
        
        .stAudioInput:hover, .stFileUploader:hover {
            border-color: #00FF94;
            box-shadow: 0 0 25px rgba(0, 255, 148, 0.4);
        }
        
        /* TABS STYLING */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #111;
            border-radius: 5px;
            color: #888;
            padding: 10px 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #00FF94 !important;
            color: #000 !important;
            font-weight: bold;
        }

        /* METRIC CARDS */
        div[data-testid="stMetricValue"] {
            font-family: 'Orbitron', sans-serif;
            color: #00FF94 !important;
        }

        /* HIDE STREAMLIT BRANDING */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* PATTERN VISUALIZER CONTAINER */
        .pattern-box {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            font-family: 'Orbitron', sans-serif;
            font-size: 1.5rem;
            color: #fff;
            margin: 20px 0;
            backdrop-filter: blur(5px);
        }
        
        .beat-tag {
            display: inline-block;
            padding: 5px 15px;
            margin: 0 5px;
            border-radius: 5px;
            background: rgba(0, 255, 148, 0.2);
            border: 1px solid #00FF94;
            color: #00FF94;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_brains():
    try:
        clf = joblib.load("beatbox_model_final.pkl")
        scaler = joblib.load("beatbox_scaler_final.pkl")
        return clf, scaler
    except Exception as e:
        return None, None

clf, scaler = load_brains()

# --- 4. PROCESSING ENGINE ---
def extract_features_single_slice(y_slice, sr):
    # VELARDO FEATURE SET (Must match training exactly)
    mfccs = np.mean(librosa.feature.mfcc(y=y_slice, sr=sr, n_mfcc=13), axis=1)
    centroid = np.mean(librosa.featur.spectral_centroid(y=y_slice, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_slice, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y_slice))
    rms = np.mean(librosa.feature.rms(y=y_slice))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y_slice, sr=sr))
    return [centroid, rolloff, zcr, rms, contrast] + list(mfccs)

def process_audio(audio_file):
    if clf is None:
        st.error("Model file not found. Please upload .pkl files.")
        return []

    SR = 22050
    SLICE_DURATION = 0.3
    # FIX 1: Lower confidence slightly to catch softer beats
    CONFIDENCE_THRESHOLD = 0.50 

    # FIX: Robust File Loading for Streamlit Cloud
    suffix = ".wav" 
    if hasattr(audio_file, "name"):
        file_ext = os.path.splitext(audio_file.name)[1]
        if file_ext:
            suffix = file_ext
            
    # Create temp file, close it immediately so other processes can access it
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
    except Exception as e:
        st.error(f"System Error: Could not write temporary file. {e}")
        return pd.DataFrame()

    try:
        # Load Audio from temp file with 60s limit
        y, _ = librosa.load(tmp_path, sr=SR, duration=60)
    except Exception as e:
        st.error(f"""
        **Audio Read Error:** {e}
        
        *Diagnosis:* This usually means the server is missing `libsndfile1`.
        *Fix:* Ensure `packages.txt` exists in your repo with `libsndfile1` inside.
        """)
        return pd.DataFrame()
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    # FIX 2: FORCE NORMALIZATION (Fixes "Faint Signal")
    y = librosa.util.normalize(y)

    # FIX 3: SMART DENOISING
    chunk_size = int(SR*0.5)
    y_clean = y # Default to raw audio
    
    if len(y) > chunk_size:
        noise_part = y[:chunk_size]
        signal_part = y[chunk_size:]
        
        noise_rms = np.mean(librosa.feature.rms(y=noise_part))
        signal_rms = np.mean(librosa.feature.rms(y=signal_part))
        
        # Only denoise if "noise part" is actually quiet (< 50% of signal volume)
        if noise_rms < 0.5 * signal_rms:
             y_clean = nr.reduce_noise(y=y, sr=SR, y_noise=noise_part, prop_decrease=0.60)
        else:
             # Just do a very light clean if we aren't sure
             y_clean = nr.reduce_noise(y=y, sr=SR, y_noise=noise_part, prop_decrease=0.10)

    # FIX 4: ADAPTIVE ONSET DETECTION
    # Try standard sensitivity
    onset_frames = librosa.onset.onset_detect(y=y_clean, sr=SR, backtrack=True, delta=0.07)
    
    # Fallback: If no beats found, try being more sensitive
    if len(onset_frames) == 0:
        onset_frames = librosa.onset.onset_detect(y=y_clean, sr=SR, backtrack=True, delta=0.03)
        
    onset_samples = librosa.frames_to_samples(onset_frames)
    
    timeline = []
    
    for start in onset_samples:
        # Snap to Peak
        search_window = int(SR * 0.05)
        if start + search_window >= len(y_clean): continue
        
        peak_offset = np.argmax(np.abs(y_clean[start : start + search_window]))
        true_start = start + peak_offset - int(SR * 0.01)
        end = true_start + int(SR * SLICE_DURATION)
        
        if true_start < 0 or end > len(y_clean): continue
        y_slice = y_clean[true_start : end]
        
        # Predict
        feat_vector = extract_features_single_slice(y_slice, SR)
        feat_vector_scaled = scaler.transform([feat_vector]) 
        
        prediction = clf.predict(feat_vector_scaled)[0]
        probs = clf.predict_proba(feat_vector_scaled)[0]
        confidence = np.max(probs)
        
        if confidence >= CONFIDENCE_THRESHOLD:
            timestamp = librosa.samples_to_time(true_start, sr=SR)
            timeline.append({"Time": timestamp, "Beat": prediction, "Confidence": confidence})
            
    return pd.DataFrame(timeline)

# --- 5. MAIN UI LAYOUT ---

# Header
st.markdown("<h1>BEAT CLASSIFIER <span style='font-size:1.5rem; vertical-align:middle; color:#555;'>// SYSTEM ACTIVE</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.7; margin-bottom: 40px;'>ML-POWERED PERCUSSIVE ANALYSIS ENGINE</p>", unsafe_allow_html=True)

# Main Container
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 🎙️ INPUT SOURCE")
    
    # Tabs for Record vs Upload
    input_tab1, input_tab2 = st.tabs(["🔴 RECORD LIVE", "📂 UPLOAD FILE"])
    
    audio_source = None
    
    with input_tab1:
        recorded_audio = st.audio_input("Record Beatbox Loop")
        if recorded_audio:
            audio_source = recorded_audio
            
    with input_tab2:
        uploaded_file = st.file_uploader("Upload Audio (m4a, wav, mp3)", type=['m4a', 'wav', 'mp3', 'ogg'])
        if uploaded_file:
            audio_source = uploaded_file
            st.audio(uploaded_file, format='audio/wav')

if audio_source is not None:
    # Processing Indicator
    with st.status("🔍 ANALYZING AUDIO SPECTRUM...", expanded=True) as status:
        st.write("Initializing Librosa...")
        st.write("Normalizing Gain...")
        df_results = process_audio(audio_source)
        st.write("Classifying Patterns...")
        status.update(label="✅ DECODING COMPLETE", state="complete", expanded=False)

    if not df_results.empty:
        # --- A. CONFETTI TRIGGER (JS) ---
        if "Kick" in df_results['Beat'].values and df_results['Confidence'].max() > 0.90:
            components.html("""
                <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
                <script>
                    confetti({
                        particleCount: 150,
                        spread: 100,
                        origin: { y: 0.6 },
                        colors: ['#00FF94', '#00B8FF', '#FFFFFF']
                    });
                </script>
            """, height=0, width=0)

        # --- B. THE PATTERN STRING ---
        st.markdown("### 🧬 DECODED SEQUENCE")
        
        html_pattern = "<div class='pattern-box'>"
        for i, row in df_results.iterrows():
            beat = row['Beat']
            if i > 0: html_pattern += "<span style='color:#555; margin:0 10px;'>➜</span>"
            html_pattern += f"<span class='beat-tag'>{beat}</span>"
        html_pattern += "</div>"
        
        st.markdown(html_pattern, unsafe_allow_html=True)

        # --- C. INTERACTIVE TIMELINE (ALTAIR) ---
        st.markdown("### 📊 RHYTHMIC TOPOLOGY")
        
        chart = alt.Chart(df_results).mark_circle(size=100).encode(
            x=alt.X('Time', title='Time (seconds)'),
            y=alt.Y('Confidence', scale=alt.Scale(domain=[0.4, 1.0])),
            color=alt.Color('Beat', scale=alt.Scale(scheme='viridis'), legend=alt.Legend(title="Sound Type")),
            tooltip=['Time', 'Beat', 'Confidence'],
            size=alt.Size('Confidence', legend=None)
        ).properties(
            height=300,
            background='transparent'
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        # --- D. DETAILED DATA ---
        with st.expander("📂 VIEW RAW TELEMETRY DATA"):
            st.dataframe(
                df_results.style.format({"Time": "{:.2f}s", "Confidence": "{:.1%}"})
                .background_gradient(subset=['Confidence'], cmap='Greens'),
                use_container_width=True
            )

    else:
        st.warning("⚠️ SIGNAL TOO WEAK. No clear beats detected. \n\n**Tip:** Leave a small silence at the very start of recording!")

else:
    # Placeholder state
    st.markdown("""
    <div style='text-align: center; padding: 50px; opacity: 0.3;'>
        <h2>WAITING FOR INPUT...</h2>
    </div>
    """, unsafe_allow_html=True)
