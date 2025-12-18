import streamlit as st
import streamlit.components.v1 as components
import librosa
import numpy as np
import pandas as pd
import noisereduce as nr
import joblib
import altair as alt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Beatmachine // Decoder",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="collapsed"
    )

# --- 2. CUSTOM CSS (THE "HELL LOTS OF CSS") ---
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
        .stAudioInput {
            border: 2px solid #333;
            border-radius: 15px;
            background: #111;
            padding: 20px;
            box-shadow: 0 0 15px rgba(0, 255, 148, 0.1);
            transition: all 0.3s ease;
        }
        
        .stAudioInput:hover {
            border-color: #00FF94;
            box-shadow: 0 0 25px rgba(0, 255, 148, 0.4);
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
    centroid = np.mean(librosa.feature.spectral_centroid(y=y_slice, sr=sr))
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
    CONFIDENCE_THRESHOLD = 0.55 # Slightly loose to catch flow

    # Load & Denoise
    y, _ = librosa.load(audio_file, sr=SR)
    noise_part = y[0:int(SR*0.5)] if len(y) > SR*0.5 else y
    y_clean = nr.reduce_noise(y=y, sr=SR, y_noise=noise_part, prop_decrease=0.80)
    
    # Onset Detect
    onset_frames = librosa.onset.onset_detect(y=y_clean, sr=SR, backtrack=True, delta=0.07)
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
    audio_input = st.audio_input("Record Beatbox Loop")

if audio_input is not None:
    # Processing Indicator
    with st.status("🔍 ANALYZING AUDIO SPECTRUM...", expanded=True) as status:
        st.write("Initializing Librosa...")
        st.write("Denoising Signal...")
        df_results = process_audio(audio_input)
        st.write("Classifying Patterns...")
        status.update(label="✅ DECODING COMPLETE", state="complete", expanded=False)

    if not df_results.empty:
        # --- A. CONFETTI TRIGGER (JS) ---
        # Trigger if we find a high-confidence Kick or special sound
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
        
        # Build HTML string for the beat tags
        html_pattern = "<div class='pattern-box'>"
        for i, row in df_results.iterrows():
            beat = row['Beat']
            # Add visual separator arrow
            if i > 0: html_pattern += "<span style='color:#555; margin:0 10px;'>➜</span>"
            html_pattern += f"<span class='beat-tag'>{beat}</span>"
        html_pattern += "</div>"
        
        st.markdown(html_pattern, unsafe_allow_html=True)

        # --- C. INTERACTIVE TIMELINE (ALTAIR) ---
        st.markdown("### 📊 RHYTHMIC TOPOLOGY")
        
        # Create a classy interactive chart
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
        st.warning("⚠️ SIGNAL TOO WEAK. PLEASE RECORD CLOSER TO MIC.")

else:
    # Placeholder state
    st.markdown("""
    <div style='text-align: center; padding: 50px; opacity: 0.3;'>
        <h2>WAITING FOR INPUT...</h2>
    </div>
    """, unsafe_allow_html=True)
