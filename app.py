import streamlit as st
import librosa
import numpy as np
import noisereduce as nr
import joblib
import os
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="Beatbox AI", page_icon="🎤")

st.title("🎤 The Beatbox Decoder")
st.write("Record a beat pattern, and the AI will decode it into 'Kick', 'Snare', 'Hat', etc.")

# --- LOAD BRAINS (Cached so it runs fast) ---
@st.cache_resource
def load_models():
    clf = joblib.load("beatbox_model_final.pkl")
    scaler = joblib.load("beatbox_scaler_final.pkl")
    return clf, scaler

try:
    clf, scaler = load_models()
    st.success("✅ AI Brain Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading models. Make sure .pkl files are in the same folder! {e}")
    st.stop()

# --- HELPER FUNCTIONS (Your Extraction Logic) ---
# (Copied from your previous script)
def extract_features_single_slice(y_slice, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=y_slice, sr=sr, n_mfcc=13), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y_slice, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_slice, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y_slice))
    rms = np.mean(librosa.feature.rms(y=y_slice))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y_slice, sr=sr))
    return [centroid, rolloff, zcr, rms, contrast] + list(mfccs)

def process_audio(audio_file):
    SR = 22050
    SLICE_DURATION = 0.3
    CONFIDENCE_THRESHOLD = 0.60
    
    # Load from Streamlit upload/record
    y, _ = librosa.load(audio_file, sr=SR)
    
    # Denoise
    noise_part = y[0:int(SR*0.5)] if len(y) > SR*0.5 else y
    y_clean = nr.reduce_noise(y=y, sr=SR, y_noise=noise_part, prop_decrease=0.80)
    
    # Detect Onsets
    onset_frames = librosa.onset.onset_detect(y=y_clean, sr=SR, backtrack=True, delta=0.07)
    onset_samples = librosa.frames_to_samples(onset_frames)
    
    timeline = []
    
    for start in onset_samples:
        search_window = int(SR * 0.05)
        if start + search_window >= len(y_clean): continue
        
        peak_offset = np.argmax(np.abs(y_clean[start : start + search_window]))
        true_start = start + peak_offset - int(SR * 0.01)
        end = true_start + int(SR * SLICE_DURATION)
        
        if true_start < 0 or end > len(y_clean): continue
        y_slice = y_clean[true_start : end]
        
        # Extract & Predict
        feat_vector = extract_features_single_slice(y_slice, SR)
        feat_vector_scaled = scaler.transform([feat_vector]) 
        
        prediction = clf.predict(feat_vector_scaled)[0]
        probs = clf.predict_proba(feat_vector_scaled)[0]
        confidence = np.max(probs)
        
        if confidence >= CONFIDENCE_THRESHOLD:
            # Save timestamp relative to start
            timestamp = librosa.samples_to_time(true_start, sr=SR)
            timeline.append({"time": timestamp, "beat": prediction, "conf": confidence})
            
    return timeline

# --- THE UI ---
audio_input = st.audio_input("Record your beat loop (5-10s)")

if audio_input is not None:
    st.audio(audio_input)
    
    with st.spinner("Decoding your beatbox..."):
        try:
            timeline = process_audio(audio_input)
            
            if not timeline:
                st.warning("No clear beats detected. Try recording closer to the mic!")
            else:
                # 1. VISUAL TIMELINE
                st.subheader("Your Pattern")
                
                # Create a simple visual string
                pattern_str = "  ➜  ".join([f"**{t['beat']}**" for t in timeline])
                st.markdown(f"### {pattern_str}")
                
                # 2. DETAILED TABLE
                st.divider()
                st.write("Detailed Breakdown:")
                
                # Display as columns
                for item in timeline:
                    col1, col2, col3 = st.columns([1, 2, 2])
                    with col1:
                        st.write(f"⏱️ {item['time']:.2f}s")
                   with col2:
                        st.write(f"🥁 **{item['beat']}**")
                    with col3:
                        # Visual progress bar for confidence
                        st.progress(item['conf'], text=f"{item['conf']*100:.0f}% Confidence")
                        
        except Exception as e:
            st.error(f"An error occurred during processing: {e}") 
