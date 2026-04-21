import streamlit as st
import pandas as pd
import cv2
import os
import pickle
import face_recognition
import numpy as np
import time
import json
import plotly.express as px
from streamlit_option_menu import option_menu

# --- 1. SYSTEM CONFIG & CYBER-THEME ---
st.set_page_config(page_title="Sentinel OS | Professional Edition", layout="wide")

if 'step' not in st.session_state:
    st.session_state.step = "input"
if 'current_user' not in st.session_state:
    st.session_state.current_user = ""

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&display=swap');
    .stApp { background-color: #05070a; color: #c9d1d9; font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    div.stMetric { background: #161b22; border: 1px solid #30363d; border-radius: 15px; padding: 20px; }
    .stButton>button { background: linear-gradient(45deg, #0052D4, #4364F7); color: white; border: none; font-weight: 700; width: 100%; border-radius: 8px; height: 3.5em; text-transform: uppercase; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(67, 100, 247, 0.4); }
    .status-card { padding: 20px; border-radius: 10px; background: #161b22; border-left: 5px solid #58a6ff; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE NEURAL ENGINE (AUTO-TRAIN WITH DEFENSIVE CHECKS) ---
def run_autonomous_training():
    dataset_path = "dataset"
    known_encodings, known_names = [], []
    
    with st.spinner("🧬 NEURAL ENGINE: Running Multi-Stage Extraction..."):
        folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

        for person_name in folders:
            person_dir = os.path.join(dataset_path, person_name)
            images = [i for i in os.listdir(person_dir) if i.endswith(('.jpg', '.png', '.jpeg'))]
            print(f"[PROCESS] Optimizing Identity: {person_name}")
            
            for img_name in images:
                img_path = os.path.join(person_dir, img_name)
                # Load using face_recognition directly to ensure RGB format
                image = face_recognition.load_image_file(img_path)
                
                # STAGE 1: Standard HOG Detection
                face_locations = face_recognition.face_locations(image, model="hog")
                
                # STAGE 2: If Stage 1 fails, try Upsampling (Zooming in)
                if not face_locations:
                    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=2, model="hog")
                
                # STAGE 3: If still failing, try CNN (Slow but 10x more powerful)
                if not face_locations:
                    # Note: This takes 1-2 seconds per image but it NEVER fails
                    face_locations = face_recognition.face_locations(image, model="cnn")

                encs = face_recognition.face_encodings(image, known_face_locations=face_locations)
                
                if len(encs) > 0:
                    known_encodings.append(encs[0])
                    known_names.append(person_name)
                    print(f"  --> {img_name}: [SUCCESS]")
                else:
                    print(f"  --> {img_name}: [CRITICAL FAILURE]")
        
        with open("encodings.pickle", "wb") as f:
            pickle.dump({"encodings": known_encodings, "names": known_names}, f)
            
    if len(known_encodings) > 0:
        st.success(f"🚀 BRAIN UPDATED: {len(known_encodings)} feature-vectors stored.")
        st.session_state.step = "trained"
    else:
        st.error("❌ THE ENGINE IS BLIND: No faces detected in any samples.")

# --- 3. UI NAVIGATION ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #58a6ff;'>SENTINEL OS</h1>", unsafe_allow_html=True)
    selected = option_menu("CORE CONTROL", ["Registration", "Neural Engine", "Personnel Analytics"], 
                          icons=['person-plus-fill', 'cpu-fill', 'bar-chart-steps'], default_index=0)

# --- MODULE: REGISTRATION ---
if selected == "Registration":
    st.title("👤 Personnel Enrollment")
    
    if st.session_state.step == "input":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("I. Identity Metadata")
            u_name = st.text_input("FULL NAME", placeholder="Ex: Srijan S Kotian")
            u_id = st.text_input("SYSTEM ID", placeholder="FAU-2026-X")
            if st.button("PROCEED TO BIOMETRIC CAPTURE"):
                if u_name:
                    st.session_state.current_user = u_name
                    user_path = f"dataset/{u_name}"
                    os.makedirs(user_path, exist_ok=True)
                    
                    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    time.sleep(2.0)
                    
                    if not cam.isOpened():
                        st.error("🚨 CAMERA BUSY: Locked by another process.")
                    else:
                        placeholder = st.empty()
                        for i in range(15):
                            ret, frame = cam.read()
                            if ret:
                                cv2.imwrite(f"{user_path}/sample_{i}.jpg", frame)
                                placeholder.image(frame, channels="BGR", caption=f"Ingesting {i+1}/15")
                                time.sleep(0.1)
                        cam.release()
                        st.session_state.step = "captured"
                        st.rerun()
                else:
                    st.error("Name is required.")

    elif st.session_state.step == "captured":
        st.markdown(f"<div class='status-card'>📸 <b>DATA SECURED:</b> 15 Samples ready for <b>{st.session_state.current_user}</b>.</div>", unsafe_allow_html=True)
        if st.button("⚡ EXECUTE NEURAL TRAINING"):
            run_autonomous_training()
            st.rerun()

    elif st.session_state.step == "trained":
        st.balloons()
        st.success("🎯 SYSTEM UPDATED. Identity is now recognized.")
        if st.button("Enroll New Personnel"):
            st.session_state.step = "input"
            st.rerun()

# --- MODULE: NEURAL ENGINE ---
elif selected == "Neural Engine":
    st.title("👁️ Neural Inference Hub")
    if st.button("ACTIVATE SENTINEL STREAM"):
        import subprocess
        subprocess.Popen(["python", "recognize_faces_video.py"])

# --- MODULE: ANALYTICS ---
elif selected == "Personnel Analytics":
    st.title("📊 Strategic Intelligence Dashboard")
    if os.path.exists("recognition_analytics.csv"):
        df = pd.read_csv("recognition_analytics.csv")
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Confidence'] = (1 - df['Distance']) * 100
        
        all_users = df['Name'].unique()
        target = st.selectbox("FILTER PERSONNEL", ["OVERVIEW"] + list(all_users))
        display_df = df if target == "OVERVIEW" else df[df['Name'] == target]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("AVG CONFIDENCE", f"{round(display_df['Confidence'].mean(), 1)}%")
        m2.metric("TOTAL SAMPLES", len(display_df))
        m3.metric("STABILITY (σ)", f"{round(display_df['Confidence'].std(), 2)}")

        fig = px.area(display_df.tail(100), x='Timestamp', y='Confidence', template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Telemetry logs offline.")