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
from PIL import Image

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

# --- 2. THE NEURAL ENGINE (AUTO-TRAIN) ---
def run_autonomous_training():
    dataset_path = "dataset"
    known_encodings, known_names = [], []
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    with st.spinner("🧬 NEURAL ENGINE: Running Multi-Stage Extraction..."):
        folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

        for person_name in folders:
            person_dir = os.path.join(dataset_path, person_name)
            images = [i for i in os.listdir(person_dir) if i.endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_name in images:
                img_path = os.path.join(person_dir, img_name)
                image = face_recognition.load_image_file(img_path)
                
                # Multi-Stage Detection
                face_locations = face_recognition.face_locations(image, model="hog")
                if not face_locations:
                    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=2, model="hog")
                
                encs = face_recognition.face_encodings(image, known_face_locations=face_locations)
                
                if len(encs) > 0:
                    known_encodings.append(encs[0])
                    known_names.append(person_name)
        
        with open("encodings.pickle", "wb") as f:
            pickle.dump({"encodings": known_encodings, "names": known_names}, f)
            
    if len(known_encodings) > 0:
        st.success(f"🚀 BRAIN UPDATED: {len(known_encodings)} feature-vectors stored.")
        st.session_state.step = "trained"
    else:
        st.error("❌ THE ENGINE IS BLIND: No faces detected.")

# --- 3. UI NAVIGATION ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #58a6ff;'>SENTINEL OS</h1>", unsafe_allow_html=True)
    selected = option_menu("CORE CONTROL", ["Registration", "Neural Engine", "Personnel Analytics"], 
                          icons=['person-plus-fill', 'cpu-fill', 'bar-chart-steps'], default_index=0)

# --- MODULE: REGISTRATION (CLOUD VERSION) ---
if selected == "Registration":
    st.title("👤 Personnel Enrollment")
    
    if st.session_state.step == "input":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("I. Identity Metadata")
            u_name = st.text_input("FULL NAME", placeholder="Ex: Srijan S Kotian")
            u_id = st.text_input("SYSTEM ID", placeholder="FAU-2026-X")
            
            # Use Browser Camera instead of CV2
            img_file = st.camera_input("BIOMETRIC SCAN")

            if img_file is not None and u_name:
                if st.button("PROCEED TO ENROLLMENT"):
                    st.session_state.current_user = u_name
                    user_path = f"dataset/{u_name}"
                    os.makedirs(user_path, exist_ok=True)
                    
                    # Process the uploaded image
                    img = Image.open(img_file)
                    img_array = np.array(img)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Save multiple samples for training
                    for i in range(5):
                        cv2.imwrite(f"{user_path}/sample_{i}.jpg", img_bgr)
                    
                    st.session_state.step = "captured"
                    st.rerun()

    elif st.session_state.step == "captured":
        st.markdown(f"<div class='status-card'>📸 <b>DATA SECURED:</b> Biometrics ready for <b>{st.session_state.current_user}</b>.</div>", unsafe_allow_html=True)
        if st.button("⚡ EXECUTE NEURAL TRAINING"):
            run_autonomous_training()
            st.rerun()

    elif st.session_state.step == "trained":
        st.balloons()
        st.success("🎯 SYSTEM UPDATED. Identity is now recognized.")
        if st.button("Enroll New Personnel"):
            st.session_state.step = "input"
            st.rerun()

# --- MODULE: NEURAL ENGINE (IMAGE INFERENCE) ---
elif selected == "Neural Engine":
    st.title("👁️ Neural Inference Hub")
    st.info("Upload a photo to verify identity against the current database.")
    
    test_img = st.file_uploader("UPLOAD TEST IMAGE", type=['jpg', 'png', 'jpeg'])
    
    if test_img and os.path.exists("encodings.pickle"):
        with open("encodings.pickle", "rb") as f:
            data = pickle.load(f)
            
        img = Image.open(test_img)
        img_array = np.array(img)
        
        # Recognition Logic
        face_locations = face_recognition.face_locations(img_array)
        face_encodings = face_recognition.face_encodings(img_array, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            name = "UNKNOWN THREAT"
            
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = data["names"][best_match_index]
            
            st.write(f"**Identified:** {name}")
            cv2.rectangle(img_array, (left, top), (right, bottom), (0, 255, 0), 2)
        
        st.image(img_array, caption="Inference Result", use_container_width=True)
    elif not os.path.exists("encodings.pickle"):
        st.warning("Database empty. Please run Registration first.")

# --- MODULE: ANALYTICS ---
elif selected == "Personnel Analytics":
    st.title("📊 Strategic Intelligence Dashboard")
    st.write("Real-time telemetry and identification logs.")
    # (Simplified dummy data for demo)
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['Confidence', 'Latency', 'Stability'])
    st.line_chart(chart_data)