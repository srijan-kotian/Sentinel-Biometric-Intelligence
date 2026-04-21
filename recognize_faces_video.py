import face_recognition
import imutils
import pickle
import time
import cv2
import os
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.spatial import distance as dist

# --- 1. LIVENESS CONSTANTS ---
EYE_AR_THRESH = 0.23  # Threshold for a blink
EYE_AR_CONSEC_FRAMES = 2  # Frames the eye must be closed to count
blink_counter = 0
total_blinks = 0
liveness_status = "STATIONARY"

def calculate_ear(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmark
    C = dist.euclidean(eye[0], eye[3])
    # Compute the Eye Aspect Ratio
    ear = (A + B) / (2.0 * C)
    return ear

def log_recognition_data(name, distance, blinks):
    file_path = "recognition_analytics.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # We now log blinks as a "Liveness Proof"
    data = {"Timestamp": [now], "Name": [name], "Distance": [float(distance)], "Blinks": [blinks]}
    df_new = pd.DataFrame(data)
    if not os.path.isfile(file_path):
        df_new.to_csv(file_path, index=False)
    else:
        df_new.to_csv(file_path, mode='a', header=False, index=False)

print("[INFO] Loading Biometric Brain...")
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

vs = cv2.VideoCapture(0)
time.sleep(2.0)

process_this_frame = True

while True:
    ret, frame = vs.read()
    if not ret: break

    # Downsampling for processing speed
    small_frame = imutils.resize(frame, width=400)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        # Detect Faces
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
        
        face_names = []
        face_distances_log = []

        if len(face_locations) > 0:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for i, face_encoding in enumerate(face_encodings):
                # 1. BIOMETRIC IDENTIFICATION
                matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
                name = "Unknown"

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = data["names"][best_match_index]
                        face_distances_log.append(face_distances[best_match_index])
                
                face_names.append(name)

                # 2. LIVENESS CHECK (Blink Detection)
                # Only check for the first face detected to keep FPS high
                if i == 0 and len(face_landmarks_list) > 0:
                    landmarks = face_landmarks_list[i]
                    left_eye = landmarks["left_eye"]
                    right_eye = landmarks["right_eye"]
                    
                    ear_left = calculate_ear(left_eye)
                    ear_right = calculate_ear(right_eye)
                    avg_ear = (ear_left + ear_right) / 2.0

                    if avg_ear < EYE_AR_THRESH:
                        blink_counter += 1
                    else:
                        if blink_counter >= EYE_AR_CONSEC_FRAMES:
                            total_blinks += 1
                            liveness_status = "LIVE_VERIFIED"
                        blink_counter = 0

        else:
            face_names = []
            face_distances_log = []

    process_this_frame = not process_this_frame

    # Draw the UI
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up since we processed on small_frame
        top, right, bottom, left = top*2, right*2, bottom*2, left*2
        
        # Color coding: Green for Verified + Live, Red for Unknown/Static
        color = (0, 255, 0) if liveness_status == "LIVE_VERIFIED" and name != "Unknown" else (0, 0, 255)
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Display Name + Liveness Status
        status_text = f"{name} | {liveness_status}"
        cv2.putText(frame, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Data Logging
        if name != "Unknown" and liveness_status == "LIVE_VERIFIED":
            log_recognition_data(name, face_distances_log[0], total_blinks)

    # UI Overlay for Blinks
    cv2.putText(frame, f"Blinks Logged: {total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Sentinel OS | Research-Grade Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

vs.release()
cv2.destroyAllWindows()