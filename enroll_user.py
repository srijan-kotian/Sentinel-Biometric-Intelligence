import cv2
import os

# 1. Setup - Use a simple Haar Cascade for live detection (it's fast)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

user_id = input("Enter unique ID (e.g., USN): ")
user_name = input("Enter Full Name: ")
directory = f"dataset/{user_id}_{user_name}"

if not os.path.exists(directory):
    os.makedirs(directory)

cam = cv2.VideoCapture(0)
count = 0

print(f"[INFO] Initializing face capture for {user_name}. Need 20 samples...")

while count < 20:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw a rectangle so the user knows they are being detected
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Increment count and save the CROPPED face
        count += 1
        img_name = f"{directory}/image_{count}.jpg"
        
        # Professional Move: Save only the face area (ROI)
        face_roi = frame[y:y+h, x:x+w]
        cv2.imwrite(img_name, face_roi)
        
        print(f"Captured {count}/20")

    cv2.imshow("Enrollment - Stay Still", frame)

    # Use a shorter waitKey to make it feel responsive
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
        break

cam.release()
cv2.destroyAllWindows()
print(f"[SUCCESS] High-quality dataset created for {user_name}")