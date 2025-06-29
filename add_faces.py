# File: add_faces.py
import cv2
import os
import pickle
import numpy as np
from deepface import DeepFace
import time
import mediapipe as mp

# Setup
dataset_dir = "saved_faces"
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs("data", exist_ok=True)
label_path = "data/labels.pkl"

# Load or create labels
if os.path.exists(label_path):
    try:
        with open(label_path, 'rb') as f:
            labels = pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load labels.pkl: {e}")
        labels = {}
else:
    labels = {}

# User input
user_id = input("Enter Your ID: ")
name = input("Enter Your Name: ")
label = f"{user_id}_{name}"

#  Duplicate ID Check
if user_id in labels:
    print(f"❌ ID {user_id} is already registered to {labels[user_id]}")
    exit()

# Start Video & Models
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Variables for Liveness Detection
blink_counter = 0
head_moved = False
initial_nose_x = None

# Detect blink and head movement
def detect_liveness(frame):
    global blink_counter, head_moved, initial_nose_x

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            top = face_landmarks.landmark[159]
            bottom = face_landmarks.landmark[145]
            left_eye_left = face_landmarks.landmark[33]
            left_eye_right = face_landmarks.landmark[133]
            nose = face_landmarks.landmark[1]

            ear = abs(top.y - bottom.y) / (abs(left_eye_left.x - left_eye_right.x) + 1e-6)
            if ear < 0.25:
                blink_counter += 1
                print("😉 Blink Detected")

            if initial_nose_x is None:
                initial_nose_x = nose.x
            elif abs(nose.x - initial_nose_x) > 0.02:
                head_moved = True
                print("👈👉 Head Movement Detected")
    return blink_counter >= 1 and head_moved

def is_duplicate(temp_face_path):
    for existing_file in os.listdir(dataset_dir):
        existing_path = os.path.join(dataset_dir, existing_file)
        try:
            result = DeepFace.verify(temp_face_path, existing_path, model_name='Facenet', enforce_detection=False)
            if result['verified'] and result['distance'] < 0.6:
                print(f"✅ Match found with: {existing_file}")
                return True
        except Exception as e:
            print(f"[WARNING] Verification error: {e}")
            continue
    return False

print("[INFO] Show your face, blink, and move head slightly to start registration...")
captured_faces = []
frame_count = 0
max_frames = 50

while True:
    ret, frame = video.read()
    if not ret:
        continue

    liveness_passed = detect_liveness(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    print(f"[DEBUG] Liveness Passed: {liveness_passed} | Faces Detected: {len(faces)} | Frames Captured: {frame_count}")

    for (x, y, w, h) in faces:
        crop_face = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if liveness_passed and frame_count < max_frames:
            crop_face_resized = cv2.resize(crop_face, (160, 160))
            captured_faces.append(crop_face_resized)
            frame_count += 1
            print(f"[INFO] Captured frame {frame_count}/{max_frames}")
            time.sleep(0.1)

    cv2.putText(frame, f"Blinks: {blink_counter} | Head Moved: {'Yes' if head_moved else 'No'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) == ord('q') or frame_count >= max_frames:
        break

video.release()
cv2.destroyAllWindows()

if frame_count >= max_frames:
    avg_face = np.mean(np.array(captured_faces), axis=0).astype(np.uint8)
    temp_path = "temp_face.jpg"
    cv2.imwrite(temp_path, avg_face)

    if is_duplicate(temp_path):
        print("❌ This face is already registered.")
        os.remove(temp_path)
    else:
        save_path = os.path.join(dataset_dir, f"{label}.jpg")
        cv2.imwrite(save_path, avg_face)
        labels[user_id] = name
        with open(label_path, 'wb') as f:
            pickle.dump(labels, f)
        print("✅ Face registered successfully.")
else:
    print("❌ Not enough frames captured for registration.")

# File: add_faces_gradio.py




