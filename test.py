
import cv2
import os
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.neighbors import KNeighborsClassifier
import time
from datetime import datetime
import csv
from win32com.client import Dispatch
import mediapipe as mp

# Text-to-Speech Setup
def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Load label dictionary
label_path = "data/labels.pkl"
if not os.path.exists(label_path):
    print("❌ labels.pkl not found.")
    exit()

with open(label_path, 'rb') as f:
    label_dict = pickle.load(f)

# Load embeddings and labels
print("🔄 Loading embeddings...")
embeddings = []
labels = []

for file in os.listdir("saved_faces"):
    if file.endswith(".jpg"):
        user_id = file.split("_")[0]
        name = label_dict.get(user_id, "Unknown")
        path = os.path.join("saved_faces", file)
        try:
            rep = DeepFace.represent(img_path=path, model_name='Facenet', enforce_detection=False)
            if rep and 'embedding' in rep[0]:
                embeddings.append(rep[0]['embedding'])
                labels.append(f"{user_id}_{name}")
        except Exception as e:
            print(f"⚠ Error generating embedding for {file}: {e}")

if not embeddings:
    print("❌ No embeddings found. Make sure you have registered faces.")
    exit()
print(f"✅ Loaded {len(embeddings)} embeddings.")

# Train KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(embeddings, labels)

# Camera setup
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Blink & movement detection
prev_x_center = None
blink_counter = 0
head_movement_detected = False

def detect_liveness(frame):
    global prev_x_center, blink_counter, head_movement_detected
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Blink
            top = face_landmarks.landmark[159]
            bottom = face_landmarks.landmark[145]
            left = face_landmarks.landmark[33]
            right = face_landmarks.landmark[133]
            ear = abs(top.y - bottom.y) / (abs(left.x - right.x) + 1e-6)
            if ear < 0.25:
                blink_counter += 1

            # Head movement
            nose = face_landmarks.landmark[1]
            x_center = nose.x
            if prev_x_center is not None and abs(x_center - prev_x_center) > 0.02:
                head_movement_detected = True
            prev_x_center = x_center

start_time = time.time()
frame_name = None
frame_id = None
found_embedding = None
crop_for_attendance = None

print("📷 Starting live preview. Look into the camera...")

# Preview for 10 seconds
while time.time() - start_time < 10:
    ret, frame = video.read()
    if not ret:
        continue

    detect_liveness(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for i, (x, y, w, h) in enumerate(faces):
        crop = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        crop_debug_path = f"debug_crop_{i}.jpg"
        cv2.imwrite(crop_debug_path, crop)  # Save crop for manual inspection

        try:
            rep = DeepFace.represent(crop, model_name='Facenet', enforce_detection=False)
            if rep and 'embedding' in rep[0]:
                embedding = rep[0]['embedding']
                output = knn.predict([embedding])[0]
                distance = knn.kneighbors([embedding])[0][0][0]
                print(f"🔍 Match: {output}, Distance: {distance:.2f}")

                if distance < 12:  # Increased threshold from 10
                    frame_id, frame_name = output.split("_", 1)
                    label_text = f"{frame_name} ({frame_id})"
                    cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    found_embedding = embedding
                    crop_for_attendance = crop
                else:
                    print("⚠ Face detected but not matched (distance too high).")
            else:
                print("❌ Failed to generate embedding for current face.")
        except Exception as e:
            print(f"❌ Exception during face recognition: {e}")

    cv2.putText(frame, f"Blinks: {blink_counter} | Head Moved: {head_movement_detected}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Preview", frame)

    if cv2.waitKey(1) == ord('q'):
        video.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Preview")

# Final liveness check
if blink_counter == 0 or not head_movement_detected:
    speak("Liveness check failed. Please blink and move head.")
    print("❌ Liveness check failed.")
    video.release()
    exit()

# Attendance logic
if found_embedding is not None:
    now = datetime.now()
    date = now.strftime("%d-%m-%Y")
    timestamp = now.strftime("%H:%M:%S")
    current_hour = now.hour

    row = [frame_name, frame_id, timestamp, date]
    os.makedirs("Attendance", exist_ok=True)
    file_path = f"Attendance/Attendance_{date}.csv"
    exists = os.path.exists(file_path)

    entry_count = 0
    if exists:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for r in reader:
                if len(r) >= 2 and r[0] == frame_name and r[1] == frame_id:
                    entry_count += 1

    if entry_count >= 2:
        speak("You have already checked in and out")
        print("⚠ Already 2 entries today.")
    else:
        status = "Check-in" if entry_count == 0 else "Check-out"

        if status == "Check-in" and current_hour >= 15:
            speak("Check-in time is over. Check-in allowed only before 3:00 PM.")
            print("❌ Check-in denied.")
        elif status == "Check-out" and current_hour < 13:
            speak("Check-out not allowed before 1:00 PM.")
            print("❌ Check-out denied.")
        else:
            row.append(status)
            speak(f"Attendance taken for {frame_name} {status}")
            with open(file_path, "a", newline='') as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(['NAME', 'USER_ID', 'TIME', 'DATE', 'STATUS'])
                writer.writerow(row)

            saved_path = f"saved_faces/{frame_id}_{frame_name}.jpg"
            if os.path.exists(saved_path):
                saved_img = cv2.imread(saved_path)
                saved_img = cv2.resize(saved_img, (300, 300))
                cv2.putText(saved_img, f"{frame_name} ({frame_id})", (10, 290),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Attendance Confirmed", saved_img)
                cv2.waitKey(4000)
                cv2.destroyWindow("Attendance Confirmed")

            print("✅ Attendance Taken")
            print(f"Name  : {frame_name}")
            print(f"UserID: {frame_id}")
            print(f"Time  : {timestamp}")
            print(f"Date  : {date}")
            print(f"Status: {status}")
else:
    speak("Face not recognized")
    print("❌ Face not recognized")

video.release()
cv2.destroyAllWindows()
