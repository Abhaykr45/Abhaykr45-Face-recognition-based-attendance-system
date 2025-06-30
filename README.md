# Abhaykr45-Face-recognition-based-attendance-system
This project is an AI-powered attendance system that uses face recognition and real-time liveness detection to automatically record attendance securely and efficiently
The system captures the user’s face using a webcam and identifies them using the Facenet model from the DeepFace library. To prevent spoofing through photos or videos, it includes liveness detection using MediaPipe, which verifies natural actions like eye blinks and head movements before allowing attendance.

🔍 Features
- 🎥 Real-time webcam-based Face Recognition using DeepFace (Facenet)

- 👁️‍🗨️ Liveness Detection with MediaPipe (eye blink & head movement)

- 🧠 Prevents spoofing using photos/videos

- 🚫 Duplicate face detection during registration

- 🕒 Time-based attendance rules (Check-in before 6 PM, Check-out after 6 PM)

- 📄 Attendance logs saved in structured CSV files

- 📊 Gradio dashboard for live attendance monitoring with search and auto-refresh

- 📸 Stores captured face images for verification

- 🔐 Only two entries (Check-in & Check-out) allowed per user per day

