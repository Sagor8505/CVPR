import os
import cv2
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime

# Disable oneDNN float warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ---- Load trained model ----
MODEL_PATH = r"C:\Users\Asus\Downloads\cvpr\Assignment2\alexnet_attendance.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

# ---- Load label encoder ----
ENCODER_PATH = r"C:\Users\Asus\Downloads\cvpr\Assignment2\label_encoder.pkl"
with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

students = list(encoder.classes_)
print("Students:", students)

# ---- Attendance CSV file ----
attendance_file = r"C:\Users\Asus\Downloads\cvpr\Assignment2\attendance_live.csv"

# Create CSV if not exists
if not os.path.exists(attendance_file):
    df_init = pd.DataFrame(columns=["Name", "Time", "RunSession"])
    df_init.to_csv(attendance_file, index=False)

# Unique run session ID for this webcam start
run_session = datetime.now().strftime("%Y%m%d_%H%M%S")
print("Current run session:", run_session)

# Track already-marked names in this session
marked_this_session = set()

# ---- Load face detector ----
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

print("Starting webcam attendance... Press ESC to quit.")
cap = cv2.VideoCapture(0)

IMG_SIZE = 227

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 4)

    # Show number of detected faces
    cv2.putText(frame, f"Faces: {len(faces)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), 2)

        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE)) / 255.0
        face = face.reshape(1, IMG_SIZE, IMG_SIZE, 3)

        pred = model.predict(face, verbose=0)
        predicted_name = encoder.inverse_transform([np.argmax(pred)])[0]

        # If this person is not yet marked in this session, mark them
        if predicted_name not in marked_this_session:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(attendance_file, "a") as f:
                f.write(f"{predicted_name},{now},{run_session}\n")
            print("Marked present:", predicted_name)

        # Add to marked set so same person isn't logged again in this session
        marked_this_session.add(predicted_name)

    cv2.imshow("Attendance Cam", frame)

    if cv2.waitKey(1) == 27:  # ESC
        print("Exiting webcam...")
        break

cap.release()
cv2.destroyAllWindows()

# ---- Print attendance log after webcam closes ----
df_final = pd.read_csv(attendance_file)
print("\nAttendance log:")
print(df_final)
