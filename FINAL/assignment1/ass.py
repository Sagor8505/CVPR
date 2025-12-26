import cv2
import numpy as np
import tensorflow as tf
import os
from collections import deque

# ------------------ Load Model ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_digit_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")


# ------------------ Prediction Smoothing ------------------
prediction_buffer = deque(maxlen=15)

# ------------------ Webcam ------------------
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1)  # use if external webcam

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Put the digit inside the box. Press 'Esc' to quit.")

# ------------------ Main Loop ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Default values (VERY IMPORTANT)
    box_color = (0, 255, 255)
    label = "Digit: ?"
    display_digit = None

    frame_h, frame_w = frame.shape[:2]

    # ROI box
    box_size = 300
    x1 = frame_w // 2 - box_size // 2
    y1 = frame_h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2]

    # ------------------ Preprocessing ------------------
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 2)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 900:
            x, y, w, h = cv2.boundingRect(largest)

            if w > 20 and h > 20:
                digit = thresh[y:y+h, x:x+w]

                # Make square image
                size = max(w, h)
                square = np.zeros((size, size), dtype=np.uint8)
                x_off = (size - w) // 2
                y_off = (size - h) // 2
                square[y_off:y_off+h, x_off:x_off+w] = digit

                # Resize
                digit_large = cv2.resize(square, (256, 256))
                digit_small = cv2.resize(digit_large, (28, 28))

                # Normalize & reshape for CNN
                digit_norm = digit_small.astype("float32") / 255.0
                digit_input = digit_norm.reshape(1, 28, 28, 1)

                # Prediction
                pred = model.predict(digit_input, verbose=0)
                current_pred = int(np.argmax(pred))
                confidence = float(pred[0][current_pred])

                prediction_buffer.append(current_pred)

                predicted_digit = max(
                    set(prediction_buffer),
                    key=prediction_buffer.count
                )

                label = f"Digit: {predicted_digit} | Confidence: {confidence:.2f}"

                box_color = (0, 255, 0) if confidence >= 0.70 else (0, 0, 255)
                display_digit = digit_large

    # ------------------ Draw UI ------------------
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    cv2.putText(
        frame,
        label,
        (x1 + 10, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        box_color,
        2
    )

    if display_digit is not None:
        cv2.imshow("Processed Digit", display_digit)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == 27:
        break

# ------------------ Cleanup ------------------
cap.release()
cv2.destroyAllWindows()
