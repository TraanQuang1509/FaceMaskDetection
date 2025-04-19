import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.lite.python.interpreter import Interpreter
import time

prev_time = 0
# Load trained mask classifier
# full model
# model = tf.keras.models.load_model("mask_classifier_mnv2.keras")

# lite model
interpreter = Interpreter(model_path="best_model_lite.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["incorrect_mask", "with_mask", "without_mask"]
IMG_SIZE = 128

# Khởi tạo Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Mở webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển sang RGB (mediapipe dùng RGB)
    frame = cv2.resize(frame, (480, 360))
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            x1 = int(x - 0.0*w)
            y1 = int(y - 0.2*h)
            x2 = int(x + w + 0.0*w)
            y2 = int(y + h + 0.2*h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            # full model
            # face_input = np.expand_dims(face_resized / 255.0, axis=0)
            # preds = model.predict(face_input, verbose=0)

            # lite model
            face_input = np.expand_dims(face_resized / 255.0, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], face_input)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])

            class_id = np.argmax(preds)
            label = class_names[class_id]
            conf = preds[0][class_id]

            color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)
            label_text = f"{label} ({conf*100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Face Mask Detection (MediaPipe)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()