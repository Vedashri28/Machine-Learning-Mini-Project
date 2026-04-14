import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import winsound
import urllib.request
import os

# ── Download model if missing ──────────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model (~30 MB), please wait...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded!")

# ── New mediapipe Tasks API ────────────────────────────────────────────────────
BaseOptions         = mp.tasks.BaseOptions
FaceLandmarker      = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode   = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

# ── Eye landmark indices (MediaPipe 468-point mesh) ────────────────────────────
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def get_eye_ratio(landmarks, eye_indices, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
              for i in eye_indices]
    horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    vertical   = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    return vertical / horizontal if horizontal else 0

# ── Camera + tracking setup ────────────────────────────────────────────────────
cap          = cv2.VideoCapture(0)
focused_time = 0
total_time   = 0
start_time   = time.time()

file   = open("focus_report.csv", "w", newline="")
writer = csv.writer(file)
writer.writerow(["Time", "Status", "Focus %"])

# ── Main loop ─────────────────────────────────────────────────────────────────
with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect(mp_image)
        status = "No Face"

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            left_ratio  = get_eye_ratio(lm, LEFT_EYE,  w, h)
            right_ratio = get_eye_ratio(lm, RIGHT_EYE, w, h)
            avg_ratio   = (left_ratio + right_ratio) / 2

            if avg_ratio < 0.20:
                status = "Sleeping"
                winsound.Beep(1000, 300)
            elif avg_ratio < 0.25:
                status = "Distracted"
                winsound.Beep(800, 200)
            else:
                status = "Focused"

        # ── Tracking ───────────────────────────────────────────────────────────
        total_time += 1
        if status == "Focused":
            focused_time += 1

        focus_score  = (focused_time / total_time * 100) if total_time else 0
        elapsed_time = int(time.time() - start_time)

        writer.writerow([elapsed_time, status, int(focus_score)])

        # ── Display ────────────────────────────────────────────────────────────
        color = (0, 255, 0) if status == "Focused" else \
                (0, 165, 255) if status == "Distracted" else (0, 0, 255)

        cv2.putText(frame, status,                    (50,  50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Focus: {int(focus_score)}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Time:  {elapsed_time}s", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("AI Focus Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
            break

cap.release()
file.close()
cv2.destroyAllWindows()
print(f"\nSession ended. Focus score: {int(focus_score)}%")
print("Report saved to focus_report.csv")