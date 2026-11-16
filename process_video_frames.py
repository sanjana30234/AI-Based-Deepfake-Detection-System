import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "model/sih1683_mobilenet_finetuned_90_28.h5"

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")

# ============================================================
# CREATE FACE DETECTOR
# ============================================================

detector = MTCNN()

# Store predictions from successfully detected faces
predictions = []

# Store frame numbers that were successfully analyzed
analyzed_frames = []

# ============================================================
# PROCESS 20 EXTRACTED FRAMES
# ============================================================

for i in range(20):

    frame_path = f"frame_{i}.jpg"

    print("\n-----------------------------")
    print(f"Processing {frame_path}")

    # Check whether frame exists
    if not os.path.exists(frame_path):
        print("Frame not found. Skipping...")
        continue

    # Read image
    img = cv2.imread(frame_path)

    if img is None:
        print("Could not read frame. Skipping...")
        continue

    # Convert BGR → RGB for MTCNN
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ========================================================
    # FACE DETECTION
    # ========================================================

    faces = detector.detect_faces(rgb_img)

    print("Faces found:", len(faces))

    if len(faces) == 0:
        print("No face detected. Skipping...")
        continue

    # ========================================================
    # SELECT LARGEST FACE
    # ========================================================

    face = max(
        faces,
        key=lambda f: f["box"][2] * f["box"][3]
    )

    x, y, w, h = face["box"]

    # Prevent negative coordinates
    x = max(0, x)
    y = max(0, y)

    # Get image dimensions
    h_img, w_img = img.shape[:2]

    # Prevent crop from going outside image
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    # Check crop dimensions
    if w <= 0 or h <= 0:
        print("Invalid face crop. Skipping...")
        continue

    # ========================================================
    # CROP FACE
    # ========================================================

    face_crop = img[y:y+h, x:x+w]

    if face_crop.size == 0:
        print("Empty face crop. Skipping...")
        continue

    # Resize to model input size
    face_crop = cv2.resize(
        face_crop,
        (224, 224)
    )

    # Convert BGR → RGB
    face_crop = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB
    )

    # ========================================================
    # PREPARE IMAGE FOR MODEL
    # ========================================================

    img_array = image.img_to_array(face_crop)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # IMPORTANT:
    # Do NOT divide by 255.
    # The trained model already contains a Rescaling layer.

    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if prediction > 0.5:

        result = "REAL"
        confidence = prediction * 100

    else:

        result = "FAKE"
        confidence = (1 - prediction) * 100

    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Raw prediction: {prediction:.4f}")

    # Store prediction
    predictions.append(prediction)

    analyzed_frames.append(i)


# ============================================================
# FINAL VIDEO ANALYSIS
# ============================================================

print("\n===================================")
print("FINAL VIDEO ANALYSIS")
print("===================================")

if len(predictions) == 0:

    print("No usable faces were detected.")
    print("Unable to determine whether the video is REAL or FAKE.")

else:

    # Average prediction across detected faces
    average_prediction = np.mean(predictions)

    # Count REAL and FAKE predictions
    real_count = sum(
        p > 0.5 for p in predictions
    )

    fake_count = sum(
        p <= 0.5 for p in predictions
    )

    total = len(predictions)

    # Calculate average confidence
    if average_prediction > 0.5:

        final_result = "REAL"
        final_confidence = average_prediction * 100

    else:

        final_result = "FAKE"
        final_confidence = (1 - average_prediction) * 100

    # ========================================================
    # DISPLAY FINAL RESULT
    # ========================================================

    print(f"Frames successfully analyzed: {total}")
    print(f"Frames skipped: {20 - total}")
    print(f"REAL frames: {real_count}")
    print(f"FAKE frames: {fake_count}")
    print(
        f"Average raw prediction: "
        f"{average_prediction:.4f}"
    )

    print("-----------------------------------")
    print(f"FINAL RESULT: {final_result}")
    print(
        f"FINAL CONFIDENCE: "
        f"{final_confidence:.2f}%"
    )
    print("===================================")