
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

from werkzeug.utils import secure_filename

from tf_keras.models import load_model
from tf_keras.preprocessing import image

from audio_utils import analyze_video_audio

import numpy as np
import os
import uuid
import cv2

from mtcnn import MTCNN


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SUSPICIOUS_FOLDER = os.path.join(BASE_DIR, "static", "suspicious")
FRAMES_FOLDER = os.path.join(BASE_DIR, "static", "frames")
CROPPED_FACES_FOLDER = os.path.join(BASE_DIR, "static", "cropped_faces")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "sih1683_mobilenet_finetuned_90_28.h5"
)

for folder in [
    UPLOAD_FOLDER,
    SUSPICIOUS_FOLDER,
    FRAMES_FOLDER,
    CROPPED_FACES_FOLDER
]:
    os.makedirs(folder, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading deepfake detection model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Deepfake detection model loaded successfully!")


# =========================================================
# FACE DETECTOR
# =========================================================

detector = MTCNN()


# =========================================================
# IMAGE ANALYSIS
# =========================================================

def analyze_image(file_path):

    img = image.load_img(
        file_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = float(
        model.predict(
            img_array,
            verbose=0
        )[0][0]
    )

    if prediction > 0.5:
        result = "REAL"
        confidence = prediction * 100
    else:
        result = "FAKE"
        confidence = (1 - prediction) * 100

    return result, float(confidence)


# =========================================================
# CLEAR OLD GENERATED VIDEO FILES
# =========================================================

def clear_video_output_folders():

    folders = [
        SUSPICIOUS_FOLDER,
        FRAMES_FOLDER,
        CROPPED_FACES_FOLDER
    ]

    for folder in folders:

        for filename in os.listdir(folder):

            old_file = os.path.join(
                folder,
                filename
            )

            if os.path.isfile(old_file):
                try:
                    os.remove(old_file)
                except OSError as error:
                    print("FILE CLEANUP ERROR:", error)


# =========================================================
# VIDEO ANALYSIS
# =========================================================

def analyze_video(file_path):

    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        return None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if total_frames <= 0:
        cap.release()
        return None

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    video_duration = total_frames / fps

    clear_video_output_folders()

    # Analyze up to 30 frames
    number_of_samples = min(30, total_frames)

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        number_of_samples,
        dtype=int
    )

    predictions = []

    real_frames = []
    fake_frames = []

    faces_analyzed = 0

    suspicious_frame_files = []
    frame_files = []
    cropped_face_files = []

    suspicious_by_frame = {}

    # -----------------------------------------------------
    # PROCESS VIDEO FRAMES
    # -----------------------------------------------------

    try:

        for frame_number in frame_indices:

            frame_number = int(frame_number)

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )

            success, frame = cap.read()

            if not success:
                continue

            timestamp = frame_number / fps

            # Save original frame
            frame_filename = f"frame_{frame_number}.jpg"

            frame_path = os.path.join(
                FRAMES_FOLDER,
                frame_filename
            )

            cv2.imwrite(
                frame_path,
                frame
            )

            frame_files.append(frame_filename)

            # Convert to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Detect faces
            faces = detector.detect_faces(rgb_frame)

            if len(faces) == 0:
                continue

            # -------------------------------------------------
            # ANALYZE EACH FACE
            # -------------------------------------------------

            for face_index, face in enumerate(faces):

                x, y, w, h = face["box"]

                frame_height, frame_width = frame.shape[:2]

                x = max(0, x)
                y = max(0, y)

                w = min(w, frame_width - x)
                h = min(h, frame_height - y)

                if w <= 0 or h <= 0:
                    continue

                face_crop = frame[
                    y:y + h,
                    x:x + w
                ]

                if face_crop.size == 0:
                    continue

                # Save cropped face
                cropped_filename = (
                    f"crop_{frame_number}_face_{face_index}.jpg"
                )

                cropped_path = os.path.join(
                    CROPPED_FACES_FOLDER,
                    cropped_filename
                )

                cv2.imwrite(
                    cropped_path,
                    face_crop
                )

                cropped_face_files.append(cropped_filename)

                # Resize face
                face_crop = cv2.resize(
                    face_crop,
                    (224, 224)
                )

                face_crop = cv2.cvtColor(
                    face_crop,
                    cv2.COLOR_BGR2RGB
                )

                img_array = image.img_to_array(face_crop)

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # Predict
                prediction = float(
                    model.predict(
                        img_array,
                        verbose=0
                    )[0][0]
                )

                predictions.append(prediction)

                faces_analyzed += 1

                # Classify
                if prediction > 0.5:

                    real_frames.append(frame_number)

                else:

                    fake_frames.append(frame_number)

                    suspicious_filename = (
                        f"frame_{frame_number}.jpg"
                    )

                    suspicious_path = os.path.join(
                        SUSPICIOUS_FOLDER,
                        suspicious_filename
                    )

                    if not os.path.exists(suspicious_path):

                        cv2.imwrite(
                            suspicious_path,
                            frame
                        )

                        suspicious_frame_files.append(
                            suspicious_filename
                        )

                    suspicious_by_frame.setdefault(
                        frame_number,
                        []
                    ).append(prediction)

    finally:
        cap.release()

    # -----------------------------------------------------
    # NO FACES FOUND
    # -----------------------------------------------------

    if len(predictions) == 0:

        return {
            "result": "UNKNOWN",
            "confidence": 0,
            "total_frames": total_frames,
            "analyzed_frames": 0,
            "real_frames": 0,
            "fake_frames": 0,
            "faces_analyzed": 0,
            "fps": fps,
            "video_duration": round(video_duration, 2),
            "suspicious_frames": [],
            "suspicious_files": [],
            "suspicious_timeline": [],
            "frame_files": frame_files,
            "cropped_face_files": cropped_face_files
        }

    # -----------------------------------------------------
    # FINAL VIDEO RESULT
    # -----------------------------------------------------

    average_prediction = float(np.mean(predictions))

    real_count = len(real_frames)
    fake_count = len(fake_frames)

    if average_prediction > 0.5:

        final_result = "REAL"
        final_confidence = average_prediction * 100

    else:

        final_result = "FAKE"
        final_confidence = (1 - average_prediction) * 100

    # -----------------------------------------------------
    # BUILD SUSPICIOUS TIMELINE
    # -----------------------------------------------------

    suspicious_timeline = []

    for frame_number in sorted(suspicious_by_frame):

        timestamp = frame_number / fps

        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)

        frame_filename = f"frame_{frame_number}.jpg"

        frame_scores = suspicious_by_frame[frame_number]

        average_frame_score = float(
            np.mean(frame_scores)
        )

        suspicious_timeline.append({

            "frame_number": frame_number,

            "timestamp": round(timestamp, 2),

            "time_label": f"{minutes:02d}:{seconds:02d}",

            "filename": frame_filename,

            "model_score": average_frame_score

        })

    # -----------------------------------------------------
    # RETURN VIDEO ANALYSIS
    # -----------------------------------------------------

    return {

        "result": final_result,

        "confidence": float(final_confidence),

        "total_frames": total_frames,

        "analyzed_frames": len(predictions),

        "real_frames": real_count,

        "fake_frames": fake_count,

        "faces_analyzed": faces_analyzed,

        "fps": fps,

        "video_duration": round(video_duration, 2),

        "suspicious_frames": sorted(set(fake_frames)),

        "suspicious_files": suspicious_frame_files,

        "suspicious_timeline": suspicious_timeline,

        "frame_files": frame_files,

        "cropped_face_files": cropped_face_files

    }


# =========================================================
# SERVE UPLOADED FILES
# =========================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# =========================================================
# HOME / UPLOAD / ANALYSIS ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "GET":

        return render_template(
            "index.html",
            error=None
        )

    # -----------------------------------------------------
    # CHECK UPLOADED FILE
    # -----------------------------------------------------

    if "file" not in request.files:

        return render_template(
            "index.html",
            error="No file selected."
        )

    file = request.files["file"]

    if file.filename == "":

        return render_template(
            "index.html",
            error="Please select an image or video."
        )

    original_filename = file.filename

    safe_filename = secure_filename(original_filename)

    if not safe_filename:

        return render_template(
            "index.html",
            error="Invalid filename."
        )

    # Unique filename
    saved_filename = (
        f"{uuid.uuid4().hex}_{safe_filename}"
    )

    file_path = os.path.join(
        UPLOAD_FOLDER,
        saved_filename
    )

    file.save(file_path)

    filename = safe_filename.lower()

    result = None
    confidence = None

    video_analysis = None
    uploaded_filename = None

    # -----------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------

    try:

        # ================================================
        # IMAGE ANALYSIS
        # ================================================

        if filename.endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):

            result, confidence = analyze_image(file_path)

            uploaded_filename = saved_filename

            print("IMAGE:", original_filename)
            print("RESULT:", result)
            print("CONFIDENCE:", confidence)

        # ================================================
        # VIDEO ANALYSIS
        # ================================================

        elif filename.endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        ):

            video_analysis = analyze_video(file_path)

            if video_analysis is None:

                return render_template(
                    "index.html",
                    error="Could not read the video."
                )

            if video_analysis["analyzed_frames"] == 0:

                return render_template(
                    "index.html",
                    error="No usable faces were detected in the video."
                )

            video_analysis["video_filename"] = saved_filename

            result = video_analysis["result"]

            confidence = video_analysis["confidence"]

            # ============================================
            # OPTIONAL AUDIO ANALYSIS
            # ============================================

            try:

                audio_analysis = analyze_video_audio(file_path)

                video_analysis["audio_analysis"] = audio_analysis

                print("AUDIO ANALYSIS:", audio_analysis)

            except Exception as audio_error:

                print("AUDIO ANALYSIS ERROR:", repr(audio_error))

                video_analysis["audio_analysis"] = None

            print("VIDEO:", original_filename)
            print("RESULT:", result)
            print("CONFIDENCE:", confidence)

            print(
                "FRAMES ANALYZED:",
                video_analysis["analyzed_frames"]
            )

            print(
                "FACES ANALYZED:",
                video_analysis["faces_analyzed"]
            )

            print(
                "SUSPICIOUS TIMELINE ENTRIES:",
                len(video_analysis["suspicious_timeline"])
            )

        # ================================================
        # UNSUPPORTED FILE
        # ================================================

        else:

            return render_template(
                "index.html",
                error=(
                    "Unsupported file type. "
                    "Please upload JPG, PNG, MP4, AVI, MOV or MKV."
                )
            )

    except Exception as error:

        print("ANALYSIS ERROR:", repr(error))

        import traceback
        traceback.print_exc()

        return render_template(
            "index.html",
            error=f"An error occurred while analyzing the file: {error}"
        )

    # -----------------------------------------------------
    # IMPORTANT: RENDER RESULTS DIRECTLY
    # No temporary result dictionary or redirect required.
    # -----------------------------------------------------

    return render_template(
        "results.html",
        result=result,
        confidence=confidence,
        video_analysis=video_analysis,
        uploaded_filename=uploaded_filename
    )


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        error="The uploaded file is too large. Maximum size is 500 MB."
    ), 413


@app.errorhandler(400)
def handle_bad_request(error):

    print("HTTP 400 ERROR:", repr(error))

    return render_template(
        "index.html",
        error="The upload request was rejected. Please try again."
    ), 400


# =========================================================
# START FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
