
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort


# Find FFmpeg automatically if it is available in PATH.
# Otherwise, use the known Windows installation path.
KNOWN_FFMPEG_PATH = (
    r"C:\Users\admin\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.1-full_build-shared\bin\ffmpeg.exe"
)

FFMPEG_PATH = shutil.which("ffmpeg") or KNOWN_FFMPEG_PATH

# Keep aasist.onnx in the same folder as this Python file.
PROJECT_DIR = Path(__file__).resolve().parent
AASIST_MODEL = PROJECT_DIR / "aasist.onnx"

TARGET_SAMPLES = 64600
TARGET_SAMPLE_RATE = 16000


def analyze_video_audio(video_path):
    """
    Extracts audio from a video and runs AASIST ONNX inference.

    Returns raw class logits. It does not claim a REAL/FAKE
    classification until the ONNX class order is verified.
    """

    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    if not AASIST_MODEL.is_file():
        raise FileNotFoundError(
            f"AASIST model not found: {AASIST_MODEL}"
        )

    if not Path(FFMPEG_PATH).is_file() and not shutil.which("ffmpeg"):
        raise FileNotFoundError(
            "FFmpeg was not found. Install FFmpeg or correct "
            "KNOWN_FFMPEG_PATH in audio_utils.py."
        )

    # Use a temporary WAV file so it is removed automatically.
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as temp_file:
        audio_path = temp_file.name

    try:
        # Extract mono audio at 16 kHz.
        command = [
            FFMPEG_PATH,
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", str(TARGET_SAMPLE_RATE),
            "-y",
            audio_path
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        # Load audio as mono float32 at 16 kHz.
        audio, sample_rate = librosa.load(
            audio_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True
        )

        if audio.size == 0:
            raise ValueError(
                "The uploaded video contains no usable audio."
            )

        audio_duration = len(audio) / sample_rate

        # Official AASIST input window: first 64,600 samples.
        chunk = audio[:TARGET_SAMPLES]

        # Pad short audio with zeros.
        if len(chunk) < TARGET_SAMPLES:
            chunk = np.pad(
                chunk,
                (0, TARGET_SAMPLES - len(chunk)),
                mode="constant"
            )

        audio_input = np.asarray(
            chunk,
            dtype=np.float32
        ).reshape(1, TARGET_SAMPLES)

        # Load and run ONNX model.
        session = ort.InferenceSession(
            str(AASIST_MODEL),
            providers=["CPUExecutionProvider"]
        )

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        output = session.run(
            [output_name],
            {input_name: audio_input}
        )

        logits = np.asarray(output[0])

        if logits.shape != (1, 2):
            raise ValueError(
                f"Unexpected AASIST output shape: {logits.shape}. "
                "Expected (1, 2)."
            )

        class_0_logit = float(logits[0][0])
        class_1_logit = float(logits[0][1])

        return {
            "audio_duration": round(audio_duration, 2),
            "audio_chunks_analyzed": 1,
            "class_0_logit": round(class_0_logit, 6),
            "class_1_logit": round(class_1_logit, 6),
            "mean_class_0_logit": round(class_0_logit, 6),
            "mean_class_1_logit": round(class_1_logit, 6),

            # Intentionally unset until output class order
            # is verified against the exact ONNX export.
            "mean_bona_fide_logit": None,
            "bona_fide_class_index_verified": False,

            "notice": (
                "Raw AASIST logits only. Class order has not "
                "yet been verified; this is not a probability "
                "or a confirmed REAL/FAKE verdict."
            )
        }

    finally:
        # Remove the temporary extracted audio.
        if os.path.exists(audio_path):
            os.remove(audio_path)
