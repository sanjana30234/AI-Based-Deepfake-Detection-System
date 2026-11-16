import librosa
import numpy as np
import onnxruntime as ort


AUDIO_FILE = "test_audio.wav"
MODEL_FILE = "aasist.onnx"

TARGET_SAMPLES = 64600
HOP_LENGTH = 32300


# Load audio
y, sr = librosa.load(
    AUDIO_FILE,
    sr=16000,
    mono=True
)

print("Audio loaded successfully")
print("Sample rate:", sr)
print("Duration:", round(len(y) / sr, 2), "seconds")


# Load AASIST model
session = ort.InferenceSession(MODEL_FILE)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# Create overlapping audio chunks
starts = sorted(
    set(
        list(
            range(
                0,
                max(1, len(y) - TARGET_SAMPLES + 1),
                HOP_LENGTH
            )
        )
        + [max(0, len(y) - TARGET_SAMPLES)]
    )
)


scores = []


# Analyze each chunk
for start in starts:

    chunk = y[start:start + TARGET_SAMPLES]

    if len(chunk) < TARGET_SAMPLES:
        chunk = np.pad(
            chunk,
            (0, TARGET_SAMPLES - len(chunk))
        )

    audio_input = (
        chunk
        .astype(np.float32)
        .reshape(1, TARGET_SAMPLES)
    )

    output = session.run(
        [output_name],
        {input_name: audio_input}
    )

    bona_fide_logit = float(output[0][0][0])

    scores.append(bona_fide_logit)

    print(
        f"Chunk {len(scores)} "
        f"bona-fide logit: {bona_fide_logit:.4f}"
    )


# Average score
mean_score = float(np.mean(scores))


print("\nAudio analysis complete")
print("Chunks analyzed:", len(scores))
print("Bona-fide logits:", [round(x, 4) for x in scores])
print("Mean bona-fide logit:", round(mean_score, 4))