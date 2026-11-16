import cv2
import os

video_path = "WIN_20260822_13_48_00_Pro.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Total frames in video:", total_frames)

# Delete old extracted frames
for i in range(20):
    old_file = f"frame_{i}.jpg"
    if os.path.exists(old_file):
        os.remove(old_file)

# Select 20 frames across the entire video
frame_positions = [
    int(i * (total_frames - 1) / 19)
    for i in range(20)
]

saved = 0

for position in frame_positions:

    cap.set(cv2.CAP_PROP_POS_FRAMES, position)

    ret, frame = cap.read()

    if not ret:
        print(f"Could not read frame at position {position}")
        continue

    filename = f"frame_{saved}.jpg"

    cv2.imwrite(filename, frame)

    print(f"Saved {filename} from video frame {position}")

    saved += 1

cap.release()

print("--------------------------------")
print(f"Done! Saved {saved} frames across the entire video.")
print("--------------------------------")
