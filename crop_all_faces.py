import cv2
from mtcnn import MTCNN
import os

detector = MTCNN()

# Loop through frame_0.jpg to frame_19.jpg
for i in range(20):
    image_path = f'frame_{i}.jpg'
    
    # Skip if this frame doesn't exist
    if not os.path.exists(image_path):
        continue
    
    img = cv2.imread(image_path)
    faces = detector.detect_faces(img)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]['box']
        # Make sure the box values aren't negative (can happen near edges)
        x, y = max(0, x), max(0, y)
        
        face_crop = img[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, (224, 224))
        
        output_path = f'cropped_faces/face_{i}.jpg'
        cv2.imwrite(output_path, face_crop)
        print(f"Frame {i}: face found and saved.")
    else:
        print(f"Frame {i}: no face found, skipped.")

print("Done processing all frames!")