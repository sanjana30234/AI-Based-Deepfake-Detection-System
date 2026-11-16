import cv2
from mtcnn import MTCNN

# Load one of yesterday's extracted frames
image_path = 'frame_2.jpg'
img = cv2.imread(image_path)

# Create the face detector
detector = MTCNN()

# Detect faces in the image
faces = detector.detect_faces(img)

print(f"Number of faces found: {len(faces)}")

if len(faces) > 0:
    x, y, w, h = faces[0]['box']
    face_crop = img[y:y+h, x:x+w]
    face_crop = cv2.resize(face_crop, (224, 224))
    cv2.imwrite('cropped_face.jpg', face_crop)
    print("Face cropped and saved as cropped_face.jpg")
else:
    print("No face detected in this image. Try a different frame (frame_2.jpg, frame_2.jpg, etc.)")
