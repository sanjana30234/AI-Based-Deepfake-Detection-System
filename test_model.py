from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('model/sih1683_mobilenet_finetuned_90_28.h5')
print("Model loaded successfully!")
model.summary()

img_path = 'cropped_face.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print(f"Prediction: FAKE (confidence: {prediction*100:.1f}%)")
else:
    print(f"Prediction: REAL (confidence: {(1-prediction)*100:.1f}%)")