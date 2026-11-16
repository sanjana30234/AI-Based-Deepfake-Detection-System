# AI-Based Deepfake Detection System

An AI-powered application designed to detect potential deepfake manipulation in uploaded images and videos using deep learning.

## Features

- Image-based deepfake detection
- Video-based deepfake detection
- Face detection and frame extraction
- Frame-level prediction and video-level result aggregation
- Audio analysis using the AASIST ONNX model
- Suspicious-frame identification and timeline reporting
- Browser-based interface for uploading media and viewing analysis results

## Technologies Used

- Python
- Flask
- TensorFlow / MobileNetV2
- MTCNN
- OpenCV
- ONNX Runtime
- HTML and CSS

## Project Structure

```text
deepfake_detector/
├── app.py
├── audio_analysis.py
├── audio_utils.py
├── aasist.onnx
├── model/
│   └── sih1683_mobilenet_finetuned_90_28.h5
├── templates/
│   ├── index.html
│   └── results.html
├── static/
└── .gitignore
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Based-Deepfake-Detection-System.git
cd AI-Based-Deepfake-Detection-System
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python app.py
```

5. Open your browser:

```text
http://127.0.0.1:5000
```

## Models

- MobileNetV2 fine-tuned model for image and video frame classification.
- AASIST ONNX model for audio analysis.

## Note

This project is intended for deepfake detection research and educational purposes. Predictions are model-based estimates and should not be treated as definitive proof that media is real or fake.

