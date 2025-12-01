# Age Detection using AI/ML

A Python-based age detection system using deep learning and computer vision. This project uses OpenCV's DNN module with pre-trained Caffe models to detect faces and estimate age groups from images.

## Features

- **Face Detection**: Automatically detects faces in images using a pre-trained deep learning model
- **Age Classification**: Classifies detected faces into 8 age groups
- **Easy to Use**: Simple API for integration into other projects
- **Visualization**: Draw detection results on images with bounding boxes and age labels

## Age Groups

The model classifies ages into the following groups:
- (0-2): Infant
- (4-6): Child  
- (8-12): Pre-teen
- (15-20): Teen
- (25-32): Young Adult
- (38-43): Adult
- (48-53): Middle Age
- (60-100): Senior

## Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NimaB990/Age-detection.git
cd Age-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained models:
```bash
python download_models.py
```

## Usage

### Command Line

Detect ages in an image:
```bash
python age_detector.py path/to/image.jpg
```

Save result with annotations:
```bash
python age_detector.py path/to/image.jpg output.jpg
```

### Python API

```python
from age_detector import AgeDetector

# Create detector
detector = AgeDetector(models_path="models")

# Load models
detector.load_models()

# Detect ages from file
results = detector.detect_age_from_file("image.jpg")

# Process results
for result in results:
    print(f"Age: {result.age_group}, Confidence: {result.confidence:.2%}")
    print(f"Face location: {result.face_bbox}")
```

### Simple Function

```python
from age_detector import detect_age_simple

# One-liner age detection
ages = detect_age_simple("image.jpg")
for age in ages:
    print(f"Detected age group: {age['age_group']}")
```

## Project Structure

```
Age-detection/
├── age_detector.py      # Main age detection module
├── download_models.py   # Script to download pre-trained models
├── requirements.txt     # Python dependencies
├── models/              # Directory for model files (created by download script)
│   ├── opencv_face_detector.pbtxt
│   ├── opencv_face_detector_uint8.pb
│   ├── age_deploy.prototxt
│   └── age_net.caffemodel
└── README.md
```

## How It Works

The age detection system works in two stages:

1. **Face Detection**: Uses OpenCV's DNN module with a pre-trained TensorFlow model to detect faces in the input image. The model outputs bounding boxes for all detected faces.

2. **Age Classification**: Each detected face is extracted, preprocessed, and passed through a pre-trained Caffe neural network that classifies the face into one of 8 age groups.

### Models Used

- **Face Detection**: OpenCV's face detector (TensorFlow-based)
- **Age Classification**: CNN trained on the Adience benchmark (Caffe-based)

## API Reference

### AgeDetector Class

#### `__init__(models_path="models")`
Initialize the age detector with path to model files.

#### `load_models() -> bool`
Load the face detection and age classification models. Returns True if successful.

#### `detect_age(image, face_confidence=0.7, padding=20) -> List[AgeDetectionResult]`
Detect ages of all faces in an image (numpy array in BGR format).

#### `detect_age_from_file(image_path, ...) -> List[AgeDetectionResult]`
Detect ages from an image file path.

#### `draw_results(image, results, ...) -> np.ndarray`
Draw detection results on an image with bounding boxes and labels.

### AgeDetectionResult

A dataclass containing:
- `face_bbox`: Tuple (x, y, width, height) of face bounding box
- `age_group`: Predicted age group string
- `confidence`: Confidence score (0.0 to 1.0)

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenCV team for the DNN module and face detector
- Gil Levi and Tal Hassner for the age classification model
- [Adience Benchmark](https://talhassner.github.io/home/projects/Adience/Adience-data.html) for the training dataset