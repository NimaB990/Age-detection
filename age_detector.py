"""
Age Detection using AI/ML

This module provides age detection functionality using deep learning models.
It uses OpenCV's DNN module with pre-trained Caffe models for face detection
and age classification.

The age detection works in two steps:
1. Detect faces in the image using a pre-trained face detection model
2. Classify the age of each detected face using an age classification model

Age Groups:
- (0-2): Infant
- (4-6): Child
- (8-12): Pre-teen
- (15-20): Teen
- (25-32): Young Adult
- (38-43): Adult
- (48-53): Middle Age
- (60-100): Senior
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AgeDetectionResult:
    """Represents the result of age detection for a single face."""
    face_bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    age_group: str
    confidence: float


class AgeDetector:
    """
    Age detector using deep learning models.
    
    This class uses pre-trained Caffe models for face detection and age estimation.
    """
    
    # Age group labels for the classification model
    AGE_GROUPS = [
        '(0-2)', '(4-6)', '(8-12)', '(15-20)', 
        '(25-32)', '(38-43)', '(48-53)', '(60-100)'
    ]
    
    # Model file names
    FACE_PROTO = "opencv_face_detector.pbtxt"
    FACE_MODEL = "opencv_face_detector_uint8.pb"
    AGE_PROTO = "age_deploy.prototxt"
    AGE_MODEL = "age_net.caffemodel"
    
    # Mean values for image preprocessing (required by the model)
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    
    def __init__(self, models_path: str = "models"):
        """
        Initialize the age detector.
        
        Args:
            models_path: Path to directory containing model files
        """
        self.models_path = models_path
        self._face_net = None
        self._age_net = None
        self._models_loaded = False
        
    def _get_model_path(self, filename: str) -> str:
        """Get full path to a model file."""
        return os.path.join(self.models_path, filename)
    
    def load_models(self) -> bool:
        """
        Load the face detection and age classification models.
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            face_proto_path = self._get_model_path(self.FACE_PROTO)
            face_model_path = self._get_model_path(self.FACE_MODEL)
            age_proto_path = self._get_model_path(self.AGE_PROTO)
            age_model_path = self._get_model_path(self.AGE_MODEL)
            
            # Check if all model files exist
            for path in [face_proto_path, face_model_path, age_proto_path, age_model_path]:
                if not os.path.exists(path):
                    print(f"Model file not found: {path}")
                    return False
            
            # Load face detection model
            self._face_net = cv2.dnn.readNet(face_model_path, face_proto_path)
            
            # Load age classification model
            self._age_net = cv2.dnn.readNet(age_model_path, age_proto_path)
            
            self._models_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def _detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.7) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Minimum confidence for face detection
            
        Returns:
            List of bounding boxes (x, y, width, height) for detected faces
        """
        if self._face_net is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        height, width = image.shape[:2]
        
        # Create blob from image for face detection
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), 
            [104, 117, 123], True, False
        )
        
        self._face_net.setInput(blob)
        detections = self._face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2 - x1, y2 - y1))
        
        return faces
    
    def _predict_age(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Predict age group from a face image.
        
        Args:
            face_image: Cropped face image as numpy array
            
        Returns:
            Tuple of (age_group, confidence)
        """
        if self._age_net is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Create blob from face image for age prediction
        blob = cv2.dnn.blobFromImage(
            face_image, 1.0, (227, 227), 
            self.MODEL_MEAN_VALUES, swapRB=False
        )
        
        self._age_net.setInput(blob)
        predictions = self._age_net.forward()
        
        # Get the age group with highest probability
        age_index = predictions[0].argmax()
        confidence = float(predictions[0][age_index])
        age_group = self.AGE_GROUPS[age_index]
        
        return age_group, confidence
    
    def detect_age(
        self, 
        image: np.ndarray, 
        face_confidence: float = 0.7,
        padding: int = 20
    ) -> List[AgeDetectionResult]:
        """
        Detect ages of all faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            face_confidence: Minimum confidence for face detection
            padding: Padding around detected face for age prediction
            
        Returns:
            List of AgeDetectionResult objects
        """
        if not self._models_loaded:
            if not self.load_models():
                raise RuntimeError("Failed to load models")
        
        results = []
        height, width = image.shape[:2]
        
        # Detect all faces in the image
        faces = self._detect_faces(image, face_confidence)
        
        for (x, y, w, h) in faces:
            # Add padding to the face region
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + w + padding)
            y2 = min(height, y + h + padding)
            
            # Extract face region
            face_image = image[y1:y2, x1:x2]
            
            if face_image.size == 0:
                continue
            
            # Predict age
            age_group, confidence = self._predict_age(face_image)
            
            results.append(AgeDetectionResult(
                face_bbox=(x, y, w, h),
                age_group=age_group,
                confidence=confidence
            ))
        
        return results
    
    def detect_age_from_file(
        self, 
        image_path: str,
        face_confidence: float = 0.7,
        padding: int = 20
    ) -> List[AgeDetectionResult]:
        """
        Detect ages from an image file.
        
        Args:
            image_path: Path to the image file
            face_confidence: Minimum confidence for face detection
            padding: Padding around detected face for age prediction
            
        Returns:
            List of AgeDetectionResult objects
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        return self.detect_age(image, face_confidence, padding)
    
    def draw_results(
        self, 
        image: np.ndarray, 
        results: List[AgeDetectionResult],
        box_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Draw detection results on an image.
        
        Args:
            image: Input image
            results: List of AgeDetectionResult objects
            box_color: Color for bounding box (BGR)
            text_color: Color for text (BGR)
            
        Returns:
            Image with drawn results
        """
        output = image.copy()
        
        for result in results:
            x, y, w, h = result.face_bbox
            
            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), box_color, 2)
            
            # Prepare label
            label = f"Age: {result.age_group}"
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                output, 
                (x, y - label_h - 10), 
                (x + label_w + 10, y),
                box_color, 
                cv2.FILLED
            )
            
            # Draw label text
            cv2.putText(
                output, label, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1
            )
        
        return output


def detect_age_simple(
    image_path: str, 
    models_path: str = "models"
) -> List[dict]:
    """
    Simple function to detect ages in an image.
    
    Args:
        image_path: Path to the image file
        models_path: Path to model files directory
        
    Returns:
        List of dictionaries with 'age_group', 'confidence', and 'bbox' keys
    """
    detector = AgeDetector(models_path)
    results = detector.detect_age_from_file(image_path)
    
    return [
        {
            'age_group': r.age_group,
            'confidence': r.confidence,
            'bbox': r.face_bbox
        }
        for r in results
    ]


if __name__ == "__main__":
    import sys
    
    print("Age Detection using AI/ML")
    print("=" * 40)
    
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("\nUsage: python age_detector.py <image_path>")
        print("\nExample: python age_detector.py sample_images/face.jpg")
        print("\nNote: Make sure to download the required model files first.")
        print("See README.md for instructions on downloading models.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Create detector and detect ages
    detector = AgeDetector()
    
    print(f"\nProcessing image: {image_path}")
    
    try:
        results = detector.detect_age_from_file(image_path)
        
        if not results:
            print("No faces detected in the image.")
        else:
            print(f"\nDetected {len(results)} face(s):\n")
            for i, result in enumerate(results, 1):
                print(f"Face {i}:")
                print(f"  Age Group: {result.age_group}")
                print(f"  Confidence: {result.confidence:.2%}")
                print(f"  Location: x={result.face_bbox[0]}, y={result.face_bbox[1]}")
                print()
        
        # Optionally save result image
        if len(sys.argv) > 2:
            output_path = sys.argv[2]
            image = cv2.imread(image_path)
            result_image = detector.draw_results(image, results)
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
            
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nMake sure model files are downloaded to the 'models' directory.")
        sys.exit(1)
