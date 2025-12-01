"""
Tests for Age Detection module.
"""

import os
import sys
import unittest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from age_detector import AgeDetector, AgeDetectionResult


class TestAgeDetector(unittest.TestCase):
    """Test cases for AgeDetector class."""
    
    def test_age_groups_defined(self):
        """Test that age groups are properly defined."""
        self.assertEqual(len(AgeDetector.AGE_GROUPS), 8)
        self.assertIn('(0-2)', AgeDetector.AGE_GROUPS)
        self.assertIn('(25-32)', AgeDetector.AGE_GROUPS)
        self.assertIn('(60-100)', AgeDetector.AGE_GROUPS)
    
    def test_detector_initialization(self):
        """Test detector initialization with custom path."""
        detector = AgeDetector(models_path="custom/path")
        self.assertEqual(detector.models_path, "custom/path")
        self.assertFalse(detector._models_loaded)
    
    def test_model_path_generation(self):
        """Test that model paths are generated correctly."""
        detector = AgeDetector(models_path="/test/models")
        path = detector._get_model_path("test.pb")
        self.assertEqual(path, "/test/models/test.pb")
    
    def test_age_detection_result_dataclass(self):
        """Test AgeDetectionResult dataclass."""
        result = AgeDetectionResult(
            face_bbox=(10, 20, 100, 100),
            age_group="(25-32)",
            confidence=0.95
        )
        self.assertEqual(result.face_bbox, (10, 20, 100, 100))
        self.assertEqual(result.age_group, "(25-32)")
        self.assertEqual(result.confidence, 0.95)
    
    def test_models_not_loaded_initially(self):
        """Test that models are not loaded on initialization."""
        detector = AgeDetector()
        self.assertFalse(detector._models_loaded)
        self.assertIsNone(detector._face_net)
        self.assertIsNone(detector._age_net)
    
    def test_load_models_returns_false_when_files_missing(self):
        """Test that load_models returns False when model files don't exist."""
        detector = AgeDetector(models_path="/nonexistent/path")
        result = detector.load_models()
        self.assertFalse(result)
        self.assertFalse(detector._models_loaded)
    
    def test_detect_age_raises_without_models(self):
        """Test that detect_age raises error when models not loaded."""
        detector = AgeDetector(models_path="/nonexistent/path")
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with self.assertRaises(RuntimeError):
            detector.detect_age(dummy_image)
    
    def test_detect_age_from_file_raises_on_invalid_path(self):
        """Test that detect_age_from_file raises error for invalid path."""
        detector = AgeDetector()
        detector._models_loaded = True  # Bypass model loading for this test
        
        with self.assertRaises(ValueError):
            detector.detect_age_from_file("/nonexistent/image.jpg")


class TestDrawResults(unittest.TestCase):
    """Test cases for drawing results."""
    
    def test_draw_results_preserves_image_size(self):
        """Test that draw_results returns image of same size."""
        detector = AgeDetector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = []
        
        output = detector.draw_results(image, results)
        
        self.assertEqual(output.shape, image.shape)
    
    def test_draw_results_with_detections(self):
        """Test draw_results with detection results."""
        detector = AgeDetector()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [
            AgeDetectionResult(
                face_bbox=(100, 100, 50, 50),
                age_group="(25-32)",
                confidence=0.9
            )
        ]
        
        output = detector.draw_results(image, results)
        
        # Output should be different from input (has annotations)
        self.assertEqual(output.shape, image.shape)
        # Check that something was drawn (not all zeros)
        self.assertGreater(np.sum(output), 0)


if __name__ == '__main__':
    unittest.main()
