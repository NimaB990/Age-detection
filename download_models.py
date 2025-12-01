#!/usr/bin/env python3
"""
Download pre-trained models for age detection.

This script downloads the required model files from public sources:
1. OpenCV Face Detection model (TensorFlow)
2. Age Classification model (Caffe)
"""

import os
import urllib.request
import sys

# Model URLs
MODELS = {
    # Face detection model (OpenCV's DNN)
    "opencv_face_detector.pbtxt": 
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb": 
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb",
    
    # Age classification model (Caffe)
    "age_deploy.prototxt": 
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt",
    "age_net.caffemodel": 
        "https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=1",
}

def download_file(url: str, filepath: str) -> bool:
    """
    Download a file from URL to local path.
    
    Args:
        url: Source URL
        filepath: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading: {os.path.basename(filepath)}")
        print(f"  From: {url}")
        
        # Download with progress
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                sys.stdout.write(f"\r  Progress: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, filepath, reporthook)
        print("\n  Done!")
        return True
        
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def main():
    """Download all required model files."""
    # Create models directory
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("Age Detection - Model Downloader")
    print("=" * 40)
    print(f"\nDownloading models to: {models_dir}\n")
    
    success_count = 0
    total_count = len(MODELS)
    
    for filename, url in MODELS.items():
        filepath = os.path.join(models_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"Skipping {filename} (already exists)")
            success_count += 1
            continue
        
        if download_file(url, filepath):
            success_count += 1
        else:
            print(f"Failed to download: {filename}")
    
    print(f"\n{'=' * 40}")
    print(f"Downloaded {success_count}/{total_count} files")
    
    if success_count == total_count:
        print("\nAll models downloaded successfully!")
        print("You can now run: python age_detector.py <image_path>")
        return 0
    else:
        print("\nSome downloads failed. Please try again or download manually.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
