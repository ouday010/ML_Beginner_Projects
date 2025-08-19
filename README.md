# ML_Beginner_Projects
Python scripts for image classification (MobileNetV2, EfficientNetB7), face/body detection (MTCNN, Haar Cascade), and real-time object detection (YOLOv8). Ideal for ML beginners to detect objects/faces in images or webcam feeds. Requires TensorFlow, OpenCV, MTCNN, Ultralytics.


# Computer Vision Detection Suite

Five Python scripts for image classification, face/body detection, and real-time object detection using pre-trained deep learning models. Ideal for beginners and developers exploring ML and computer vision without needing custom datasets.

**Short Description (233 chars)**: Five Python scripts for image classification (MobileNetV2, EfficientNetB7), face/body detection (MTCNN, Haar Cascade), and real-time object detection (YOLOv8). Ideal for ML beginners to detect objects/faces in images or webcam feeds.

## Scripts Overview

1. **image_classifier.py**: Classifies the main object in an image using MobileNetV2 (ImageNet, ~70-75% accuracy).
2. **face_detection.py**: Detects faces (MTCNN, ~95% accuracy) and bodies (Haar Cascade, ~60-70% accuracy) in an image.
3. **object_detection.py**: Classifies the main object in an image using EfficientNetB7 (ImageNet, ~84% accuracy).
4. **real_time_fd.py**: Real-time face and body detection via webcam using MTCNN and Haar Cascade.
5. **real_time_od.py**: Real-time multi-object detection via webcam using YOLOv8 (COCO, ~50-60% mAP).

## Who Can Benefit
- **ML Beginners**: Learn pre-trained model usage for classification and detection.
- **Students/Developers**: Build projects or prototypes for image/video analysis.
- **Hobbyists**: Detect objects or people in photos or live feeds.
- **Educators**: Teach computer vision concepts with practical examples.

## Requirements
- **Libraries**: Install via pip:
  ```bash
  pip install tensorflow opencv-python mtcnn ultralytics numpy
