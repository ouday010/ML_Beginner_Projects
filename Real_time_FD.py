import cv2
import sys
from mtcnn import MTCNN
import numpy as np
import time

# Load Haar Cascade for body detection
body_cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
body_cascade = cv2.CascadeClassifier(body_cascade_path)

if body_cascade.empty():
    print("Error: Could not load body cascade classifier.")
    sys.exit(1)

# Initialize MTCNN for face detection
try:
    face_detector = MTCNN()  # Using default parameters for multi-face detection
except Exception as e:
    print(f"Error initializing MTCNN: {e}")
    sys.exit(1)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, change if using another camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

# Set frame skip for performance (process every nth frame)
frame_skip = 3  # Process every 3rd frame to improve speed
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_count += 1
    # Skip frames to reduce processing load
    if frame_count % frame_skip != 0:
        cv2.imshow("Human Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Preprocess frame: Resize and enhance contrast
    frame = cv2.resize(frame, (min(frame.shape[1], 640), min(frame.shape[0], 480)))  # Smaller size for speed
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)  # Enhance contrast
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces with MTCNN
    try:
        start_time = time.time()
        faces = face_detector.detect_faces(rgb_frame)
        face_detection_time = time.time() - start_time
    except Exception as e:
        print(f"Error during face detection: {e}")
        faces = []

    # Convert to grayscale for body detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bodies with Haar Cascade
    bodies = body_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(50, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles and text for faces
    num_faces_detected = 0
    for face in faces:
        confidence = face['confidence']
        if confidence > 0.7:  # Threshold for multi-face detection
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)  # Ensure positive coordinates
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Face ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            num_faces_detected += 1
        else:
            print(f"Skipped face with low confidence: {confidence:.2f}")

    # Draw rectangles for bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            frame,
            "Body",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    # Display detection results on frame
    cv2.putText(
        frame,
        f"Faces: {num_faces_detected}, Bodies: {len(bodies)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"FPS: {1.0 / face_detection_time:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Show the frame
    cv2.imshow("Human Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()