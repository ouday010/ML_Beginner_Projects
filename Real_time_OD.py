from ultralytics import YOLO
import cv2
import time
import numpy as np

# Load the pre-trained YOLOv8 model (nano version for speed, use 'yolov8m.pt' for higher accuracy)
model = YOLO("yolov8n.pt")  # Downloads automatically if not present

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, change to 1 or other index for external cameras
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set frame skip for performance (process every nth frame)
frame_skip = 2  # Process every 2nd frame to balance speed and accuracy
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_count += 1
    # Skip frames to improve performance
    if frame_count % frame_skip != 0:
        cv2.imshow("Real-Time Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Preprocess frame: Resize and enhance contrast
    frame = cv2.resize(frame, (min(frame.shape[1], 640), min(frame.shape[0], 480)))  # Smaller size for speed
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)  # Enhance contrast

    # Perform object detection
    start_time = time.time()
    results = model.predict(frame, conf=0.4, iou=0.5)  # Confidence threshold=0.4, IoU=0.5 for non-max suppression
    detection_time = time.time() - start_time

    # Draw bounding boxes and labels
    num_objects = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = model.names[class_id]
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            num_objects += 1

    # Display detection stats and FPS
    cv2.putText(
        frame,
        f"Objects: {num_objects}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"FPS: {1.0 / detection_time:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Show the frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()