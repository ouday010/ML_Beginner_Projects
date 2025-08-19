import cv2
import sys
from mtcnn import MTCNN
import numpy as np

# Load Haar Cascade for body detection
body_cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
body_cascade = cv2.CascadeClassifier(body_cascade_path)

if body_cascade.empty():
    print("Error: Could not load body cascade classifier.")
    sys.exit(1)

# Initialize MTCNN for face detection
try:
    face_detector = MTCNN()  # Removed unsupported parameters
except Exception as e:
    print(f"Error initializing MTCNN: {e}")
    sys.exit(1)

# Load image
image_path = "image.jpg"  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image at '{image_path}'. Please check the file path.")
    sys.exit(1)

# Preprocess image: Resize and enhance contrast
image = cv2.resize(image, (min(image.shape[1], 1280), min(image.shape[0], 720)))  # Limit max size
image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)  # Enhance contrast
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces with MTCNN
try:
    faces = face_detector.detect_faces(rgb_image)
except Exception as e:
    print(f"Error during face detection: {e}")
    faces = []

# Convert to grayscale for body detection
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect bodies with Haar Cascade
bodies = body_cascade.detectMultiScale(
    gray_img,
    scaleFactor=1.05,  # Adjusted for sensitivity
    minNeighbors=6,    # Reduce false positives
    minSize=(50, 100),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# Draw rectangles and text for faces
num_faces_detected = 0
for face in faces:
    confidence = face['confidence']
    if confidence > 0.7:  # Lowered threshold for multi-face detection
        x, y, w, h = face['box']
        # Ensure positive coordinates
        x, y = max(0, x), max(0, y)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
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
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(
        image,
        "Body",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )

# Output detection results
num_bodies = len(bodies)
if num_faces_detected > 0 or num_bodies > 0:
    print(f"Human Detected: {num_faces_detected} face(s), {num_bodies} body(ies)")
else:
    print("No humans detected (object or empty scene)")

# Save and display the output image
output_path = "output_detected.jpg"
cv2.imwrite(output_path, image)
print(f"Output image saved as '{output_path}'")
cv2.imshow("Human Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()