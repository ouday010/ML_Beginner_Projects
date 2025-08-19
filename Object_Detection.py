import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# Load the pre-trained EfficientNetB7 model with ImageNet weights
model = EfficientNetB7(weights="imagenet")

# Define the path to the image file
image_path = "image.jpg"  # Replace with your image path

try:
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at '{image_path}'")
    
    # Enhance contrast for better detection
    image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model's input size (600x600 for EfficientNetB7)
    image = cv2.resize(image, (600, 600))
    
    # Convert to array and add batch dimension
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Preprocess for EfficientNetB7
    image_array = preprocess_input(image_array)
    
    # Make predictions
    predictions = model.predict(image_array)
    
    # Decode top 5 predictions (increased from 3 for more context)
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    
    # Print predictions with confidence threshold
    print("\nPredictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        if score > 0.1:  # Filter low-confidence predictions
            print(f"{i+1}: {label} ({score:.2f})")
        else:
            print(f"Skipped low-confidence prediction: {label} ({score:.2f})")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Error during processing: {e}")