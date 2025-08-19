import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Constants
IMG_SIZE = (224, 224)  # MobileNetV2 input size

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet', input_shape=(*IMG_SIZE, 3))

# Function to classify an image
def classify_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Preprocess for MobileNetV2

    # Make prediction
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions

    # Print results
    print("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score * 100:.2f}%)")

    return decoded_predictions

# Example usage
if __name__ == "__main__":
    image_path = 'image.jpg'  # Replace with your image path
    try:
        predictions = classify_image(image_path)
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found. Please provide a valid image path.")