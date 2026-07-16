import os
import tensorflow as tf
from models.model_loader import load_trained_model

# Global model variable to hold the loaded model in memory
MODEL = None
MODEL_VERSION = "v1.0.0"

def init_model(model_path="models/saved/pneumonia_model_best.h5"):
    """
    Initializes the global model once.
    """
    global MODEL
    if MODEL is None:
        try:
            MODEL = load_trained_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            # Fallback for dev if no model exists yet
            MODEL = "DUMMY"

def preprocess_image(image_path: str):
    """
    Prepares image for model inference.
    """
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

# Call this on backend startup (can be called from main.py)
# init_model()
