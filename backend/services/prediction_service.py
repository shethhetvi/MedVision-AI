import os
import tensorflow as tf
from models.model_loader import load_trained_model
from gradcam.core import make_gradcam_heatmap
from gradcam.overlay import save_and_display_gradcam

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

def predict_pneumonia(image_path: str):
    """
    Runs model inference and generates Grad-CAM.
    """
    global MODEL
    if MODEL is None:
        init_model()
        
    if MODEL == "DUMMY" or MODEL is None:
        return {
            "prediction": "Pneumonia",
            "confidence": 0.99,
            "model_version": "v1.0.0-dummy",
            "heatmap_path": None
        }
        
    img_array = preprocess_image(image_path)
    
    # Predict
    preds = MODEL.predict(img_array)
    confidence = float(preds[0][0])
    is_pneumonia = confidence > 0.5
    
    # Generate Grad-CAM
    # Assuming 'top_activation' is the last conv layer in EfficientNetB0
    last_conv_layer_name = "top_activation"
    heatmap = make_gradcam_heatmap(img_array, MODEL, last_conv_layer_name)
    
    heatmap_path = image_path.replace(".jpg", "_cam.jpg").replace(".png", "_cam.png")
    save_and_display_gradcam(image_path, heatmap, heatmap_path)
    
    return {
        "prediction": "Pneumonia" if is_pneumonia else "Normal",
        "confidence": confidence if is_pneumonia else 1.0 - confidence,
        "model_version": MODEL_VERSION,
        "heatmap_path": heatmap_path
    }

# Call this on backend startup
# init_model()
