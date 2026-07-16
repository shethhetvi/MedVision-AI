import os
import tensorflow as tf

def load_trained_model(model_path: str):
    """
    Loads a trained Keras model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    print(f"Loading model from {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def save_model(model, save_path: str):
    """
    Saves a trained model to disk.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
