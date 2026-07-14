import time
import random

def predict_pneumonia(image_path: str):
    time.sleep(1) # simulate latency
    is_pneumonia = random.choice([True, False])
    confidence = random.uniform(0.75, 0.99)
    return {
        "prediction": "Pneumonia" if is_pneumonia else "Normal",
        "confidence": confidence,
        "model_version": "v1.0.0-dummy"
    }
