import os
import pytest
import numpy as np
import tensorflow as tf
from backend.models.loader import load_trained_model
from backend.services.prediction_service import preprocess_image

MODEL_PATH = "models/saved/pneumonia_model_best.h5"

def test_model_loading():
    assert os.path.exists(MODEL_PATH), f"Model path {MODEL_PATH} does not exist."
    model = load_trained_model(MODEL_PATH)
    assert model is not None, "Model failed to load."
    assert model.input_shape == (None, 224, 224, 3)

def test_model_inference_shape():
    model = load_trained_model(MODEL_PATH)
    dummy_input = np.random.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)
    assert preds.shape == (1, 1)
    assert 0.0 <= float(preds[0][0]) <= 1.0
