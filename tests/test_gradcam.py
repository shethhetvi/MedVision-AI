import os
import pytest
from backend.services.prediction_service import predict_pneumonia

def test_gradcam_generation():
    test_img = "data/test_images/xray_pneumonia.jpg"
    if not os.path.exists(test_img):
        pytest.skip("Test image xray_pneumonia.jpg not present")
        
    result = predict_pneumonia(test_img)
    assert result["prediction"] in ["Pneumonia", "Normal"]
    assert "confidence" in result
    assert result["heatmap_path"] is not None
    assert os.path.exists(result["heatmap_path"])
