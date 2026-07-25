import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_upload_and_predict():
    test_img_path = "data/test_images/xray_normal.jpg"
    if not os.path.exists(test_img_path):
        pytest.skip("Test image xray_normal.jpg not present")
        
    with open(test_img_path, "rb") as f:
        response = client.post("/api/upload/", files={"file": ("xray_normal.jpg", f, "image/jpeg")})
    
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    file_id = data["file_id"]

    # Test Prediction endpoint
    pred_res = client.post(f"/api/predict/{file_id}")
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert "result" in pred_data
    assert pred_data["result"]["prediction"] in ["Normal", "Pneumonia"]

    # Test PDF Report endpoint
    report_res = client.get(f"/api/report/{file_id}")
    assert report_res.status_code == 200
    assert report_res.headers["content-type"] == "application/pdf"
