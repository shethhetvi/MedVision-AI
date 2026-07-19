import requests
import time
from PIL import Image
import os

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    print("Wait for server to start...")
    # Wait for the server to be up
    for _ in range(10):
        try:
            r = requests.get(f"{BASE_URL}/")
            if r.status_code == 200:
                print("Server is up!")
                break
        except requests.ConnectionError:
            time.sleep(2)
    else:
        print("Server did not start in time.")
        return
        
    print("\n--- Testing Upload ---")
    # 1. Create a dummy test image
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    img_path = "test_patient.png"
    img.save(img_path)
    
    with open(img_path, "rb") as f:
        files = {"file": ("test_patient.png", f, "image/png")}
        r = requests.post(f"{BASE_URL}/api/upload/", files=files)
    
    print(f"Upload Status: {r.status_code}")
    assert r.status_code == 200
    upload_res = r.json()
    print("Upload Response:", upload_res)
    file_id = upload_res["file_id"]
    filename = upload_res["filename"]
    saved_filename = f"{file_id}_{filename}"
    
    print("\n--- Testing Prediction ---")
    r = requests.post(f"{BASE_URL}/api/predict/{file_id}")
    print(f"Predict Status: {r.status_code}")
    assert r.status_code == 200
    predict_res = r.json()
    print("Predict Response:", predict_res)
    
    print("\n--- Testing PDF Report ---")
    r = requests.get(f"{BASE_URL}/api/report/{file_id}")
    print(f"Report Status: {r.status_code}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    print("PDF Report generated and downloaded successfully.")
    
    print("\n--- Testing Static Files ---")
    r = requests.get(f"{BASE_URL}/static/uploads/{saved_filename}")
    print(f"Static File Status: {r.status_code}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    print("Static Image served successfully.")
    
    print("\nAll tests passed successfully!")
    
    # Cleanup
    if os.path.exists(img_path):
        os.remove(img_path)

if __name__ == "__main__":
    run_tests()
