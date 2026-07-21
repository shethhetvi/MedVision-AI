"""
Run full model evaluation + test the two user-provided X-ray images.
"""
import os
import sys
import requests
import numpy as np
import shutil

# ── 1. Test the two X-ray images via the API ────────────────────────────────
API_URL = "http://localhost:8000"
TEST_IMAGES = {
    "xray_normal.jpg":   "data/test_images/xray_normal.jpg",
    "xray_pneumonia.jpg":"data/test_images/xray_pneumonia.jpg",
}

print("=" * 60)
print("  MedVision AI — Live Inference Test")
print("=" * 60)

for label, path in TEST_IMAGES.items():
    if not os.path.exists(path):
        print(f"\n⚠️  Image not found: {path}  (skipping)")
        continue

    with open(path, "rb") as f:
        content = f.read()
    files = {"file": (label, content, "image/jpeg")}
    
    # Upload
    res = requests.post(f"{API_URL}/api/upload/", files=files)
    if res.status_code != 200:
        print(f"\n❌  Upload failed for {label}: {res.text}")
        continue
    file_id = res.json()["file_id"]
    
    # Predict
    pred_res = requests.post(f"{API_URL}/api/predict/{file_id}")
    if pred_res.status_code != 200:
        print(f"\n❌  Prediction failed for {label}: {pred_res.text}")
        continue
    result = pred_res.json()["result"]
    latency = pred_res.json()["latency_ms"]
    
    print(f"\n📷  Image:       {label}")
    print(f"    Prediction:  {result['prediction']}")
    print(f"    Confidence:  {result['confidence'] * 100:.2f}%")
    print(f"    Model:       {result['model_version']}")
    print(f"    Latency:     {latency:.1f} ms")
    if result.get("heatmap_path"):
        print(f"    Heatmap:     ✅ {result['heatmap_path']}")
    else:
        print(f"    Heatmap:     ⚠️  Not generated")

# ── 2. Full evaluation on test set ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  MedVision AI — Test Set Evaluation")
print("=" * 60)

sys.path.insert(0, ".")
import tensorflow as tf
from backend.models.loader import load_trained_model
from backend.training.dataset import get_dataloaders
from backend.evaluation.metrics import evaluate_model

MODEL_PATH = "models/saved/pneumonia_model_best.h5"
TEST_DIR   = "datasets/pneumonia/chest_xray/test"

if not os.path.exists(MODEL_PATH):
    print(f"❌  Model not found at {MODEL_PATH}")
    sys.exit(1)

if not os.path.exists(TEST_DIR):
    print(f"❌  Test dataset not found at {TEST_DIR}")
    sys.exit(1)

print(f"\nLoading model: {MODEL_PATH}")
model = load_trained_model(MODEL_PATH)

print(f"Loading test dataset: {TEST_DIR}")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)
AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

print("\nRunning evaluation on test set...\n")
metrics, (fpr, tpr, roc_auc) = evaluate_model(model, test_ds)

print("-" * 40)
print(f"  Accuracy:  {metrics['accuracy']  * 100:.2f}%")
print(f"  Precision: {metrics['precision'] * 100:.2f}%")
print(f"  Recall:    {metrics['recall']    * 100:.2f}%")
print(f"  F1-Score:  {metrics['f1']        * 100:.2f}%")
print(f"  ROC-AUC:   {metrics['roc_auc']   :.4f}")
print("-" * 40)
print("\n✅  Evaluation complete!")
