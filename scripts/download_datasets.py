import os
import sys

def main():
    # Read Kaggle credentials from environment variables
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    
    if not username or not key:
        print("Error: KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set.")
        sys.exit(1)
        
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Error: The 'kaggle' python package is not installed. Run 'pip install kaggle'")
        sys.exit(1)

    print("Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()

    datasets = {
        "Pneumonia (Chest X-Ray)": {
            "id": "paultimothymooney/chest-xray-pneumonia",
            "path": "datasets/pneumonia"
        },
        "Brain Tumor (MRI)": {
            "id": "navoneel/brain-mri-images-for-brain-tumor-detection",
            "path": "datasets/brain_tumor"
        },
        "Skin Cancer (HAM10000)": {
            "id": "kmader/skin-cancer-mnist-ham10000",
            "path": "datasets/skin_cancer"
        }
    }

    print("\nStarting dataset downloads...")
    for name, config in datasets.items():
        dataset_id = config["id"]
        download_path = config["path"]
        
        # Check if already downloaded (simple heuristic: folder has contents)
        if os.path.exists(download_path) and len(os.listdir(download_path)) > 0:
            print(f"[{name}] Already exists in {download_path}. Skipping...")
            continue
            
        print(f"[{name}] Downloading {dataset_id} into {download_path}...")
        os.makedirs(download_path, exist_ok=True)
        try:
            api.dataset_download_cli(dataset_id, unzip=True, path=download_path)
            print(f"[{name}] Download and unzip complete!")
        except Exception as e:
            print(f"[{name}] Error downloading dataset: {e}")

    print("\nAll dataset processing completed!")

if __name__ == "__main__":
    main()
