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
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Standard load failed ({e}), attempting config-cleaned load...")
        if model_path.endswith(".h5"):
            try:
                import h5py, json
                with h5py.File(model_path, 'r+') as f:
                    if 'model_config' in f.attrs:
                        config_str = f.attrs['model_config']
                        if isinstance(config_str, bytes): config_str = config_str.decode('utf-8')
                        config = json.loads(config_str)
                        def clean_config(obj):
                            if isinstance(obj, dict):
                                for key in ['renorm', 'renorm_clipping', 'renorm_momentum', 'quantization_config']:
                                    obj.pop(key, None)
                                for k, v in list(obj.items()): clean_config(v)
                            elif isinstance(obj, list):
                                for item in obj: clean_config(item)
                        clean_config(config)
                        f.attrs['model_config'] = json.dumps(config).encode('utf-8')
                return tf.keras.models.load_model(model_path, compile=False)
            except Exception as e2:
                print(f"Config-cleaned load also failed: {e2}")
        return None

def save_model(model, save_path: str):
    """
    Saves a trained model to disk.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")
