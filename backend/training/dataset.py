import tensorflow as tf
from tensorflow.keras import layers

def get_augmentation_pipeline():
    """
    Data augmentation layers for medical X-ray imaging.
    Targeted to expand feature space (rotation, zoom, contrast, translation).
    """
    return tf.keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
    ], name="data_augmentation")

def get_dataloaders(data_dir: str, batch_size: int = 32, img_size: tuple = (224, 224), augment: bool = True):
    """
    Creates tf.data.Dataset objects for training and validation.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    if augment:
        aug_pipeline = get_augmentation_pipeline()
        train_ds = train_ds.map(
            lambda x, y: (aug_pipeline(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )
        
    # Configure dataset for performance
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds

