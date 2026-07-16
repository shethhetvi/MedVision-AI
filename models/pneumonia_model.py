import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Builds a transfer learning model using EfficientNetB0.
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom classification head
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    if num_classes == 2:
        # Binary classification
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        # Multi-class classification
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Test model creation
    model = build_model()
    model.summary()
