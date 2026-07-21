import tensorflow as tf
# pyrefly: ignore [missing-import]
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

def unfreeze_and_compile(model, unfreeze_layers=40, learning_rate=1e-5):
    """
    Unfreezes the top `unfreeze_layers` of the base EfficientNetB0 architecture
    and re-compiles the model with a small learning rate and BinaryFocalCrossentropy.
    """
    try:
        base_model = model.get_layer("efficientnetb0")
    except ValueError:
        # Fallback if layer is named differently
        base_model = model.layers[1]
        
    base_model.trainable = True
    
    # Freeze all layers except the last `unfreeze_layers`
    if unfreeze_layers > 0:
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
            
    trainable_count = sum([1 for l in base_model.layers if l.trainable])
    print(f"🔓 Base model un-frozen: {trainable_count} layers set to trainable (last {unfreeze_layers} layers).")
    
    # Try BinaryFocalCrossentropy to focus learning on hard/imbalanced examples
    try:
        loss_fn = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0)
    except AttributeError:
        loss_fn = 'binary_crossentropy'
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    # Test model creation & unfreezing
    model = build_model()
    model.summary()
    unfreeze_and_compile(model, unfreeze_layers=40)

