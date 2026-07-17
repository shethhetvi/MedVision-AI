import os
import argparse
import tensorflow as tf
from backend.training.dataset import get_dataloaders
from backend.models.architectures.efficientnet import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="MedVision AI - Training Script")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="../datasets/pneumonia", help="Path to dataset")
    parser.add_argument("--model_dir", type=str, default="../models/saved", help="Path to save models")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Starting training process with arguments:")
    print(args)
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load dataset
    print("Loading datasets...")
    train_ds, val_ds = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    
    # Build model
    print("Building model...")
    model = build_model()
    
    # Callbacks
    checkpoint_filepath = os.path.join(args.model_dir, "pneumonia_model_best.h5")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Training loop
    print("Starting training loop...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[model_checkpoint_callback, early_stopping]
    )
    
    print(f"Training complete. Best model saved to {checkpoint_filepath}")

if __name__ == "__main__":
    main()
