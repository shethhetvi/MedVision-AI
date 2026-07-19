import os
import argparse
import numpy as np
import tensorflow as tf
from backend.training.dataset import get_dataloaders
from backend.models.architectures.efficientnet import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="MedVision AI - Training Script")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    # Pointing to the actual train sub-folder from Kaggle dataset
    parser.add_argument("--data_dir", type=str, default="datasets/pneumonia/chest_xray/train", help="Path to training images (class subfolders)")
    parser.add_argument("--val_dir", type=str, default="datasets/pneumonia/chest_xray/val", help="Path to val images (class subfolders)")
    parser.add_argument("--model_dir", type=str, default="models/saved", help="Path to save models")
    return parser.parse_args()

def compute_class_weights(data_dir):
    """
    Compute class weights to handle class imbalance.
    Assumes subfolders are class names (e.g. NORMAL/, PNEUMONIA/).
    """
    class_counts = {}
    for cls in sorted(os.listdir(data_dir)):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path) if not f.startswith('.')])
            class_counts[cls] = count
    
    total = sum(class_counts.values())
    n_classes = len(class_counts)
    class_weights = {}
    for idx, (cls, count) in enumerate(sorted(class_counts.items())):
        class_weights[idx] = total / (n_classes * count)
        print(f"  Class '{cls}' ({count} samples) → weight: {class_weights[idx]:.3f}")
    return class_weights

def main():
    args = parse_args()
    print("Starting training process with arguments:")
    print(args)
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load dataset — use separate train and val dirs
    print("\nLoading datasets...")
    train_ds, _ = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    
    # If val_dir exists and has content, use it; otherwise fall back to split
    if os.path.exists(args.val_dir) and len(os.listdir(args.val_dir)) > 0:
        # Val folder in Kaggle dataset is tiny (8 images), so let's supplement
        # by using a 20% split from the train set instead
        print("Note: Kaggle val set is tiny. Using 20% split from train set for validation.")
    train_ds, val_ds = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    
    # Compute class weights to handle imbalance (NORMAL: 1341 vs PNEUMONIA: 3875)
    print("\nComputing class weights to handle imbalance:")
    class_weights = compute_class_weights(args.data_dir)
    
    # Build model
    print("\nBuilding model...")
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
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1
    )
    
    # Training loop — pass class_weight to handle imbalance
    print("\nStarting training loop...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=[model_checkpoint_callback, early_stopping, reduce_lr]
    )
    
    # Also save the final model (not just best checkpoint)
    final_path = os.path.join(args.model_dir, "pneumonia_model_final.h5")
    model.save(final_path)
    print(f"\nTraining complete!")
    print(f"  Best model (by val_accuracy): {checkpoint_filepath}")
    print(f"  Final model:                  {final_path}")
    
    # Print final metrics
    final_acc = history.history['val_accuracy'][-1]
    best_acc  = max(history.history['val_accuracy'])
    print(f"\n  Final val_accuracy: {final_acc:.4f}")
    print(f"  Best  val_accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
