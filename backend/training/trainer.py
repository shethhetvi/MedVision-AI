import os
import argparse
import numpy as np
import tensorflow as tf
from backend.training.dataset import get_dataloaders
from backend.models.architectures.efficientnet import build_model, unfreeze_and_compile

def parse_args():
    parser = argparse.ArgumentParser(description="MedVision AI - Fine-Tuning Script")
    parser.add_argument("--stage1_epochs", type=int, default=4, help="Warmup head epochs")
    parser.add_argument("--stage2_epochs", type=int, default=6, help="Fine-tuning top layers epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--unfreeze_layers", type=int, default=40, help="Number of top layers to unfreeze")
    parser.add_argument("--data_dir", type=str, default="datasets/pneumonia/chest_xray/train", help="Path to dataset")
    parser.add_argument("--model_dir", type=str, default="models/saved", help="Path to save models")
    return parser.parse_args()

def compute_class_weights(data_dir):
    """
    Compute balanced class weights to handle class imbalance.
    NORMAL: 1341 (weight ~1.94) vs PNEUMONIA: 3875 (weight ~0.67)
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
    print("Starting 2-Stage MedVision AI Fine-Tuning Process...")
    print(args)
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Load dataset with augmentation for training
    print("\nLoading datasets with Data Augmentation...")
    train_ds, val_ds = get_dataloaders(args.data_dir, batch_size=args.batch_size, augment=True)
    
    # Compute balanced class weights
    print("\nComputing balanced class weights:")
    class_weights = compute_class_weights(args.data_dir)
    
    # ── STAGE 1: Warmup Classification Head ────────────────────────────────────
    print("\n" + "="*60)
    print("  STAGE 1: Training Classification Head (Frozen Base Model)")
    print("="*60)
    model = build_model()
    
    checkpoint_filepath = os.path.join(args.model_dir, "pneumonia_model_best.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.stage1_epochs,
        class_weight=class_weights,
        callbacks=[checkpoint_callback]
    )
    
    # ── STAGE 2: Fine-Tune Top 40 Layers of Base Model ────────────────────────
    print("\n" + "="*60)
    print(f"  STAGE 2: Fine-Tuning Top {args.unfreeze_layers} Layers (lr=1e-5, Focal Loss)")
    print("="*60)
    
    model = unfreeze_and_compile(model, unfreeze_layers=args.unfreeze_layers, learning_rate=1e-5)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1
    )
    
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.stage2_epochs,
        class_weight=class_weights,
        callbacks=[checkpoint_callback, early_stopping, reduce_lr]
    )
    
    # Save final fine-tuned model
    final_path = os.path.join(args.model_dir, "pneumonia_model_final.h5")
    model.save(final_path)
    model.save(checkpoint_filepath)
    
    print("\n" + "="*60)
    print("✅ Fine-Tuning Complete!")
    print(f"  Best Model Saved to: {checkpoint_filepath}")
    print(f"  Final Model Saved to: {final_path}")
    print("="*60)

if __name__ == "__main__":
    main()

