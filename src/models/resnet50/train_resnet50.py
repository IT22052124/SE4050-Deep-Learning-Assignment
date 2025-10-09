"""
Training script for ResNet50 brain tumor classification.

Optimized to match VGG16's successful training approach with tf.data pipeline.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from src.models.resnet50.build_resnet50 import build_resnet50_optimized
from src.common.dataset_utils import create_datasets_from_splits
from src.common.preprocessing import get_augmentation_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 for Brain Tumor classification")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.getenv("DATA_DIR", "/content/drive/MyDrive/BrainTumor/data/processed"),
        help="Folder containing train/val/test structure",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.getenv("RESULTS_DIR", "/content/drive/MyDrive/BrainTumor/Result/resnet50"),
        help="Directory to save results",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size (224 recommended for ResNet50)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    print(f"🚀 Starting ResNet50 training...")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Results directory: {args.results_dir}")
    print(f"   Input size: {args.input_size}x{args.input_size}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs: {args.epochs}")
    
    # === Load Data (Same as VGG16 but for pre-split directories) ===
    augment = get_augmentation_pipeline()
    train_ds, val_ds, test_ds, class_names, class_counts = create_datasets_from_splits(
        args.data_dir, 
        args.batch_size, 
        img_size=(args.input_size, args.input_size), 
        augment_fn=augment,
        allowed_classes=['no', 'yes']  # Only use binary classification classes
    )
    
    print(f"✅ Dataset loaded with classes: {class_names}")
    
    # === Build Single Optimized ResNet50 Model ===
    input_shape = (args.input_size, args.input_size, 3)
    model = build_resnet50_optimized(input_shape)
    
    print(f"✅ Built optimized ResNet50 model")
    model.summary()
    
    # Save model summary
    with open(os.path.join(args.results_dir, "ResNet50_model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    
    # === Callbacks (Same as VGG16) ===
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.results_dir, "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=5, 
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # === Train (Single-stage like VGG16) ===
    print(f"🏋️ Training ResNet50 model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # === Save Training Plot ===
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('ResNet50 - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('ResNet50 - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('ResNet50 Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "training_plot.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # === Save Metrics ===
    final_metrics = {
        "model_name": "ResNet50_Optimized",
        "train_accuracy": float(max(history.history['accuracy'])),
        "val_accuracy": float(max(history.history['val_accuracy'])),
        "train_loss": float(min(history.history['loss'])),
        "val_loss": float(min(history.history['val_loss'])),
        "epochs_trained": len(history.history['accuracy']),
        "config": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "input_size": args.input_size,
            "input_shape": input_shape
        }
    }
    
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)
    
    print(f"✅ Training complete!")
    print(f"   Best validation accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
