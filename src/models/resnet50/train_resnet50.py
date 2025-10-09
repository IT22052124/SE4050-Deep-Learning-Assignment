"""
Training script for ResNet50 brain tumor classification.

Simplified version with single optimized ResNet50 model for easy comparison.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from src.models.resnet50.build_resnet50 import build_resnet50_optimized, unfreeze_resnet50_for_finetuning


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
    
    # === Load Data ===
    # Use the preprocessed data directly with STRONGER augmentation
    print(f"Loading data from preprocessed directories: {args.data_dir}")
    
    # Define image data generators with IMPROVED augmentation for better generalization
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        # Stronger data augmentation
        rotation_range=20,           # Increased from 5
        width_shift_range=0.2,       # Increased from 0.1
        height_shift_range=0.2,      # Increased from 0.1
        horizontal_flip=True,
        zoom_range=0.2,              # Increased from 0.1
        shear_range=0.15,            # Added shear
        brightness_range=[0.8, 1.2], # Added brightness variation
        fill_mode='nearest'
    )
    
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Load data from directory structure
    train_ds = train_datagen.flow_from_directory(
        os.path.join(args.data_dir, 'train'),
        target_size=(args.input_size, args.input_size),
        batch_size=args.batch_size,
        class_mode='binary',
        classes=['no', 'yes']
    )
    
    val_ds = valid_datagen.flow_from_directory(
        os.path.join(args.data_dir, 'val'),
        target_size=(args.input_size, args.input_size),
        batch_size=args.batch_size,
        class_mode='binary',
        classes=['no', 'yes']
    )
    
    test_ds = test_datagen.flow_from_directory(
        os.path.join(args.data_dir, 'test'),
        target_size=(args.input_size, args.input_size),
        batch_size=args.batch_size,
        class_mode='binary',
        classes=['no', 'yes']
    )
    
    class_names = ['no', 'yes']
    print(f"✅ Dataset loaded with classes: {class_names}")
    print(f"   Found {train_ds.samples} training samples")
    print(f"   Found {val_ds.samples} validation samples")
    print(f"   Found {test_ds.samples} test samples")
    
    # === Build Single Optimized ResNet50 Model ===
    input_shape = (args.input_size, args.input_size, 3)
    model = build_resnet50_optimized(input_shape)
    
    print(f"✅ Built optimized ResNet50 model")
    model.summary()
    
    # Save model summary
    with open(os.path.join(args.results_dir, "ResNet50_model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    
    # === Callbacks ===
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
            patience=7,  # Increased patience
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
    
    # === STAGE 1: Train with frozen base (fast initial training) ===
    print(f"\n{'='*60}")
    print(f"STAGE 1: Training with frozen ResNet50 base")
    print(f"{'='*60}")
    initial_epochs = min(10, args.epochs)  # Train for 10 epochs or less
    
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n✅ Stage 1 complete!")
    print(f"   Best validation accuracy: {max(history_stage1.history['val_accuracy']):.4f}")
    
    # === STAGE 2: Fine-tune with unfrozen top layers ===
    if args.epochs > initial_epochs:
        print(f"\n{'='*60}")
        print(f"STAGE 2: Fine-tuning with unfrozen top ResNet50 layers")
        print(f"{'='*60}")
        
        # Unfreeze top layers for fine-tuning
        model = unfreeze_resnet50_for_finetuning(model, learning_rate=0.00001)  # Lower LR for fine-tuning
        
        # Update callbacks to save fine-tuned model
        finetune_callbacks = [
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
                patience=2,
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        # Continue training with fine-tuning
        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epochs,
            epochs=args.epochs,
            callbacks=finetune_callbacks,
            verbose=1
        )
        
        print(f"\n✅ Stage 2 (fine-tuning) complete!")
        print(f"   Best validation accuracy: {max(history_stage2.history['val_accuracy']):.4f}")
        
        # Combine histories
        history = type('obj', (object,), {
            'history': {
                'accuracy': history_stage1.history['accuracy'] + history_stage2.history['accuracy'],
                'val_accuracy': history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy'],
                'loss': history_stage1.history['loss'] + history_stage2.history['loss'],
                'val_loss': history_stage1.history['val_loss'] + history_stage2.history['val_loss']
            }
        })()
    else:
        history = history_stage1
    
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
