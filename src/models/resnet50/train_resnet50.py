"""
Training script for ResNet50 brain tumor classification.

This version implements a strong two-stage fine-tuning pipeline with:
- Proper ResNet50 preprocessing (preprocess_input)
- Stage 1: Train classifier head with the base frozen
- Stage 2: Unfreeze top layers and fine-tune with a lower LR
- Class weighting for imbalance, label smoothing, and AUC metric
- Robust callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from src.models.resnet50.build_resnet50 import (
    build_resnet50_frozen_head,
    fine_tune_resnet50,
)
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
    parser.add_argument("--epochs_stage1", type=int, default=8, help="Epochs for frozen head training")
    parser.add_argument("--epochs_stage2", type=int, default=20, help="Epochs for fine-tuning")
    parser.add_argument("--learning_rate_head", type=float, default=1e-4, help="Learning rate for head training")
    parser.add_argument("--learning_rate_ft", type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--trainable_layers", type=int, default=80, help="Number of top base layers to unfreeze in stage 2")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size (224 recommended for ResNet50)")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision if supported")
    
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
    
    # Optional mixed precision
    if args.mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("✅ Mixed precision enabled")
        except Exception as e:
            print(f"⚠️  Could not enable mixed precision: {e}")

    # === Load Data (pre-split directories) ===
    augment = get_augmentation_pipeline()
    train_ds, val_ds, test_ds, class_names, class_counts = create_datasets_from_splits(
        args.data_dir, 
        args.batch_size, 
        img_size=(args.input_size, args.input_size), 
        augment_fn=augment,
        allowed_classes=['no', 'yes']  # Only use binary classification classes
    )
    
    print(f"✅ Dataset loaded with classes: {class_names}")
    
    # Apply proper ResNet preprocessing; our pipeline scales to [0,1], so undo to [0,255] first
    def apply_resnet_preprocess(x, y):
        x = preprocess_input(x * 255.0)
        return x, y
    train_ds = train_ds.map(apply_resnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(apply_resnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(apply_resnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # === Build model and Stage 1 training (frozen base) ===
    input_shape = (args.input_size, args.input_size, 3)
    model = build_resnet50_frozen_head(
        input_shape=input_shape,
        dense_units=(512, 256),
        dropout=(0.5, 0.3),
        l2_reg=1e-4,
        learning_rate=args.learning_rate_head,
        label_smoothing=0.05,
    )
    
    print(f"✅ Built ResNet50 model (stage 1: frozen base)")
    model.summary()
    
    # Save model summary
    with open(os.path.join(args.results_dir, "ResNet50_model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    
    # === Callbacks (Same as VGG16) ===
    callbacks_stage1 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.results_dir, "best_stage1.h5"),
            monitor="val_auc",
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=4,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # === Train (Single-stage like VGG16) ===
    # Compute class weights
    total = sum(class_counts)
    class_weight = {i: total / (len(class_counts) * cnt) for i, cnt in enumerate(class_counts) if cnt > 0}
    print(f"Class weights: {class_weight}")

    print(f"🏋️ Stage 1: Training classifier head (frozen base)...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_stage1,
        callbacks=callbacks_stage1,
        verbose=1,
        class_weight=class_weight,
    )

    # === Stage 2: Fine-tune top layers ===
    model = fine_tune_resnet50(
        model,
        trainable_layers=args.trainable_layers,
        learning_rate=args.learning_rate_ft,
        label_smoothing=0.0,
        freeze_batchnorm=True,
    )

    callbacks_stage2 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.results_dir, "best_model.h5"),
            monitor="val_auc",
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=6,
            mode='max',
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

    print(f"🏋️ Stage 2: Fine-tuning top {args.trainable_layers} layers of ResNet50 base...")
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_stage2,
        callbacks=callbacks_stage2,
        verbose=1,
        class_weight=class_weight,
    )
    
    # === Save Training Plot ===
    # === Save Training Plots ===
    def plot_history(hist, title_prefix, out_name):
        plt.figure(figsize=(14, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(hist.history.get('accuracy', []), label='Train', linewidth=2)
        plt.plot(hist.history.get('val_accuracy', []), label='Val', linewidth=2)
        plt.title(f'{title_prefix} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(hist.history.get('auc', []), label='Train', linewidth=2)
        plt.plot(hist.history.get('val_auc', []), label='Val', linewidth=2)
        plt.title(f'{title_prefix} - AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(hist.history.get('loss', []), label='Train', linewidth=2)
        plt.plot(hist.history.get('val_loss', []), label='Val', linewidth=2)
        plt.title(f'{title_prefix} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'{title_prefix} History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, out_name), dpi=300, bbox_inches='tight')
        plt.close()

    plot_history(history1, 'ResNet50 Stage 1', 'training_stage1.png')
    plot_history(history2, 'ResNet50 Stage 2', 'training_stage2.png')
    
    # === Save Metrics ===
    # Merge histories for quick summary
    def safe_max(d, key):
        return float(max(d.get(key, [0.0]) or [0.0]))
    def safe_min(d, key):
        return float(min(d.get(key, [float('inf')]) or [float('inf')]))

    final_metrics = {
        "model_name": "ResNet50_TwoStage",
        "stage1": {
            "best_val_acc": safe_max(history1.history, 'val_accuracy'),
            "best_val_auc": safe_max(history1.history, 'val_auc'),
            "min_val_loss": safe_min(history1.history, 'val_loss'),
            "epochs": len(history1.history.get('loss', [])),
        },
        "stage2": {
            "best_val_acc": safe_max(history2.history, 'val_accuracy'),
            "best_val_auc": safe_max(history2.history, 'val_auc'),
            "min_val_loss": safe_min(history2.history, 'val_loss'),
            "epochs": len(history2.history.get('loss', [])),
        },
        "config": {
            "batch_size": args.batch_size,
            "epochs_stage1": args.epochs_stage1,
            "epochs_stage2": args.epochs_stage2,
            "learning_rate_head": args.learning_rate_head,
            "learning_rate_ft": args.learning_rate_ft,
            "trainable_layers": args.trainable_layers,
            "input_size": args.input_size,
            "input_shape": input_shape
        }
    }
    
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)
    
    print(f"✅ Training complete!")
    print(f"   Best validation AUC: {max(final_metrics['stage1']['best_val_auc'], final_metrics['stage2']['best_val_auc']):.4f}")
    print(f"   Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
