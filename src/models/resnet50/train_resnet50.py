"""
Training script for ResNet50 Transfer Learning Model

This script trains a ResNet50-based transfer learning model for brain tumor classification.
It supports both binary and multi-class classification with two-stage training:
1. Feature extraction: Train only the custom head with frozen ResNet50 base
2. Fine-tuning: Unfreeze top layers and train end-to-end with lower learning rate

Expected dataset layout for --data_dir:
  For binary:
    data_dir/
      yes/
        img1.jpg ...
      no/
        img2.jpg ...
        
  For multi-class:
    data_dir/
      glioma/
        img1.jpg ...
      meningioma/
        img2.jpg ...
      pituitary/
        img3.jpg ...
      no_tumor/
        img4.jpg ...
"""

# src/models/resnet50/train_resnet50.py
import os
import json
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from src.models.resnet50.build_resnet50 import build_resnet50_model, unfreeze_base_model
from src.common.dataset_utils import create_datasets
from src.common.preprocessing import get_augmentation_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 for Brain Tumor classification")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.getenv("DATA_DIR", "/content/drive/MyDrive/BrainTumor"),
        help="Folder containing class subfolders",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.getenv("RESULTS_DIR", "/content/drive/MyDrive/brain_tumor_project/results/resnet50"),
        help="Folder to write checkpoints and plots",
    )
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", 16)))
    parser.add_argument("--epochs_stage1", type=int, default=20, 
                       help="Epochs for stage 1 (feature extraction)")
    parser.add_argument("--epochs_stage2", type=int, default=20,
                       help="Epochs for stage 2 (fine-tuning)")
    parser.add_argument("--img_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument(
        "--classes",
        type=str,
        nargs="*",
        default=None,
        help="Restrict to these class folders (order defines label mapping)",
    )
    parser.add_argument(
        "--two_stage",
        action="store_true",
        default=True,
        help="Enable two-stage training (feature extraction + fine-tuning)",
    )
    parser.add_argument(
        "--unfreeze_layers",
        type=int,
        default=30,
        help="Number of top ResNet50 layers to unfreeze in stage 2",
    )
    return parser.parse_args()


def plot_training_history(history, stage, save_path):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history.get("accuracy", []), label="Train Accuracy", marker='o')
    axes[0].plot(history.history.get("val_accuracy", []), label="Val Accuracy", marker='s')
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{stage} - Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history.get("loss", []), label="Train Loss", marker='o')
    axes[1].plot(history.history.get("val_loss", []), label="Val Loss", marker='s')
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{stage} - Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("=" * 70)
    print("🧠 ResNet50 Transfer Learning for Brain Tumor Classification")
    print("=" * 70)

    # === LOAD DATA ===
    print("\n📂 Loading dataset...")
    augment = get_augmentation_pipeline()
    train_ds, val_ds, test_ds, class_names, train_counts = create_datasets(
        args.data_dir, 
        args.batch_size, 
        tuple(args.img_size), 
        augment, 
        allowed_classes=args.classes
    )
    
    num_classes = len(class_names)
    print(f"✅ Found {num_classes} classes: {class_names}")
    print(f"✅ Training samples per class: {train_counts}")

    # Persist class names for evaluation
    with open(os.path.join(args.results_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    # === BUILD MODEL ===
    print(f"\n🏗️  Building ResNet50 model (num_classes={num_classes})...")
    model, base_model = build_resnet50_model(
        input_shape=(*args.img_size, 3),
        num_classes=num_classes,
        freeze_base=True
    )
    
    print("\n📊 Model Architecture:")
    model.summary()
    
    # Count trainable parameters
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    print(f"\n📈 Trainable parameters: {trainable_count:,}")
    print(f"📉 Non-trainable parameters: {non_trainable_count:,}")
    print(f"📊 Total parameters: {trainable_count + non_trainable_count:,}")

    # === CLASS WEIGHTS (to handle class imbalance) ===
    total = sum(train_counts) if train_counts else 0
    class_weight = None
    if total > 0 and num_classes > 0:
        class_weight = {
            i: (total / (num_classes * max(1, cnt))) 
            for i, cnt in enumerate(train_counts)
        }
        print(f"\n⚖️  Class weights (to handle imbalance): {class_weight}")

    # === STAGE 1: FEATURE EXTRACTION ===
    print("\n" + "=" * 70)
    print("🚀 STAGE 1: Feature Extraction (Frozen ResNet50 base)")
    print("=" * 70)
    
    # Callbacks for Stage 1
    callbacks_stage1 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.results_dir, "best_model_stage1.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train Stage 1
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_stage1,
        callbacks=callbacks_stage1,
        class_weight=class_weight,
        verbose=1
    )

    # Save Stage 1 history plot
    plot_training_history(
        history_stage1, 
        "Stage 1: Feature Extraction",
        os.path.join(args.results_dir, "history_stage1.png")
    )
    
    # Stage 1 metrics
    stage1_metrics = {
        "stage1_val_acc": float(history_stage1.history.get("val_accuracy", [0])[-1]),
        "stage1_train_acc": float(history_stage1.history.get("accuracy", [0])[-1]),
        "stage1_val_loss": float(history_stage1.history.get("val_loss", [0])[-1]),
        "stage1_train_loss": float(history_stage1.history.get("loss", [0])[-1]),
    }
    
    print(f"\n✅ Stage 1 Complete!")
    print(f"   Train Accuracy: {stage1_metrics['stage1_train_acc']:.4f}")
    print(f"   Val Accuracy: {stage1_metrics['stage1_val_acc']:.4f}")

    # === STAGE 2: FINE-TUNING (Optional) ===
    history_stage2 = None
    if args.two_stage and args.epochs_stage2 > 0:
        print("\n" + "=" * 70)
        print("🔥 STAGE 2: Fine-Tuning (Unfreezing top ResNet50 layers)")
        print("=" * 70)
        
        # Unfreeze top layers
        model = unfreeze_base_model(model, base_model, args.unfreeze_layers)
        
        # Count trainable parameters after unfreezing
        trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
        print(f"\n📈 Trainable parameters: {trainable_count:,}")
        print(f"📉 Non-trainable parameters: {non_trainable_count:,}")
        
        # Callbacks for Stage 2
        callbacks_stage2 = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(args.results_dir, "best_model_stage2.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-8,
                verbose=1
            )
        ]

        # Train Stage 2
        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs_stage2,
            callbacks=callbacks_stage2,
            class_weight=class_weight,
            verbose=1
        )

        # Save Stage 2 history plot
        plot_training_history(
            history_stage2,
            "Stage 2: Fine-Tuning",
            os.path.join(args.results_dir, "history_stage2.png")
        )
        
        # Stage 2 metrics
        stage2_metrics = {
            "stage2_val_acc": float(history_stage2.history.get("val_accuracy", [0])[-1]),
            "stage2_train_acc": float(history_stage2.history.get("accuracy", [0])[-1]),
            "stage2_val_loss": float(history_stage2.history.get("val_loss", [0])[-1]),
            "stage2_train_loss": float(history_stage2.history.get("loss", [0])[-1]),
        }
        stage1_metrics.update(stage2_metrics)
        
        print(f"\n✅ Stage 2 Complete!")
        print(f"   Train Accuracy: {stage2_metrics['stage2_train_acc']:.4f}")
        print(f"   Val Accuracy: {stage2_metrics['stage2_val_acc']:.4f}")
        
        # Save final model
        model.save(os.path.join(args.results_dir, "best_model.keras"))
    else:
        # If no stage 2, save stage 1 model as final
        model.save(os.path.join(args.results_dir, "best_model.keras"))

    # === COMBINED HISTORY PLOT ===
    if history_stage2 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Combine histories
        stage1_acc = history_stage1.history.get("accuracy", [])
        stage1_val_acc = history_stage1.history.get("val_accuracy", [])
        stage2_acc = history_stage2.history.get("accuracy", [])
        stage2_val_acc = history_stage2.history.get("val_accuracy", [])
        
        stage1_loss = history_stage1.history.get("loss", [])
        stage1_val_loss = history_stage1.history.get("val_loss", [])
        stage2_loss = history_stage2.history.get("loss", [])
        stage2_val_loss = history_stage2.history.get("val_loss", [])
        
        epochs_stage1 = range(1, len(stage1_acc) + 1)
        epochs_stage2 = range(len(stage1_acc) + 1, len(stage1_acc) + len(stage2_acc) + 1)
        
        # Accuracy
        axes[0].plot(epochs_stage1, stage1_acc, 'b-', label='Stage 1 Train', marker='o')
        axes[0].plot(epochs_stage1, stage1_val_acc, 'b--', label='Stage 1 Val', marker='o')
        axes[0].plot(epochs_stage2, stage2_acc, 'r-', label='Stage 2 Train', marker='s')
        axes[0].plot(epochs_stage2, stage2_val_acc, 'r--', label='Stage 2 Val', marker='s')
        axes[0].axvline(x=len(stage1_acc), color='gray', linestyle=':', label='Fine-tuning starts')
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Combined Training History - Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(epochs_stage1, stage1_loss, 'b-', label='Stage 1 Train', marker='o')
        axes[1].plot(epochs_stage1, stage1_val_loss, 'b--', label='Stage 1 Val', marker='o')
        axes[1].plot(epochs_stage2, stage2_loss, 'r-', label='Stage 2 Train', marker='s')
        axes[1].plot(epochs_stage2, stage2_val_loss, 'r--', label='Stage 2 Val', marker='s')
        axes[1].axvline(x=len(stage1_acc), color='gray', linestyle=':', label='Fine-tuning starts')
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Combined Training History - Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, "history_combined.png"))
        plt.close()

    # Save all metrics
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(stage1_metrics, f, indent=4)

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"📁 Results saved to: {args.results_dir}")
    print(f"🎯 Final model: {os.path.join(args.results_dir, 'best_model.keras')}")
    print(f"📊 Metrics: {os.path.join(args.results_dir, 'metrics.json')}")
    print(f"📈 Plots: {os.path.join(args.results_dir, 'history_*.png')}")


if __name__ == "__main__":
    main()
