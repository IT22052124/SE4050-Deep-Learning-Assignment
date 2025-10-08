"""
Training script for ResNet50 brain tumor classification.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from src.models.resnet50.build_resnet50 import (
    build_resnet50_basic,
    build_resnet50_fine_tuned,
    build_resnet50_enhanced
)
from src.common.dataset_utils import create_datasets
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
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=["basic", "fine_tuned", "enhanced"],
        default="enhanced",
        help="ResNet50 model variant to train"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--input_size", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"🚀 Starting ResNet50 training...")
    print(f"   Model variant: {args.model_variant}")
    print(f"   Data dir: {args.data_dir}")
    print(f"   Results dir: {args.results_dir}")

    # === Load Data ===
    augment = get_augmentation_pipeline()
    train_ds, val_ds, test_ds, class_names, train_counts = create_datasets(
        args.data_dir,
        args.batch_size,
        img_size=(args.input_size, args.input_size),
        augment_fn=augment
    )
    print(f"✅ Dataset loaded with classes: {class_names}")
    print(f"   Training samples per class: {train_counts}")

    # === Build Model ===
    input_shape = (args.input_size, args.input_size, 3)
    if args.model_variant == "basic":
        model = build_resnet50_basic(input_shape)
    elif args.model_variant == "fine_tuned":
        model = build_resnet50_fine_tuned(input_shape)
    else:
        model = build_resnet50_enhanced(input_shape)

    print(f"✅ Built {args.model_variant} ResNet50 model")
    model.summary()

    with open(os.path.join(args.results_dir, f"{args.model_variant}_model_summary.txt"), "w") as f:
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

    # === Train ===
    print(f"🏋️ Training {args.model_variant} ResNet50 model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # === Save Training Plots ===
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('ResNet50 Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('ResNet50 Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "training_plot.png"), dpi=300)
    plt.show()

    # === Save Metrics ===
    metrics = {
        "model_variant": args.model_variant,
        "train_accuracy": float(max(history.history['accuracy'])),
        "val_accuracy": float(max(history.history['val_accuracy'])),
        "train_loss": float(min(history.history['loss'])),
        "val_loss": float(min(history.history['val_loss'])),
        "epochs_trained": len(history.history['accuracy']),
        "config": vars(args)
    }

    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Training complete! Best val accuracy: {metrics['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
