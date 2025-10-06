"""
Training entrypoint for the simple CNN.

Now supports passing paths and hyperparameters via CLI args or env vars, so it can run easily in Colab.

Expected dataset layout for --data_dir:
  data_dir/
    yes/
      img1.jpg ...
    no/
      img2.jpg ...

The loader will create train/val/test splits internally.
"""

# src/models/cnn/train_cnn.py
import os
import json
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from src.models.cnn.build_cnn import build_cnn_model
from src.common.dataset_utils import create_datasets
from src.common.preprocessing import get_augmentation_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN for Brain Tumor classification")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.getenv("DATA_DIR", "/content/drive/MyDrive/BrainTumor"),
        help="Folder containing class subfolders (yes/no)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.getenv("RESULTS_DIR", "/content/drive/MyDrive/brain_tumor_project/results/cnn"),
        help="Folder to write checkpoints and plots",
    )
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", 32)))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", 30)))
    parser.add_argument("--img_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument(
        "--classes",
        type=str,
        nargs="*",
        default=os.getenv("CLASSES", "yes no").split(),
        help="Restrict to these class folders (order defines label mapping)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # === LOAD DATA ===
    augment = get_augmentation_pipeline()
    train_ds, val_ds, test_ds, class_names, train_counts = create_datasets(
        args.data_dir, args.batch_size, tuple(args.img_size), augment, allowed_classes=args.classes
    )

    # Persist class names for evaluation scripts
    with open(os.path.join(args.results_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    # === BUILD MODEL ===
    model = build_cnn_model((*args.img_size, 3))
    model.summary()

    # === CLASS WEIGHTS (to mitigate class imbalance) ===
    total = sum(train_counts) if train_counts else 0
    class_weight = None
    if total > 0:
        # Inverse frequency: weight_i = total / (num_classes * count_i)
        num_classes = len(train_counts)
        class_weight = {
            i: (total / (num_classes * max(1, cnt))) for i, cnt in enumerate(train_counts)
        }
        print("Using class weights:", class_weight)
        # Convert to per-example sample weights to avoid Keras bug with class_weight
        weight_vec = tf.constant([class_weight[i] for i in range(len(class_weight))], dtype=tf.float32)
        train_ds_sw = train_ds.map(
            lambda x, y: (x, y, tf.gather(weight_vec, tf.cast(y, tf.int32))),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        train_ds = train_ds_sw

    # === CALLBACKS ===
    # Learning rate schedule: cosine decay
    initial_lr = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr, decay_steps=max(1, args.epochs) * 100
    )

    # Recompile model with LR schedule
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=model.loss,
        metrics=model.metrics,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.results_dir, "best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
    ]

    # === TRAIN ===
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        # class_weight not used; per-example sample weights are provided by dataset
    )

    # === SAVE PLOTS ===
    plt.figure(figsize=(8, 4))
    plt.plot(history.history.get("accuracy", []), label="Train")
    plt.plot(history.history.get("val_accuracy", []), label="Val")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "history.png"))

    # Save last epoch metrics for quick reference
    metrics_out = {
        "val_acc": float(history.history.get("val_accuracy", [None])[-1] or 0.0),
        "train_acc": float(history.history.get("accuracy", [None])[-1] or 0.0),
    }
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=4)

    print("✅ Training complete. Best model saved to:", args.results_dir)


if __name__ == "__main__":
    main()
