"""
Training entrypoint for the simple CNN.

Now supports passing paths and hyperparameters via CLI args or env vars, so it can run easily in Colab.

Supports two different dataset layouts:

1. Original layout:
  data_dir/
    yes/
      img1.jpg ...
    no/
      img2.jpg ...
  The loader will create train/val/test splits internally.

2. Processed layout with predefined splits:
  data_dir/
    train/
      yes/
        img1.jpg ...
      no/
        img2.jpg ...
    val/
      yes/ ...
      no/ ...
    test/
      yes/ ...
      no/ ...
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
        help="Folder containing class subfolders (yes/no) or train/val/test structure",
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
    parser.add_argument(
        "--use_processed",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use the processed data structure with train/val/test folders (0=auto-detect, 1=force)",
    )
    return parser.parse_args()


def create_dataset_from_processed(data_path, subset, img_size=(224, 224), batch_size=32, is_training=False):
    """Create a dataset from a directory with train/val/test structure.
    For the processed data structure with predefined train/val/test splits.
    """
    subset_dir = os.path.join(data_path, subset)
    if not os.path.exists(subset_dir):
        raise FileNotFoundError(f"Directory not found: {subset_dir}")
        
    if is_training:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
    return datagen.flow_from_directory(
        subset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=is_training
    )

def create_datasets_from_processed(data_dir, batch_size=32, img_size=(224, 224)):
    """Create datasets from a directory with train/val/test structure."""
    # Create generators for train, val, and test sets
    train_gen = create_dataset_from_processed(
        data_dir, 'train', img_size, batch_size, is_training=True
    )
    val_gen = create_dataset_from_processed(
        data_dir, 'val', img_size, batch_size, is_training=False
    )
    test_gen = create_dataset_from_processed(
        data_dir, 'test', img_size, batch_size, is_training=False
    )
    
    # Get class names
    class_names = list(train_gen.class_indices.keys())
    
    # Calculate class counts for weighting
    train_counts = [0] * len(class_names)
    classes = train_gen.classes
    for cls_idx in classes:
        train_counts[cls_idx] += 1
    
    # Convert to TensorFlow datasets
    train_ds = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_generator(
        lambda: val_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_generator(
        lambda: test_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names, train_counts

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # Check if we should use the processed data structure
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')
    
    use_processed = args.use_processed == 1 or (
        os.path.exists(train_dir) and 
        os.path.exists(val_dir) and 
        os.path.exists(test_dir)
    )
    
    # === LOAD DATA ===
    if use_processed:
        print(f"Using processed data structure with train/val/test folders from: {args.data_dir}")
        train_ds, val_ds, test_ds, class_names, train_counts = create_datasets_from_processed(
            args.data_dir, args.batch_size, tuple(args.img_size)
        )
    else:
        print(f"Using original data structure with class folders from: {args.data_dir}")
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

    # === CALLBACKS ===
    # Learning rate schedule: cosine decay
    initial_lr = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr, decay_steps=max(1, args.epochs) * 100
    )

    # Prepare loss (use weighted BCE if class weights available)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    if class_weight is not None:
        cw0 = float(class_weight.get(0, 1.0))
        cw1 = float(class_weight.get(1, 1.0))

        def weighted_bce(y_true, y_pred):
            y_true_f = tf.cast(y_true, tf.float32)
            y_pred_f = tf.cast(y_pred, tf.float32)
            # Ensure shapes match (e.g., (B,) -> (B,1) to match model output)
            y_true_f = tf.reshape(y_true_f, tf.shape(y_pred_f))
            # Clip to avoid log(0)
            y_pred_f = tf.clip_by_value(y_pred_f, 1e-7, 1.0 - 1e-7)
            # Weighted BCE formula
            pos_term = -cw1 * y_true_f * tf.math.log(y_pred_f)
            neg_term = -cw0 * (1.0 - y_true_f) * tf.math.log(1.0 - y_pred_f)
            loss = pos_term + neg_term
            return tf.reduce_mean(loss)

        loss_fn = weighted_bce

    # Recompile model with LR schedule and selected loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=['accuracy'],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.results_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
    ]

    # === TRAIN ===
    # Calculate steps per epoch to fix the "Unknown" progress issue
    steps_per_epoch = None
    validation_steps = None
    
    # Get steps per epoch from the generator if using ImageDataGenerator
    if hasattr(train_ds, 'samples'):
        steps_per_epoch = train_ds.samples // args.batch_size
        print(f"Setting steps_per_epoch to: {steps_per_epoch}")
    
    if hasattr(val_ds, 'samples'):
        validation_steps = val_ds.samples // args.batch_size
        print(f"Setting validation_steps to: {validation_steps}")
    
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
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
