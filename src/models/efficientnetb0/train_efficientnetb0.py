import os, json, argparse, tensorflow as tf
from src.common.dataset_utils import create_datasets
from src.models.efficientnetb0.build_efficientnetb0 import build_efficientnetb0_model

def parse_args():
    p = argparse.ArgumentParser(description="Train EfficientNetB0 model")
    p.add_argument("--data_dir", type=str, required=True, help="Path to dataset (train/val/test)")
    p.add_argument("--results_dir", type=str, default="./results/efficientnetb0", help="Where to save outputs")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--img_size", type=int, nargs=2, default=(224, 224))
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    train_ds, val_ds, test_ds, class_names, train_counts = create_datasets(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size),
        allowed_classes=["yes", "no"]
    )

    print(f"✅ Classes: {class_names}")
    print(f"✅ Training samples: {sum(train_counts)}")

    model = build_efficientnetb0_model(input_shape=(*args.img_size, 3))

    ckpt_path = os.path.join(args.results_dir, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_accuracy', mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Save history
    hist_json = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(os.path.join(args.results_dir, "history.json"), "w") as f:
        json.dump(hist_json, f, indent=2)

    model.save(os.path.join(args.results_dir, "final_model.keras"))
    print("✅ Training complete, model saved.")

if __name__ == "__main__":
    main()
