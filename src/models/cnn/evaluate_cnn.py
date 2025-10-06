"""
Evaluation entrypoint for the simple CNN on the test split created from a provided folder.

By default, reads model and class names from the results directory used during training.
"""

# src/models/cnn/evaluate_cnn.py
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from src.common.dataset_utils import create_datasets
from src.common.gradcam import generate_gradcam


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CNN on test split")
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
        help="Folder with saved model and outputs",
    )
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", 32)))
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
    os.makedirs(os.path.join(args.results_dir, "gradcam"), exist_ok=True)

    # Load datasets with the same split logic
    _, _, test_ds, class_names = create_datasets(
        args.data_dir, batch_size=args.batch_size, img_size=tuple(args.img_size), allowed_classes=args.classes
    )

    # Fall back to saved class names if present
    class_names_path = os.path.join(args.results_dir, "class_names.json")
    if os.path.exists(class_names_path):
        try:
            with open(class_names_path) as f:
                saved_classes = json.load(f)
                if saved_classes:
                    class_names = saved_classes
        except Exception:
            pass

    # Load model
    model_path = os.path.join(args.results_dir, "best_model.h5")
    model = tf.keras.models.load_model(model_path)

    # Predict
    y_true, y_pred = [], []
    for imgs, labels in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(labels.numpy())
        # Binary head: sigmoid single unit
        y_pred.extend((preds > 0.5).astype(int).flatten())

    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    acc = np.mean(y_true == y_pred)
    print(f"✅ Test Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Ensure labels length matches matrix shape
    display_labels = class_names
    if len(display_labels) != cm.shape[0]:
        # Fallback to numeric labels if mismatch occurs
        display_labels = list(map(str, range(cm.shape[0])))
    ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(cmap="Blues")
    plt.title("Confusion Matrix - CNN")
    plt.savefig(os.path.join(args.results_dir, "confusion_matrix.png"))
    plt.close()

    # Classification report
    rep = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(args.results_dir, "classification_report.txt"), "w") as f:
        f.write(rep)

    # Append/Write metrics
    metrics_path = os.path.join(args.results_dir, "metrics.json")
    metrics = {"test_acc": float(acc)}
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                prev = json.load(f)
            prev.update(metrics)
            with open(metrics_path, "w") as f:
                json.dump(prev, f, indent=4)
        else:
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
    except Exception:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    # Grad-CAM
    generate_gradcam(model, test_ds, os.path.join(args.results_dir, "gradcam"), class_names)
    print("✅ Evaluation complete.")


if __name__ == "__main__":
    main()
