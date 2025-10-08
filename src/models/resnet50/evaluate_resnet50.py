"""
Evaluation script for ResNet50 brain tumor classification.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.common.dataset_utils import create_datasets
from src.common.gradcam import generate_gradcam


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ResNet50 for Brain Tumor classification")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.getenv("DATA_DIR", "/content/drive/MyDrive/BrainTumor/data/processed"),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.getenv("RESULTS_DIR", "/content/drive/MyDrive/BrainTumor/Result/resnet50"),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to .h5 model file"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--generate_gradcam", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model_path or os.path.join(args.results_dir, "best_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"📊 Evaluating ResNet50 model from {model_path}")

    # === Load Data ===
    train_ds, val_ds, test_ds, class_names = create_datasets(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=(args.input_size, args.input_size)
    )
    print(f"✅ Dataset loaded with classes: {class_names}")

    # === Load Model ===
    model = tf.keras.models.load_model(model_path)

    y_true, y_pred, y_proba = [], [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_proba.extend(preds.flatten())
        y_pred.extend((preds > 0.5).astype(int).flatten())

    # === Metrics ===
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"📈 Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - ResNet50")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "confusion_matrix.png"), dpi=300)
    plt.show()

    # === Classification Report ===
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(args.results_dir, "classification_report.txt"), "w") as f:
        f.write("Classification Report - ResNet50\n")
        f.write("="*60 + "\n" + report)

    # === Save metrics ===
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names
    }
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # === Grad-CAM ===
    if args.generate_gradcam:
        print("🎯 Generating Grad-CAM...")
        generate_gradcam(
            model=model,
            dataset=test_ds,
            save_dir=os.path.join(args.results_dir, "gradcam"),
            class_names=class_names,
            num_images=5,
            max_samples=3
        )
        print("✅ Grad-CAM saved!")

    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
