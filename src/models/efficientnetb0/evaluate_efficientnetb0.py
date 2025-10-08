import os, json, argparse, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from src.common.dataset_utils import create_datasets
from src.common.gradcam import generate_gradcam

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate EfficientNetB0 model")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="./results/efficientnetb0")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size", type=int, nargs=2, default=(224, 224))
    p.add_argument("--limit_eval", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    _, _, test_ds, class_names, _ = create_datasets(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size),
        allowed_classes=["yes", "no"]
    )

    model_path = os.path.join(args.results_dir, "best_model.keras")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    y_true, y_pred = [], []
    for i, (x, y) in enumerate(test_ds):
        preds = model.predict(x, verbose=0)
        y_true.extend(y.numpy())
        y_pred.extend((preds > 0.5).astype(int).flatten())
        if args.limit_eval and i >= args.limit_eval:
            break

    y_true, y_pred = np.array(y_true, int), np.array(y_pred, int)
    acc = np.mean(y_true == y_pred)
    print(f"✅ Test Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
    plt.title("Confusion Matrix - EfficientNetB0")
    plt.savefig(os.path.join(args.results_dir, "confusion_matrix.png"))
    plt.close()

    rep = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(args.results_dir, "classification_report.txt"), "w") as f:
        f.write(rep)

    # Save metrics
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": float(acc)}, f, indent=4)

    # Grad-CAM
    generate_gradcam(model, test_ds, os.path.join(args.results_dir, "gradcam"), class_names)
    print("✅ Evaluation complete.")

if __name__ == "__main__":
    main()
