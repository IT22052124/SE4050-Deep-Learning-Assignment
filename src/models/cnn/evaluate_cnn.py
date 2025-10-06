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
    # create_datasets now returns (train, val, test, class_names, train_counts)
    _, _, test_ds, class_names, _ = create_datasets(
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

    # Load model: prefer native Keras format, fallback to H5 (compile=False)
    keras_path = os.path.join(args.results_dir, "best_model.keras")
    h5_path = os.path.join(args.results_dir, "best_model.h5")
    if os.path.exists(keras_path):
        model = tf.keras.models.load_model(keras_path, compile=False)
    elif os.path.exists(h5_path):
        model = tf.keras.models.load_model(h5_path, compile=False)
    else:
        raise FileNotFoundError(f"No model checkpoint found at {keras_path} or {h5_path}")

    # Compile with standard loss/metric for evaluation and predict
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) 

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
        
    # Create a visual classification report
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score
        import seaborn as sns
        
        metrics = {
            'Precision': precision_score(y_true, y_pred, average=None),
            'Recall': recall_score(y_true, y_pred, average=None),
            'F1-Score': f1_score(y_true, y_pred, average=None)
        }
        
        plt.figure(figsize=(10, 6))
        for i, metric_name in enumerate(metrics.keys()):
            plt.subplot(1, 3, i+1)
            values = metrics[metric_name]
            sns.barplot(x=class_names, y=values)
            plt.title(metric_name)
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, "metrics_by_class.png"))
        plt.close()
    except Exception as e:
        print(f"⚠️ Error generating classification metrics visualization: {e}")

    # Append/Write metrics
    metrics_path = os.path.join(args.results_dir, "metrics.json")
    metrics = {"test_acc": float(acc)}
    
    # Generate ROC curve and precision-recall curve
    try:
        # Get raw probabilities for the curves
        y_true_roc, y_prob_roc = [], []
        for imgs, labels in test_ds:
            probs = model.predict(imgs, verbose=0)
            y_true_roc.extend(labels.numpy())
            y_prob_roc.extend(probs.flatten())
            
        y_true_roc = np.array(y_true_roc)
        y_prob_roc = np.array(y_prob_roc)
        
        # Import necessary metrics from sklearn
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true_roc, y_prob_roc)
        roc_auc = auc(fpr, tpr)
        metrics["roc_auc"] = float(roc_auc)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.results_dir, "roc_curve.png"))
        plt.close()
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true_roc, y_prob_roc)
        pr_auc = auc(recall, precision)
        average_precision = average_precision_score(y_true_roc, y_prob_roc)
        metrics["pr_auc"] = float(pr_auc)
        metrics["average_precision"] = float(average_precision)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {average_precision:.3f})')
        plt.axhline(y=sum(y_true_roc)/len(y_true_roc), color='red', linestyle='--', 
                   label=f'No Skill ({sum(y_true_roc)/len(y_true_roc):.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(args.results_dir, "precision_recall_curve.png"))
        plt.close()
        
        print(f"✅ ROC AUC: {roc_auc:.4f}, Average Precision: {average_precision:.4f}")
    except Exception as e:
        print(f"⚠️ Error generating ROC/PR curves: {e}")
    
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
    except Exception as e:
        print(f"⚠️ Error saving metrics: {e}")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    # Grad-CAM
    generate_gradcam(model, test_ds, os.path.join(args.results_dir, "gradcam"), class_names)
    print("✅ Evaluation complete.")


if __name__ == "__main__":
    main()
