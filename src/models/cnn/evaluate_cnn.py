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
        help="Folder containing class subfolders (yes/no) or train/val/test structure",
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
    parser.add_argument(
        "--use_processed",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use the processed data structure with train/val/test folders (0=auto-detect, 1=force)",
    )
    parser.add_argument(
        "--limit_eval",
        type=int,
        default=0,
        help="Limit evaluation to specified number of batches (0=no limit, useful for faster evaluation)",
    )
    return parser.parse_args()


def create_test_dataset_from_processed(test_dir, batch_size=32, img_size=(224, 224)):
    """Create a test dataset from a directory with class subdirectories.
    For the processed data structure where test data is in its own directory.
    """
    # Check for existence of expected class folders
    yes_dir = os.path.join(test_dir, 'yes')
    no_dir = os.path.join(test_dir, 'no')
    
    if not os.path.exists(yes_dir) or not os.path.exists(no_dir):
        print(f"Warning: Expected class folders not found in {test_dir}")
        print(f"Yes folder exists: {os.path.exists(yes_dir)}")
        print(f"No folder exists: {os.path.exists(no_dir)}")
        print("Available directories:")
        for item in os.listdir(test_dir):
            if os.path.isdir(os.path.join(test_dir, item)):
                print(f" - {item}")
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Important for evaluation
    )
    
    # Convert the Keras generator to a TensorFlow dataset
    test_ds = tf.data.Dataset.from_generator(
        lambda: test_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    
    return test_ds, list(test_gen.class_indices.keys())

def main():
    args = parse_args()
    os.makedirs(os.path.join(args.results_dir, "gradcam"), exist_ok=True)
    
    # Check if we should use the processed data structure
    test_dir = os.path.join(args.data_dir, 'test')
    use_processed = args.use_processed == 1 or os.path.exists(test_dir)
    
    if args.limit_eval > 0:
        print(f"⚡ FAST MODE: Limiting evaluation to {args.limit_eval} batches for quicker results")
    
    if use_processed and os.path.exists(test_dir):
        print(f"Using processed data structure. Test directory: {test_dir}")
        test_ds, class_names = create_test_dataset_from_processed(
            test_dir, batch_size=args.batch_size, img_size=tuple(args.img_size)
        )
    else:
        print(f"Using original data structure with class folders directly under {args.data_dir}")
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
    batch_count = 0
    print("Evaluating model on test data...")
    
    for imgs, labels in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(labels.numpy())
        # Binary head: sigmoid single unit
        y_pred.extend((preds > 0.5).astype(int).flatten())
        
        batch_count += 1
        print(f"Processed batch {batch_count}", end="\r")
        
        # If limit_eval is set, stop after processing that many batches
        if args.limit_eval > 0 and batch_count >= args.limit_eval:
            print(f"\nReached evaluation limit ({args.limit_eval} batches)")
            break

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
        # Get raw probabilities for the curves - use a limited number of batches for faster evaluation
        y_true_roc, y_prob_roc = [], []
        
        # Use args.limit_eval if specified, otherwise default to 10 batches for curves
        max_eval_batches = args.limit_eval if args.limit_eval > 0 else 10
        batch_count = 0
        
        print(f"Generating predictions for ROC curve and PR curve (max {max_eval_batches} batches)...")
        for imgs, labels in test_ds:
            probs = model.predict(imgs, verbose=0)
            y_true_roc.extend(labels.numpy())
            y_prob_roc.extend(probs.flatten())
            
            batch_count += 1
            print(f"Processed ROC batch {batch_count}/{max_eval_batches}", end="\r")
            if batch_count >= max_eval_batches:
                break
                
        y_true_roc = np.array(y_true_roc)
        y_prob_roc = np.array(y_prob_roc)
        
        print(f"Using {len(y_true_roc)} samples for curves (limited for speed)")
        
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

    # Grad-CAM - also respect the limit_eval parameter
    print("Generating Grad-CAM visualizations...")
    max_gradcam_samples = args.limit_eval if args.limit_eval > 0 else None
    generate_gradcam(model, test_ds, os.path.join(args.results_dir, "gradcam"), class_names, max_samples=max_gradcam_samples)
    print("✅ Evaluation complete.")


if __name__ == "__main__":
    main()
