"""
Evaluation script for ResNet50 Transfer Learning Model

This script evaluates a trained ResNet50 model on the test set and generates:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC Curve (binary) or Multi-class ROC Curves
- Precision-Recall Curve
- Grad-CAM visualizations
- Per-class metrics visualization
"""

# src/models/resnet50/evaluate_resnet50.py
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    ConfusionMatrixDisplay,
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score,
    roc_auc_score
)
from src.common.dataset_utils import create_datasets
from src.common.gradcam import generate_gradcam


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ResNet50 on test split")
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
        help="Folder with saved model and outputs",
    )
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", 16)))
    parser.add_argument("--img_size", type=int, nargs=2, default=(224, 224))
    parser.add_argument(
        "--classes",
        type=str,
        nargs="*",
        default=None,
        help="Restrict to these class folders (order defines label mapping)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="best_model.keras",
        help="Name of the model file to evaluate",
    )
    return parser.parse_args()


def plot_multiclass_roc_curves(y_true, y_pred_proba, class_names, save_path):
    """Plot ROC curves for multi-class classification (one-vs-rest)."""
    from sklearn.preprocessing import label_binarize
    
    n_classes = len(class_names)
    
    # Binarize the labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, color in zip(range(n_classes), colors[:n_classes]):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return roc_auc


def plot_metrics_comparison(metrics_dict, class_names, save_path):
    """Create a grouped bar chart comparing precision, recall, and F1-score."""
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, metric in enumerate(metrics_names):
        values = metrics_dict[metric]
        offset = (i - 1) * width
        ax.bar(x + offset, values, width, label=metric, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, metric in enumerate(metrics_names):
        values = metrics_dict[metric]
        offset = (i - 1) * width
        for j, v in enumerate(values):
            ax.text(j + offset, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("📊 ResNet50 Model Evaluation")
    print("=" * 70)
    
    os.makedirs(os.path.join(args.results_dir, "gradcam"), exist_ok=True)

    # === LOAD DATA ===
    print("\n📂 Loading test dataset...")
    _, _, test_ds, class_names, _ = create_datasets(
        args.data_dir, 
        batch_size=args.batch_size, 
        img_size=tuple(args.img_size), 
        allowed_classes=args.classes
    )

    # Load saved class names if available
    class_names_path = os.path.join(args.results_dir, "class_names.json")
    if os.path.exists(class_names_path):
        try:
            with open(class_names_path) as f:
                saved_classes = json.load(f)
                if saved_classes:
                    class_names = saved_classes
        except Exception:
            pass
    
    num_classes = len(class_names)
    print(f"✅ Classes: {class_names}")

    # === LOAD MODEL ===
    print(f"\n🔍 Loading model: {args.model_name}...")
    model_path = os.path.join(args.results_dir, args.model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Recompile for evaluation
    if num_classes == 2:
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
    else:
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
    
    print("✅ Model loaded successfully")
    
    # === PREDICTIONS ===
    print("\n🔮 Generating predictions...")
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for imgs, labels in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_true.extend(labels.numpy())
        
        if num_classes == 2:
            # Binary classification
            y_pred_proba.extend(preds.flatten())
            y_pred.extend((preds > 0.5).astype(int).flatten())
        else:
            # Multi-class classification
            y_pred_proba.extend(preds)
            y_pred.extend(np.argmax(preds, axis=1))
    
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    y_pred_proba = np.array(y_pred_proba)
    
    # === METRICS ===
    print("\n📈 Computing metrics...")
    acc = np.mean(y_true == y_pred)
    print(f"✅ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    metrics = {"test_acc": float(acc)}

    # === CONFUSION MATRIX ===
    print("\n📊 Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - ResNet50', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # === CLASSIFICATION REPORT ===
    print("\n📋 Generating classification report...")
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(rep)
    
    with open(os.path.join(args.results_dir, "classification_report.txt"), "w") as f:
        f.write("ResNet50 Transfer Learning - Classification Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(rep)

    # === PER-CLASS METRICS VISUALIZATION ===
    print("\n📊 Generating per-class metrics visualization...")
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics_dict = {
            'Precision': precision_score(y_true, y_pred, average=None, zero_division=0),
            'Recall': recall_score(y_true, y_pred, average=None, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average=None, zero_division=0)
        }
        
        plot_metrics_comparison(
            metrics_dict, 
            class_names,
            os.path.join(args.results_dir, "metrics_by_class.png")
        )
        
        # Store average metrics
        metrics.update({
            "precision_macro": float(np.mean(metrics_dict['Precision'])),
            "recall_macro": float(np.mean(metrics_dict['Recall'])),
            "f1_macro": float(np.mean(metrics_dict['F1-Score']))
        })
        
    except Exception as e:
        print(f"⚠️ Error generating per-class metrics: {e}")

    # === ROC CURVES ===
    print("\n📈 Generating ROC curves...")
    try:
        if num_classes == 2:
            # Binary ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            metrics["roc_auc"] = float(roc_auc)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve - ResNet50', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_dir, "roc_curve.png"), dpi=300)
            plt.close()
            
            print(f"✅ ROC AUC: {roc_auc:.4f}")
        else:
            # Multi-class ROC curves
            roc_auc_dict = plot_multiclass_roc_curves(
                y_true, 
                y_pred_proba, 
                class_names,
                os.path.join(args.results_dir, "roc_curve_multiclass.png")
            )
            
            # Calculate macro-average AUC
            macro_auc = np.mean(list(roc_auc_dict.values()))
            metrics["roc_auc_macro"] = float(macro_auc)
            
            for i, cls in enumerate(class_names):
                metrics[f"roc_auc_{cls}"] = float(roc_auc_dict[i])
            
            print(f"✅ ROC AUC (Macro): {macro_auc:.4f}")
            for i, cls in enumerate(class_names):
                print(f"   {cls}: {roc_auc_dict[i]:.4f}")
                
    except Exception as e:
        print(f"⚠️ Error generating ROC curves: {e}")

    # === PRECISION-RECALL CURVE ===
    print("\n📈 Generating Precision-Recall curve...")
    try:
        if num_classes == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall, precision)
            average_precision = average_precision_score(y_true, y_pred_proba)
            metrics["pr_auc"] = float(pr_auc)
            metrics["average_precision"] = float(average_precision)
            
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AP = {average_precision:.3f})')
            plt.axhline(y=sum(y_true)/len(y_true), color='red', linestyle='--',
                       label=f'No Skill ({sum(y_true)/len(y_true):.3f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curve - ResNet50', fontsize=14, fontweight='bold')
            plt.legend(loc="upper right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_dir, "precision_recall_curve.png"), dpi=300)
            plt.close()
            
            print(f"✅ Average Precision: {average_precision:.4f}")
    except Exception as e:
        print(f"⚠️ Error generating PR curve: {e}")

    # === SAVE METRICS ===
    metrics_path = os.path.join(args.results_dir, "metrics.json")
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

    # === GRAD-CAM VISUALIZATIONS ===
    print("\n🔥 Generating Grad-CAM visualizations...")
    try:
        generate_gradcam(
            model, 
            test_ds, 
            os.path.join(args.results_dir, "gradcam"), 
            class_names,
            num_images=10
        )
    except Exception as e:
        print(f"⚠️ Error generating Grad-CAM: {e}")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)
    print(f"📁 Results saved to: {args.results_dir}")
    print(f"📊 Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    if "roc_auc" in metrics:
        print(f"📈 ROC AUC: {metrics['roc_auc']:.4f}")
    if "roc_auc_macro" in metrics:
        print(f"📈 ROC AUC (Macro): {metrics['roc_auc_macro']:.4f}")
    print("\n📂 Generated files:")
    print(f"   - confusion_matrix.png")
    print(f"   - classification_report.txt")
    print(f"   - metrics_by_class.png")
    print(f"   - roc_curve*.png")
    if num_classes == 2:
        print(f"   - precision_recall_curve.png")
    print(f"   - gradcam/ (visualizations)")
    print(f"   - metrics.json")


if __name__ == "__main__":
    main()
