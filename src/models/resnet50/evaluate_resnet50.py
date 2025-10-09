"""
Evaluation script for ResNet50 brain tumor classification.

Simplified version for single optimized ResNet50 model evaluation.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.common.gradcam import generate_gradcam
from tensorflow.keras.applications.resnet50 import preprocess_input


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ResNet50 for Brain Tumor classification")
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
        help="Directory containing results and model",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file (if not provided, uses best_model.h5 in results_dir)"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument("--generate_gradcam", action="store_true", help="Generate Grad-CAM visualizations")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine model path
    if args.model_path is None:
        model_path = os.path.join(args.results_dir, "best_model.h5")
    else:
        model_path = args.model_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print(f"📊 Starting ResNet50 evaluation...")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Results directory: {args.results_dir}")
    print(f"   Model path: {model_path}")
    print(f"   Input size: {args.input_size}x{args.input_size}")
    
    # Create gradcam directory
    os.makedirs(os.path.join(args.results_dir, "gradcam"), exist_ok=True)
    
    # === Load Data ===
    # Use the preprocessed data directly without additional preprocessing
    print(f"Loading data from preprocessed directories: {args.data_dir}")
    
    # Define image data generator for test data with proper ResNet50 preprocessing
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
    
    # Load test data from directory structure
    test_ds = test_datagen.flow_from_directory(
        os.path.join(args.data_dir, 'test'),
        target_size=(args.input_size, args.input_size),
        batch_size=args.batch_size,
        class_mode='binary',
        shuffle=False,
        classes=['no', 'yes']
    )
    
    class_names = ['no', 'yes']
    print(f"✅ Dataset loaded with classes: {class_names}")
    print(f"   Found {test_ds.samples} test samples")
    
    # === Load Model ===
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded from: {model_path}")
    
    # === Evaluate on Test Set ===
    print("🔍 Evaluating on test set...")
    
    # Reset the test dataset to ensure we start from the beginning
    test_ds.reset()
    
    # Get model evaluation
    evaluation = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")
    
    # Get predictions
    test_ds.reset()
    y_true = test_ds.classes
    steps = (test_ds.samples + test_ds.batch_size - 1) // test_ds.batch_size
    
    predictions = model.predict(test_ds, steps=steps, verbose=1)
    y_pred_proba = predictions.flatten()
    y_pred = (predictions > 0.5).astype(int).flatten()
    
    # Calculate comprehensive metrics
    test_accuracy = accuracy_score(y_true, y_pred)
    test_precision = precision_score(y_true, y_pred)
    test_recall = recall_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred)
    
    print(f"📈 Test Results:")
    print(f"   Accuracy: {test_accuracy:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall: {test_recall:.4f}")
    print(f"   F1-Score: {test_f1:.4f}")
    
    # === Generate Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix - ResNet50", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # === Generate Classification Report ===
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"\n📋 Classification Report:")
    print("="*60)
    print(report)
    
    with open(os.path.join(args.results_dir, "classification_report.txt"), "w") as f:
        f.write("Classification Report - ResNet50\n")
        f.write("="*60 + "\n")
        f.write(report)
    
    # === Save Evaluation Metrics ===
    evaluation_metrics = {
        "test_accuracy": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1_score": float(test_f1),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "total_test_samples": len(y_true)
    }
    
    # Load existing metrics if they exist
    metrics_path = os.path.join(args.results_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            existing_metrics = json.load(f)
        existing_metrics.update(evaluation_metrics)
        evaluation_metrics = existing_metrics
    
    with open(metrics_path, "w") as f:
        json.dump(evaluation_metrics, f, indent=4)
    
    # === Generate Grad-CAM ===
    if args.generate_gradcam:
        print("🔍 Generating Grad-CAM visualizations...")
        try:
            # Create a TF dataset from some test images for GradCAM
            test_images = []
            test_labels = []
            
            # Get a batch of images for GradCAM from the test_ds
            test_ds.reset()  # Reset before getting batch
            batch_x, batch_y = next(test_ds)
            
            # Add batch data to our lists
            for i in range(len(batch_x)):
                test_images.append(batch_x[i])
                test_labels.append(int(batch_y[i]))
                
            # Convert to tensors
            test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)
            test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int32)
            
            # Create a TensorFlow dataset that the original GradCAM function can work with
            gradcam_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
            gradcam_ds = gradcam_ds.batch(min(len(test_images), args.batch_size))
            
            generate_gradcam(
                model=model, 
                dataset=gradcam_ds, 
                save_dir=os.path.join(args.results_dir, "gradcam"), 
                class_names=class_names,
                num_images=5,
                max_samples=3
            )
            print("✅ Grad-CAM visualizations generated!")
        except Exception as e:
            print(f"⚠️  Grad-CAM generation failed: {str(e)}")
    
    # === Generate Sample Predictions Visualization ===
    plt.figure(figsize=(15, 10))
    max_samples = 12

    # Build a visual-only generator (rescaled for display), then preprocess for prediction
    vis_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    vis_flow = vis_datagen.flow_from_directory(
        os.path.join(args.data_dir, 'test'),
        target_size=(args.input_size, args.input_size),
        batch_size=max_samples,
        class_mode='binary',
        shuffle=True,
        classes=['no', 'yes']
    )

    batch_x_vis, batch_y_vis = next(vis_flow)
    # Prepare inputs for model prediction
    batch_x_pred = preprocess_input(batch_x_vis * 255.0)
    predictions = model.predict(batch_x_pred, verbose=0)

    for i in range(min(len(batch_x_vis), max_samples)):
        plt.subplot(3, 4, i + 1)
        plt.imshow(batch_x_vis[i])  # nicely scaled [0,1] RGB

        # Get true and predicted labels
        true_label = class_names[int(batch_y_vis[i])]
        pred_prob = predictions[i][0]
        pred_label = class_names[1] if pred_prob > 0.5 else class_names[0]
        confidence = pred_prob if pred_prob > 0.5 else (1 - pred_prob)

        # Color based on correctness
        color = 'green' if true_label == pred_label else 'red'

        plt.title(
            f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}',
            color=color, fontsize=10
        )
        plt.axis('off')
    
    plt.suptitle('ResNet50 - Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "sample_predictions.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Evaluation complete!")
    print(f"   Test accuracy: {test_accuracy:.4f}")
    print(f"   Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
