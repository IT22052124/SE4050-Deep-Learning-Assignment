import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from src.common.dataset_utils import create_datasets
from src.common.gradcam import generate_gradcam

DATA_DIR = "/content/drive/MyDrive/brain_tumor_project/data/processed"
RESULTS_DIR = "/content/drive/MyDrive/brain_tumor_project/results/vgg16"
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.h5")

os.makedirs(os.path.join(RESULTS_DIR, "gradcam"), exist_ok=True)

train_ds, val_ds, test_ds, class_names = create_datasets(DATA_DIR, batch_size=32, img_size=(128,128))

# === Load Model ===
model = tf.keras.models.load_model(MODEL_PATH)

# === Evaluate ===
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend((preds > 0.5).astype(int).flatten())

acc = np.mean(np.array(y_true) == np.array(y_pred))
print(f"✅ Test Accuracy: {acc:.4f}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - VGG16")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

generate_gradcam(model, test_ds, save_dir=os.path.join(RESULTS_DIR, "gradcam"), class_names=class_names)

print("✅ Evaluation complete. Results saved to:", RESULTS_DIR)
