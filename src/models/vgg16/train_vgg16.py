import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from src.models.vgg16.build_vgg16 import build_vgg16_model
from src.common.dataset_utils import create_datasets
from src.common.preprocessing import get_augmentation_pipeline

DATA_DIR = "/content/drive/MyDrive/brain_tumor_project/data/processed"
RESULTS_DIR = "/content/drive/MyDrive/brain_tumor_project/results/vgg16"
BATCH_SIZE = 32
EPOCHS = 15

os.makedirs(RESULTS_DIR, exist_ok=True)

# === Load Data ===
augment = get_augmentation_pipeline()
train_ds, val_ds, test_ds, class_names = create_datasets(DATA_DIR, BATCH_SIZE, img_size=(128,128), augment_fn=augment)

# === Build Model ===
model = build_vgg16_model(input_shape=(128,128,3))
model.summary()
with open(os.path.join(RESULTS_DIR, "model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# === Callbacks ===
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(RESULTS_DIR, "best_model.h5"),
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
]

# === Train ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# === Save Training Plot ===
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('VGG16 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "training_plot.png"))
plt.close()

# === Save Metrics ===
final_metrics = {
    "train_accuracy": float(history.history['accuracy'][-1]),
    "val_accuracy": float(history.history['val_accuracy'][-1]),
    "val_loss": float(history.history['val_loss'][-1])
}
with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=4)

print("✅ Training complete. Best model saved to:", RESULTS_DIR)
