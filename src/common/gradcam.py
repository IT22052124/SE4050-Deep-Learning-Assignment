# src/common/gradcam.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_gradcam(model, dataset, save_dir, class_names, num_images=5):
    os.makedirs(save_dir, exist_ok=True)

    # Grab one batch to infer image size and to visualize
    batch = None
    for batch in dataset.take(1):
        pass
    if batch is None:
        print("Grad-CAM: dataset is empty, skipping.")
        return

    images, labels = batch
    h, w = images.shape[1], images.shape[2]
    if h is None or w is None:
        # Fallback to common default if static dims not available
        h, w = 224, 224

    # Build a functional graph using a fresh Input bound to (h,w,3)
    inputs = tf.keras.Input(shape=(int(h), int(w), 3))
    outputs = model(inputs)

    # Find last Conv2D layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        print("Grad-CAM: No Conv2D layer found, skipping.")
        return

    target = last_conv_layer.output
    grad_model = tf.keras.models.Model(inputs=[inputs], outputs=[target, outputs])

    # Generate heatmaps for up to num_images from this batch
    num = min(num_images, images.shape[0])
    for i in range(num):
        img = images[i]
        label = int(labels[i].numpy())
        img_array = tf.expand_dims(img, axis=0)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            # Binary case: predictions shape (B,1)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        denom = tf.reduce_max(heatmap) + 1e-8
        heatmap = heatmap / denom
        heatmap = heatmap.numpy()

        plt.imshow(img)
        plt.imshow(heatmap, cmap='jet', alpha=0.4)
        plt.axis('off')
        plt.title(f"Grad-CAM: Actual={class_names[label]}")
        plt.savefig(os.path.join(save_dir, f"gradcam_{i}.png"))
        plt.close()
