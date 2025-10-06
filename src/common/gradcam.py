# src/common/gradcam.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_gradcam(model, dataset, save_dir, class_names, num_images=5):
    os.makedirs(save_dir, exist_ok=True)

    # Grab one batch and ensure the model is built by a forward pass
    batch = None
    for batch in dataset.take(1):
        pass
    if batch is None:
        print("Grad-CAM: dataset is empty, skipping.")
        return

    images, labels = batch
    _ = model(images[:1], training=False)  # build model so model.input/output are defined

    # Find last Conv2D layer in the built model
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        print("Grad-CAM: No Conv2D layer found, skipping.")
        return

    target = last_conv_layer.output
    grad_model = tf.keras.models.Model(inputs=[model.input], outputs=[target, model.output])

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
