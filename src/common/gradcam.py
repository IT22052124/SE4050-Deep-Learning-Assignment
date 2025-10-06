# src/common/gradcam.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_gradcam(model, dataset, save_dir, class_names, num_images=5):
    os.makedirs(save_dir, exist_ok=True)
    # Try to find the last conv layer for Grad-CAM
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if last_conv is None:
        # Fallback: try index -3 as in the original code
        target = model.get_layer(index=-3).output
    else:
        target = model.get_layer(last_conv).output

    grad_model = tf.keras.models.Model([model.inputs], [target, model.output])

    for images, labels in dataset.take(1):
        for i in range(num_images):
            img = images[i]
            label = labels[i].numpy()
            img_array = tf.expand_dims(img, axis=0)

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                loss = predictions[:, 0]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]

            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= tf.reduce_max(heatmap)
            heatmap = heatmap.numpy()

            # Overlay heatmap
            plt.imshow(img)
            plt.imshow(heatmap, cmap='jet', alpha=0.4)
            plt.axis('off')
            plt.title(f"Grad-CAM: Actual={class_names[int(label)]}")
            plt.savefig(os.path.join(save_dir, f"gradcam_{i}.png"))
            plt.close()
