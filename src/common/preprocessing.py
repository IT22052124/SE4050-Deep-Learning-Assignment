# src/common/preprocessing.py
import tensorflow as tf
from tensorflow.keras import layers

def get_augmentation_pipeline():
    """
    Shared augmentation pipeline for all models.
    Lightweight augmentations suitable for MRI data.
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")
    return data_augmentation
