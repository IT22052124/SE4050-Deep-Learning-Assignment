# src/models/efficientnetb0/build_efficientnetb0.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

def build_efficientnetb0_model(input_shape=(224, 224, 3), num_classes=1, fine_tune_at=None):
    """
    Build an EfficientNetB0 model for binary classification.
    - input_shape: tuple, image size (224, 224, 3)
    - num_classes: 1 (sigmoid) for binary
    - fine_tune_at: optional layer index to unfreeze
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
