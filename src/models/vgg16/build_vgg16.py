import tensorflow as tf
from tensorflow.keras import layers, models

def build_vgg16_model(input_shape=(128,128,3)):
    """
    Build transfer learning model using pretrained VGG16.
    """
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    base_model.trainable = False  # freeze base layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # binary classification
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
