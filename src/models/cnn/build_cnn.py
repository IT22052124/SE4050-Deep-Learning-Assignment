# src/models/cnn/build_cnn.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(224,224,3)):
    model = models.Sequential([
        # First Conv Block - keep image size relatively large at first
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(2,2),  # 112x112x32
        
        # Second Conv Block
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),  # 56x56x64
        
        # Third Conv Block
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),  # 28x28x128
        
        # Fourth Conv Block - Additional pooling to reduce dimensions further
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),  # 14x14x128
        
        # Fifth Conv Block - Further dimension reduction
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),  # 7x7x128
        
        # Flatten and Dense layers - Much smaller feature space now (6272 vs 100352)
        layers.Flatten(),
        layers.Dense(128, activation='relu'),  # Reduced from 256 to 128
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
