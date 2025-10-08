import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

def build_vgg16_optimized(input_shape=(224,224,3)):
    """
    Optimized VGG16 transfer learning model for brain tumor classification.
    Uses fine-tuning approach for best performance while remaining efficient.
    """
    # Load pre-trained VGG16 model
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Fine-tuning: Freeze early layers, unfreeze last block for better feature learning
    base_model.trainable = True
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    # Build the complete model with optimized architecture
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # Lower LR for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Legacy functions for backward compatibility
def build_vgg16_basic(input_shape=(224,224,3)):
    """Legacy function - use build_vgg16_optimized instead"""
    return build_vgg16_optimized(input_shape)

def build_vgg16_fine_tuned(input_shape=(224,224,3)):
    """
    Fine-tuned VGG16 model - unfreeze last few layers
    """
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze early layers, unfreeze last block
    base_model.trainable = True
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  # Lower LR for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_vgg16_enhanced(input_shape=(224,224,3)):
    """
    Enhanced VGG16 model with additional regularization
    """
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.6),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Legacy function for backward compatibility
def build_vgg16_model(input_shape=(224,224,3)):
    """
    Legacy function - returns the basic VGG16 model
    """
    return build_vgg16_basic(input_shape)
