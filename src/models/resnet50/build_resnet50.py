# src/models/resnet50/build_resnet50.py
"""
ResNet50 Transfer Learning Model Builder

This module builds a ResNet50-based transfer learning model for brain tumor classification.
ResNet50 is pre-trained on ImageNet and fine-tuned for medical image classification.

Key Features:
- Pre-trained ResNet50 backbone (frozen initially)
- Custom classification head
- Support for both binary and multi-class classification
- Layer freezing/unfreezing for fine-tuning
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


def build_resnet50_model(input_shape=(224, 224, 3), num_classes=4, freeze_base=True):
    """
    Build ResNet50 transfer learning model for brain tumor classification.
    
    Parameters:
        input_shape (tuple): Input image shape (height, width, channels)
        num_classes (int): Number of output classes
            - 2 for binary (tumor/no tumor)
            - 4 for multi-class (glioma, meningioma, pituitary, no tumor)
        freeze_base (bool): Whether to freeze ResNet50 base layers initially
        
    Returns:
        model: Compiled Keras model ready for training
    """
    
    # Load pre-trained ResNet50 without top classification layer
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling=None  # We'll add our own pooling
    )
    
    # Freeze base model layers for transfer learning
    if freeze_base:
        base_model.trainable = False
        print(f"✅ ResNet50 base frozen with {len(base_model.layers)} layers")
    else:
        base_model.trainable = True
        print(f"✅ ResNet50 base trainable with {len(base_model.layers)} layers")
    
    # Build the model
    inputs = layers.Input(shape=input_shape, name='input_layer')
    
    # Preprocessing for ResNet50 (expects inputs in range [0, 255])
    # Our images are normalized to [0, 1], so scale back
    x = layers.Rescaling(scale=255.0, name='rescale_for_resnet')(inputs)
    
    # Apply ResNet50 preprocessing
    x = tf.keras.applications.resnet50.preprocess_input(x)
    
    # ResNet50 backbone
    x = base_model(x, training=False)
    
    # Custom classification head
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    
    # Dense layers with dropout for regularization
    x = layers.Dense(512, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    
    x = layers.Dense(256, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    
    # Output layer
    if num_classes == 2:
        # Binary classification
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    else:
        # Multi-class classification
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()]
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet50_BrainTumor')
    
    # Compile model with Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss,
        metrics=metrics
    )
    
    return model, base_model


def unfreeze_base_model(model, base_model, num_layers_to_unfreeze=30):
    """
    Unfreeze the top layers of ResNet50 base for fine-tuning.
    
    Parameters:
        model: The full model
        base_model: The ResNet50 base model
        num_layers_to_unfreeze (int): Number of top layers to unfreeze
        
    Returns:
        model: Model with unfrozen layers, recompiled with lower learning rate
    """
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze all layers except the last num_layers_to_unfreeze
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - num_layers_to_unfreeze)
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True
    
    print(f"✅ Unfroze top {num_layers_to_unfreeze} layers of ResNet50")
    print(f"   Trainable layers: {sum(1 for l in base_model.layers if l.trainable)}/{total_layers}")
    
    # Recompile with lower learning rate for fine-tuning
    # Detect if binary or multi-class from output layer
    output_shape = model.output.shape[-1]
    if output_shape == 1:
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    else:
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
        loss=loss,
        metrics=metrics
    )
    
    return model


if __name__ == "__main__":
    # Test model building
    print("Testing ResNet50 model building...")
    
    # Test binary classification
    print("\n=== Binary Classification Model ===")
    model_binary, base = build_resnet50_model(num_classes=2)
    model_binary.summary()
    
    # Test multi-class classification
    print("\n=== Multi-class Classification Model ===")
    model_multi, base = build_resnet50_model(num_classes=4)
    model_multi.summary()
    
    print("\n✅ Model building tests passed!")
