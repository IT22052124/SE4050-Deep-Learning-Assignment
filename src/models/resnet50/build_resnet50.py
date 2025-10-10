import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ResNet50


def build_resnet50_optimized(input_shape=(224, 224, 3)):
    """
    Optimized ResNet50 model matching VGG16's successful architecture.
    Uses fine-tuning approach with unfrozen last block for best performance.
    """
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Fine-tuning: Unfreeze last 4 layers (similar to VGG16's approach)
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

    # Use same learning rate as VGG16 for consistency
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def unfreeze_resnet50_for_finetuning(model, learning_rate=0.00001):
    """
    DEPRECATED: No longer needed as we now use single-stage fine-tuning.
    Kept for backward compatibility.
    """
    # Get the base model (first layer in Sequential)
    base_model = model.layers[0]
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze all layers except the last 30 (approximately last 2 residual blocks)
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Unfroze last 30 layers of ResNet50 for fine-tuning")
    print(f"   Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")
    
    return model


def build_resnet50_basic(input_shape=(224, 224, 3)):
    """
    Basic ResNet50 transfer learning model - frozen base layers
    """
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze base layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_resnet50_fine_tuned(input_shape=(224, 224, 3)):
    """
    Fine-tuned ResNet50 model - unfreeze last few layers
    """
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    base_model.trainable = True
    # Freeze early layers, unfreeze last few blocks
    for layer in base_model.layers[:-10]:
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_resnet50_enhanced(input_shape=(224, 224, 3)):
    """
    Enhanced ResNet50 model with extra dense layers and regularization
    """
    base_model = ResNet50(
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


# Legacy alias
def build_resnet50_model(input_shape=(224, 224, 3)):
    return build_resnet50_basic(input_shape)


# New recommended helpers
def build_resnet50_frozen_head(input_shape=(224, 224, 3),
                               dense_units=(512, 256),
                               dropout=(0.5, 0.3),
                               l2_reg=1e-4,
                               learning_rate=1e-4,
                               label_smoothing=0.05):
    """
    Build ResNet50 with ImageNet weights and a custom classification head.
    Base CNN is frozen for the first training stage.
    """
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout[i] if i < len(dropout) else dropout[-1])(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs, name='ResNet50_frozen_head')
    # Attach backbone reference for reliable fine-tuning later
    setattr(model, '_backbone', base_model)

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc')
        ],
    )
    return model


def fine_tune_resnet50(model, trainable_layers=50, learning_rate=1e-5, label_smoothing=0.0, freeze_batchnorm=True):
    """
    Unfreeze the top `trainable_layers` of the ResNet50 base for fine-tuning.
    Optionally keep BatchNorm layers frozen (recommended for small datasets).
    Recompiles the model with a lower LR.
    """
    # Prefer using the stored backbone reference
    base_model = getattr(model, '_backbone', None)

    if base_model is not None:
        base_model.trainable = True
        # Freeze all but the last `trainable_layers` of the backbone
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        if freeze_batchnorm:
            for layer in base_model.layers:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
    else:
        # Robust fallback: operate directly on model layers
        # Consider Conv2D and related backbone layers; skip Dense/Dropout/BN in the head
        backbone_layers = [l for l in model.layers if isinstance(l, (layers.Conv2D, layers.BatchNormalization))]
        # Unfreeze the last `trainable_layers` backbone layers
        for i, l in enumerate(backbone_layers):
            # Default freeze
            l.trainable = False
        for l in backbone_layers[-trainable_layers:]:
            # Unfreeze selected
            l.trainable = True
        if freeze_batchnorm:
            # Ensure BN stay frozen
            for l in backbone_layers:
                if isinstance(l, layers.BatchNormalization):
                    l.trainable = False

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc')
        ],
    )
    return model
