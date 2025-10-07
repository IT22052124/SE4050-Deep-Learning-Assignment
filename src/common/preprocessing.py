# src/common/preprocessing.py
"""
Preprocessing utilities for image data augmentation and dataset creation.
This file provides helper functions for creating augmented datasets for training.
"""

import tensorflow as tf
from tensorflow.keras import layers
import os
import numpy as np

def get_augmentation_pipeline():
    """Returns a standard data augmentation pipeline for brain tumor images"""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

def create_augmented_dataset(data_dir, batch_size=32, img_size=(224, 224), 
                            shuffle=True, use_augmentation=True):
    """Creates an augmented dataset from a directory with class subdirectories"""
    
    # Basic rescaling for all datasets
    rescale = tf.keras.layers.Rescaling(1./255)
    
    # Create basic image generator with rescaling
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Create train dataset
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=shuffle
    )
    
    # Create validation dataset
    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    # Convert to TensorFlow datasets
    train_ds = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    
    val_ds = tf.data.Dataset.from_generator(
        lambda: val_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    
    # Add augmentation to training dataset if requested
    if use_augmentation:
        augmentation = get_augmentation_pipeline()
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Optimize both datasets
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, train_gen.class_indices

def count_samples_in_directory(directory):
    """Count the number of image files in each subdirectory"""
    counts = {}
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            image_files = [f for f in os.listdir(subdir_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            counts[subdir] = len(image_files)
    return counts

