#This handles:
    #Mounting Google Drive
    #Loading dataset from posixpath import split from yes/ and no/ folders
    #Train/validation/test split (70/15/15)
    #TensorFlow dataset pipeline creation

# src/common/dataset_utils.py
import tensorflow as tf
import os
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.AUTOTUNE

def load_image_paths(data_dir: str):
    """
    Reads all image paths from 'yes' and 'no' subfolders.
    Args:
        data_dir (str): Path to the main dataset directory containing /yes and /no folders.
    Returns:
        filepaths (list): List of image paths.
        labels (list): Corresponding labels ('yes' or 'no').
        class_names (list): ['no', 'yes']
    """
    data_dir = pathlib.Path(data_dir)
    classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    filepaths, labels = [], []

    for cls in classes:
        for img_path in (data_dir / cls).glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                filepaths.append(str(img_path))
                labels.append(cls)

    print(f"✅ Found {len(filepaths)} images across {len(classes)} classes: {classes}")
    return filepaths, labels, classes


def create_splits(filepaths, labels, test_size=0.15, val_size=0.15, seed=42):
    """
    Splits filepaths & labels into train, validation, and test sets.
    """
    train_files, test_files, train_labels, test_labels = train_test_split(
        filepaths, labels, test_size=test_size, stratify=labels, random_state=seed)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=val_size/(1-test_size),
        stratify=train_labels, random_state=seed)

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)


def preprocess_image(path, label, img_size=(128, 128)):
    """
    Loads and preprocesses a single image.
    - Decodes JPEG/PNG
    - Resizes
    - Normalizes to [0,1]
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0
    return img, label


def make_dataset(filepaths, labels, class_names, batch_size=32, shuffle=True, augment_fn=None):
    """
    Builds a tf.data.Dataset for efficient GPU training.
    """
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    labels_idx = [class_to_index[l] for l in labels]

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels_idx))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=42)
    ds = ds.map(lambda p, l: preprocess_image(p, l), num_parallel_calls=AUTOTUNE)
    if augment_fn:
        ds = ds.map(lambda x, y: (augment_fn(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def create_datasets(data_dir, batch_size=32, img_size=(128, 128), augment_fn=None):
    """
    Master function to create train/val/test datasets ready for model training.
    """
    filepaths, labels, class_names = load_image_paths(data_dir)
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = create_splits(filepaths, labels)

    train_ds = make_dataset(train_files, train_labels, class_names, batch_size, shuffle=True, augment_fn=augment_fn)
    val_ds   = make_dataset(val_files, val_labels, class_names, batch_size, shuffle=False)
    test_ds  = make_dataset(test_files, test_labels, class_names, batch_size, shuffle=False)

    return train_ds, val_ds, test_ds, class_names
