# This handles:
    # Mounting Google Drive (from Colab notebook, if used)
    # Loading dataset from folders (e.g., yes/ and no/)
    # Train/validation/test split (70/15/15)
    # TensorFlow dataset pipeline creation

# src/common/dataset_utils.py
import tensorflow as tf, pathlib
from sklearn.model_selection import train_test_split
AUTOTUNE = tf.data.AUTOTUNE

def load_image_paths(data_dir, allowed_classes=None):
    """Scan a directory and collect image paths and labels.

    Parameters:
        data_dir (str | Path): Root folder containing class subfolders.
        allowed_classes (list[str] | None): If provided, only include these class folders (in this order).

    Returns:
        (filepaths, labels, classes) where classes is the list of class names in the label mapping order.
    """
    data_dir = pathlib.Path(data_dir)
    all_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    if allowed_classes:
        classes = [c for c in allowed_classes if (data_dir / c).is_dir()]
    else:
        classes = sorted([p.name for p in all_dirs])

    filepaths, labels = [], []
    for cls in classes:
        cls_dir = data_dir / cls
        for img_path in cls_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                filepaths.append(str(img_path))
                labels.append(cls)
    print(f"✅ Found {len(filepaths)} images in {classes}")
    return filepaths, labels, classes

def create_splits(filepaths, labels, test_size=0.15, val_size=0.15, seed=42):
    t_files, te_files, t_labels, te_labels = train_test_split(filepaths, labels, test_size=test_size, stratify=labels, random_state=seed)
    tr_files, va_files, tr_labels, va_labels = train_test_split(t_files, t_labels, test_size=val_size/(1-test_size), stratify=t_labels, random_state=seed)
    # Quick diagnostics
    def counts(lbls):
        from collections import Counter
        return dict(Counter(lbls))
    print("Split sizes:", {
        'train': len(tr_files), 'val': len(va_files), 'test': len(te_files)
    })
    print("Train label counts:", counts(tr_labels))
    print("Val label counts:", counts(va_labels))
    print("Test label counts:", counts(te_labels))
    return (tr_files, tr_labels), (va_files, va_labels), (te_files, te_labels)

def preprocess_image(path, label, img_size=(224,224)):
    img_bytes = tf.io.read_file(path)
    # decode_image supports PNG/JPEG, sets dynamic shape; we fix shape after resize
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    # Ensure static channel dimension for downstream layers
    img.set_shape([img_size[0], img_size[1], 3])
    return img, label

def make_dataset(files, labels, class_names, batch_size=32, shuffle=True, augment_fn=None):
    class_to_idx = {c:i for i,c in enumerate(class_names)}
    labels_idx = [class_to_idx[l] for l in labels]
    ds = tf.data.Dataset.from_tensor_slices((files, labels_idx))
    if shuffle: ds = ds.shuffle(len(files), seed=42)
    ds = ds.map(lambda p,l: preprocess_image(p,l), num_parallel_calls=AUTOTUNE)
    if augment_fn: ds = ds.map(lambda x,y:(augment_fn(x),y), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)

def create_datasets(data_dir, batch_size=32, img_size=(224,224), augment_fn=None, allowed_classes=None):
    f, l, c = load_image_paths(data_dir, allowed_classes=allowed_classes)
    (trf, trl), (vaf, val), (tef, tel) = create_splits(f, l)
    # Compute training label counts for class weighting
    class_to_idx = {name: i for i, name in enumerate(c)}
    tr_idx = [class_to_idx[name] for name in trl]
    counts = [0] * len(c)
    for i in tr_idx:
        counts[i] += 1
    
    tr = make_dataset(trf, trl, c, batch_size, True, augment_fn)
    va = make_dataset(vaf, val, c, batch_size, False)
    te = make_dataset(tef, tel, c, batch_size, False)
    return tr, va, te, c, counts


def create_datasets_from_splits(data_dir, batch_size=32, img_size=(224,224), augment_fn=None, allowed_classes=None):
    """
    Load datasets from pre-split train/val/test directories.
    
    Parameters:
        data_dir (str | Path): Root folder containing train/, val/, test/ subdirectories
        batch_size (int): Batch size for datasets
        img_size (tuple): Target image size (height, width)
        augment_fn: Augmentation function to apply to training data
        allowed_classes (list[str] | None): If provided, only include these class folders
    
    Returns:
        (train_ds, val_ds, test_ds, class_names, class_counts)
    """
    import pathlib
    data_dir = pathlib.Path(data_dir)
    
    # Load train data
    train_dir = data_dir / 'train'
    train_f, train_l, class_names = load_image_paths(train_dir, allowed_classes)
    
    # Load val data
    val_dir = data_dir / 'val'
    val_f, val_l, _ = load_image_paths(val_dir, allowed_classes)
    
    # Load test data
    test_dir = data_dir / 'test'
    test_f, test_l, _ = load_image_paths(test_dir, allowed_classes)
    
    # Compute training label counts for class weighting
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    train_idx = [class_to_idx[name] for name in train_l]
    counts = [0] * len(class_names)
    for i in train_idx:
        counts[i] += 1
    
    print(f"📊 Dataset splits loaded:")
    print(f"   Train: {len(train_f)} images")
    print(f"   Val: {len(val_f)} images")
    print(f"   Test: {len(test_f)} images")
    print(f"   Classes: {class_names}")
    print(f"   Class counts in training: {dict(zip(class_names, counts))}")
    
    # Create datasets
    train_ds = make_dataset(train_f, train_l, class_names, batch_size, True, augment_fn)
    val_ds = make_dataset(val_f, val_l, class_names, batch_size, False, None)
    test_ds = make_dataset(test_f, test_l, class_names, batch_size, False, None)
    
    return train_ds, val_ds, test_ds, class_names, counts

