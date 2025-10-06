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
    return (tr_files, tr_labels), (va_files, va_labels), (te_files, te_labels)

def preprocess_image(path, label, img_size=(224,224)):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)/255.0
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
