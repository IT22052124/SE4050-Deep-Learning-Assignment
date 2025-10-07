# SE4050-Deep-Learning-Assignment

# Dataset Information

Dataset used: [Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

This dataset was downloaded from Kaggle and stored locally in Google Drive.
It contains MRI images classified into “yes” (tumor) and “no” (no tumor),
plus an additional folder `Br35H-Mask-RCNN` with segmentation masks.

Due to size and licensing restrictions, the dataset is **not uploaded to this GitHub repository**.

If you wish to reproduce the results:

1. Download the dataset from Kaggle.
2. Place it in your Google Drive under:

## Run on Google Colab

Follow these steps to train and evaluate the CNN using your dataset stored in Google Drive (e.g., `MyDrive/BrainTumor/yes` and `MyDrive/BrainTumor/no`). The scripts were updated to accept CLI flags or environment variables so you can point to any folder with class subfolders.

Prerequisites:

- In Colab, set Runtime > Change runtime type > Hardware accelerator: GPU (recommended).

1. Mount Google Drive and clone the repo

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!git clone https://github.com/IT22052124/SE4050-Deep-Learning-Assignment.git
%cd SE4050-Deep-Learning-Assignment
```

2. Install dependencies

```python
!pip install -U pip
!pip install -r requirements.txt

import tensorflow as tf, platform
print('TF version:', tf.__version__, 'Python:', platform.python_version())
print('GPUs:', tf.config.list_physical_devices('GPU'))
```

3. Train

```python
DATA_DIR = '/content/drive/MyDrive/BrainTumor'   # folder containing class subfolders (yes/no)
RESULTS_DIR = '/content/drive/MyDrive/brain_tumor_project/results/cnn'

!python -m src.models.cnn.train_cnn --data_dir "$DATA_DIR" --results_dir "$RESULTS_DIR" --epochs 10 --batch_size 32 --img_size 224 224
```

4. Evaluate

```python
!python -m src.models.cnn.evaluate_cnn --data_dir "$DATA_DIR" --results_dir "$RESULTS_DIR" --batch_size 32 --img_size 224 224
```

Outputs will be written to `RESULTS_DIR`:

- `best_model.h5` (best checkpoint)
- `history.png` (accuracy curves)
- `confusion_matrix.png`, `classification_report.txt`
- `gradcam/` (sample Grad-CAM visualizations)

Notes:

- The scripts split the data (70/15/15) with a fixed seed for reproducibility. You only need to provide a single folder containing class subfolders.
- If your dataset path differs, adjust `DATA_DIR` accordingly.
- If TensorFlow 2.16.0 isn’t available in Colab, you can try `pip install tensorflow==2.15.*` and re-run.


# Let's verify the existence of raw data directories first
import os
from pathlib import Path
import cv2
import random
import shutil
from tqdm.notebook import tqdm

print("Checking raw data directories...")
print(f"Looking for raw data in: {RAW_DATA_DIR}")

# First verify the raw data exists and list what's available
if os.path.exists(RAW_DATA_DIR):
    dirs = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    print(f"Found directories: {dirs}")
    
    # Check for yes/no directories specifically
    yes_dir = os.path.join(RAW_DATA_DIR, "yes")
    no_dir = os.path.join(RAW_DATA_DIR, "no")
    
    yes_exists = os.path.exists(yes_dir)
    no_exists = os.path.exists(no_dir)
    
    print(f"'yes' directory exists: {yes_exists}")
    print(f"'no' directory exists: {no_exists}")
    
    # Count files in each directory
    if yes_exists:
        yes_files = [f for f in os.listdir(yes_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(yes_files)} image files in 'yes' directory")
        if yes_files:
            print(f"Sample files: {yes_files[:3]}")
    
    if no_exists:
        no_files = [f for f in os.listdir(no_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(no_files)} image files in 'no' directory")
        if no_files:
            print(f"Sample files: {no_files[:3]}")
            
    # If the directories don't exist, search for alternatives
    if not (yes_exists and no_exists):
        print("\nSearching for alternative image directories...")
        all_subdirs = []
        for root, dirs, _ in os.walk(RAW_DATA_DIR):
            for d in dirs:
                subdir_path = os.path.join(root, d)
                img_count = len([f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                if img_count > 0:
                    all_subdirs.append((subdir_path, img_count))
        
        if all_subdirs:
            print("Found alternative directories with images:")
            for path, count in all_subdirs:
                print(f"  - {path}: {count} images")
        else:
            print("No alternative directories with images found.")
else:
    print(f"❌ Error: Raw data directory {RAW_DATA_DIR} does not exist!")
    print("Please make sure your Google Drive contains the correct folder structure.")
    
# Now let's define our preprocessing functions
# Constants
RANDOM_SEED = 42
IMG_SIZE = (224, 224)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

def create_dirs(base_path, classes):
    """Create directory structure for processed data"""
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(base_path, split, cls), exist_ok=True)
            
def split_and_copy(source_root, dest_root, classes):
    """Split and copy files into train/val/test directories"""
    random.seed(RANDOM_SEED)
    create_dirs(dest_root, classes)
    
    # Track total files processed
    total_processed = 0
    
    for cls in classes:
        cls_dir = os.path.join(source_root, cls)
        if not os.path.exists(cls_dir):
            print(f"Warning: Class directory {cls_dir} not found.")
            continue
            
        # Get all image files
        imgs = []
        for ext in ['.jpg', '.jpeg', '.png']:
            imgs.extend([os.path.join(cls_dir, f) for f in os.listdir(cls_dir) 
                         if f.lower().endswith(ext)])
            
        if not imgs:
            print(f"No images found in {cls_dir}")
            continue
            
        print(f"Found {len(imgs)} images in {cls_dir}")
        random.shuffle(imgs)
        n = len(imgs)
        n_val, n_test = int(n * VAL_SPLIT), int(n * TEST_SPLIT)
        n_train = n - n_val - n_test
        
        splits = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train+n_val],
            "test": imgs[n_train+n_val:]
        }
        
        for split, files in splits.items():
            print(f"Processing {cls} -> {split}: {len(files)} images")
            for img_path in tqdm(files):
                try:
                    # Read and resize image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read {img_path}")
                        continue
                    
                    # Resize image
                    resized = cv2.resize(img, IMG_SIZE)
                    
                    # Save to destination
                    out_path = os.path.join(dest_root, split, cls, os.path.basename(img_path))
                    cv2.imwrite(out_path, resized)
                    total_processed += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    return total_processed

print("\nStarting preprocessing...")
print(f"Reading raw images from: {RAW_DATA_DIR}")
print(f"Saving processed images to: {PROCESSED_DATA_DIR}")

# Process the data
if os.path.exists(RAW_DATA_DIR) and (os.path.exists(os.path.join(RAW_DATA_DIR, "yes")) or 
                                     os.path.exists(os.path.join(RAW_DATA_DIR, "no"))):
    total_files = split_and_copy(RAW_DATA_DIR, PROCESSED_DATA_DIR, ["yes", "no"])
    print(f"✅ Preprocessing completed successfully! Processed {total_files} images.")
else:
    print("⚠️ Could not find expected yes/no folders in the raw data directory.")
    print("Please make sure your Google Drive contains the correct folder structure,")
    print("or manually upload the brain tumor dataset with 'yes' and 'no' subfolders.")
    
# Count files in each split to verify
print("\nVerifying processed data:")
total_processed_files = 0
for split in ["train", "val", "test"]:
    split_dir = os.path.join(PROCESSED_DATA_DIR, split)
    if os.path.exists(split_dir):
        yes_path = os.path.join(split_dir, "yes")
        no_path = os.path.join(split_dir, "no")
        
        yes_count = len(os.listdir(yes_path)) if os.path.exists(yes_path) else 0
        no_count = len(os.listdir(no_path)) if os.path.exists(no_path) else 0
        split_total = yes_count + no_count
        total_processed_files += split_total
        
        print(f"{split.upper()}: yes={yes_count}, no={no_count}, total={split_total}")

print(f"Total processed images: {total_processed_files}")

# If no files were processed, show a warning
if total_processed_files == 0:
    print("\n⚠️ WARNING: No images were processed. Training will likely fail.")
    print("Please check your Google Drive folder structure and raw data.")