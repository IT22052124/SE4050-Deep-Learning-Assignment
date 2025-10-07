"""
Preprocessing script for brain tumor classification dataset.

This script handles:
1. Checking raw data directories
2. Splitting data into train/val/test sets
3. Resizing images to a standard size
4. Verifying the processed dataset structure

Usage:
- Call main() with optional raw_data_dir and processed_data_dir parameters
- The script will automatically check if preprocessing is needed
"""

import os
import random
import cv2
from tqdm import tqdm

# ==== CONFIG ====
RANDOM_SEED = 42
IMG_SIZE = (224, 224)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Default paths for Google Colab (these can be overridden when calling functions)
DRIVE_ROOT = "/content/drive/MyDrive"
BRAIN_TUMOR_DIR = f"{DRIVE_ROOT}/BrainTumor"
DATA_DIR = f"{BRAIN_TUMOR_DIR}/data"
RAW_DATA_DIR = f"{DATA_DIR}/archive"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"

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
            for img_path in tqdm(files, desc=f"{cls}->{split}", leave=False):
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

def verify_dataset(processed_dir):
    """Verify processed dataset structure and count images"""
    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")
    test_dir = os.path.join(processed_dir, "test")
    
    processed_exists = (os.path.exists(train_dir) and 
                       os.path.exists(val_dir) and 
                       os.path.exists(test_dir))
    
    if processed_exists:
        print("\nProcessed data exists. Checking counts...")
        train_yes = len(os.listdir(os.path.join(train_dir, "yes"))) if os.path.exists(os.path.join(train_dir, "yes")) else 0
        train_no = len(os.listdir(os.path.join(train_dir, "no"))) if os.path.exists(os.path.join(train_dir, "no")) else 0
        val_yes = len(os.listdir(os.path.join(val_dir, "yes"))) if os.path.exists(os.path.join(val_dir, "yes")) else 0
        val_no = len(os.listdir(os.path.join(val_dir, "no"))) if os.path.exists(os.path.join(val_dir, "no")) else 0
        test_yes = len(os.listdir(os.path.join(test_dir, "yes"))) if os.path.exists(os.path.join(test_dir, "yes")) else 0
        test_no = len(os.listdir(os.path.join(test_dir, "no"))) if os.path.exists(os.path.join(test_dir, "no")) else 0
        
        print(f"Train: {train_yes} yes, {train_no} no")
        print(f"Validation: {val_yes} yes, {val_no} no")
        print(f"Test: {test_yes} yes, {test_no} no")
        
        return processed_exists, train_yes > 0 and train_no > 0
    
    return False, False

def main(raw_data_dir=None, processed_data_dir=None):
    """Main preprocessing function to split data into train/val/test"""
    # Use provided paths or defaults
    raw_data_dir = raw_data_dir or RAW_DATA_DIR
    processed_data_dir = processed_data_dir or PROCESSED_DATA_DIR
    
    print("Checking raw data directories...")
    print(f"Looking for raw data in: {raw_data_dir}")
    
    # First verify the raw data exists and list what's available
    if os.path.exists(raw_data_dir):
        dirs = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]
        print(f"Found directories: {dirs}")
        
        # Check for yes/no directories specifically
        yes_dir = os.path.join(raw_data_dir, "yes")
        no_dir = os.path.join(raw_data_dir, "no")
        
        yes_exists = os.path.exists(yes_dir)
        no_exists = os.path.exists(no_dir)
        
        print(f"'yes' directory exists: {yes_exists}")
        print(f"'no' directory exists: {no_exists}")
        
        # Count files in each directory
        if yes_exists:
            yes_files = [f for f in os.listdir(yes_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(yes_files)} image files in 'yes' directory")
        
        if no_exists:
            no_files = [f for f in os.listdir(no_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(no_files)} image files in 'no' directory")
        
        # Check if preprocessing is needed
        processed_exists, class_folders_valid = verify_dataset(processed_data_dir)
        
        if processed_exists and class_folders_valid:
            print("\nProcessed data already exists with valid class folders. Skipping preprocessing.")
            return True
        else:
            print("\nStarting preprocessing...")
            print(f"Reading raw images from: {raw_data_dir}")
            print(f"Saving processed images to: {processed_data_dir}")
            
            # Process the data
            if yes_exists and no_exists:
                total_files = split_and_copy(raw_data_dir, processed_data_dir, ["yes", "no"])
                print(f"✅ Preprocessing completed successfully! Processed {total_files} images.")
                return True
            else:
                print("⚠️ Could not find expected yes/no folders in the raw data directory.")
                print("Please make sure your data contains the correct folder structure.")
                return False
    else:
        print(f"❌ Error: Raw data directory {raw_data_dir} does not exist!")
        return False

if __name__ == "__main__":
    main()