# First, let's verify that we have processed data to train on
import os

# Check if processed data exists with the right structure
train_dir = os.path.join(PROCESSED_DATA_DIR, "train")
val_dir = os.path.join(PROCESSED_DATA_DIR, "val")
test_dir = os.path.join(PROCESSED_DATA_DIR, "test")

processed_data_exists = (os.path.exists(train_dir) and 
                         os.path.exists(val_dir) and 
                         os.path.exists(test_dir))

# Check if we have class folders in the train directory
class_folders_exist = False
if processed_data_exists:
    train_yes = os.path.join(train_dir, "yes")
    train_no = os.path.join(train_dir, "no")
    
    class_folders_exist = (os.path.exists(train_yes) and 
                           os.path.exists(train_no))
    
    if class_folders_exist:
        yes_count = len(os.listdir(train_yes))
        no_count = len(os.listdir(train_no))
        
        print(f"Train directory has {yes_count} 'yes' images and {no_count} 'no' images")
        
        if yes_count == 0 or no_count == 0:
            class_folders_exist = False
            print("⚠️ One of the class folders is empty!")

# Choose the right directory for training
if processed_data_exists and class_folders_exist:
    # If processed data with correct structure exists, use it
    TRAIN_DATA_DIR = PROCESSED_DATA_DIR
    print(f"Using processed data for training: {TRAIN_DATA_DIR}")
    print("The script will automatically detect the train/val/test structure")
elif os.path.exists(os.path.join(RAW_DATA_DIR, "yes")) and os.path.exists(os.path.join(RAW_DATA_DIR, "no")):
    # If raw data exists with yes/no folders, use the original data structure
    TRAIN_DATA_DIR = RAW_DATA_DIR
    print(f"Using raw data for training: {TRAIN_DATA_DIR}")
    print("The script will use the original data structure with class folders")
else:
    # Fall back to the original DATA_DIR (parent of archive)
    TRAIN_DATA_DIR = DATA_DIR
    print(f"⚠️ Could not find properly structured data. Trying: {TRAIN_DATA_DIR}")
    print("Training may fail if the correct data structure is not found.")

# Train the model
print("\nTraining the CNN model...")
print(f"Using data from: {TRAIN_DATA_DIR}")
print(f"Saving results to: {RESULTS_DIR}")

# Use the enhanced train_cnn.py script which now handles both data structures
!python -m src.models.cnn.train_cnn \
    --data_dir {TRAIN_DATA_DIR} \
    --results_dir {RESULTS_DIR} \
    --epochs 30 \
    --batch_size 32 \
    --img_size 224 224 \
    --use_processed {1 if processed_data_exists and class_folders_exist else 0}