import os, shutil, random, cv2
from pathlib import Path
from tqdm import tqdm

# ==== CONFIG ====
RANDOM_SEED = 42
IMG_SIZE = (224, 224)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

drive_root = "/content/drive/MyDrive/brain_tumor_project"
raw_root = Path(drive_root) / "data/archive"
processed_root = Path(drive_root) / "data/processed"
mask_root = raw_root / "Br35H-Mask-RCNN"

def create_dirs(base_path, classes):
    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(base_path / split / cls, exist_ok=True)

def split_and_copy(source_root, dest_root, classes):
    random.seed(RANDOM_SEED)
    create_dirs(dest_root, classes)
    for cls in classes:
        imgs = list((source_root / cls).glob("*.jpg")) + list((source_root / cls).glob("*.png"))
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
            for img in tqdm(files, desc=f"{cls}->{split}", leave=False):
                out = dest_root / split / cls / img.name
                shutil.copy(img, out)

def apply_mask(image_path, mask_path):
    img = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or img is None: return None
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    return cv2.bitwise_and(img, img, mask=mask)

def create_masked_dataset(mask_root, dest_root):
    out_root = dest_root / "masked"
    for split in ["TRAIN", "VAL", "TEST"]:
        img_dir = mask_root / split / "images"
        mask_dir = mask_root / split / "masks"
        out_dir = out_root / split.lower()
        os.makedirs(out_dir, exist_ok=True)
        for img_file in tqdm(list(img_dir.glob("*.jpg")), desc=f"Masking {split}", leave=False):
            mask_file = mask_dir / img_file.name
            masked = apply_mask(img_file, mask_file)
            if masked is not None:
                resized = cv2.resize(masked, IMG_SIZE)
                cv2.imwrite(str(out_dir / img_file.name), resized)

def main():
    print("🔹 Splitting yes/no folders")
    split_and_copy(raw_root, processed_root, ["yes", "no"])
    print("🔹 Creating masked dataset")
    create_masked_dataset(mask_root, processed_root)
    print("✅ Preprocessing done")

if __name__ == "__main__":
    main()



    """
 Run once in Colab:

!python src/common/preprocess.py
"""