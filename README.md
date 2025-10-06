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
