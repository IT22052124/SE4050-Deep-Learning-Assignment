# src/preprocess.py
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

RANDOM_STATE = 42

def main(infile="data/raw/diabetes.csv", outdir="data/processed", models_dir="models"):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(infile)
    zero_missing = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df[zero_missing] = df[zero_missing].replace(0, np.nan)

    # Split: train 70%, temp 30% -> then split temp into val/test 50/50 => each 15%
    train, temp = train_test_split(df, test_size=0.3, stratify=df['Outcome'], random_state=RANDOM_STATE)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['Outcome'], random_state=RANDOM_STATE)

    feature_cols = [c for c in df.columns if c != 'Outcome']

    preprocessing = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    X_train = preprocessing.fit_transform(train[feature_cols])
    X_val = preprocessing.transform(val[feature_cols])
    X_test = preprocessing.transform(test[feature_cols])

    train_proc = pd.DataFrame(X_train, columns=feature_cols)
    train_proc['Outcome'] = train['Outcome'].values

    val_proc = pd.DataFrame(X_val, columns=feature_cols)
    val_proc['Outcome'] = val['Outcome'].values

    test_proc = pd.DataFrame(X_test, columns=feature_cols)
    test_proc['Outcome'] = test['Outcome'].values

    train_proc.to_csv(os.path.join(outdir, "train.csv"), index=False)
    val_proc.to_csv(os.path.join(outdir, "val.csv"), index=False)
    test_proc.to_csv(os.path.join(outdir, "test.csv"), index=False)

    joblib.dump(preprocessing, os.path.join(models_dir, "preprocessing_pipeline.joblib"))
    print("Saved processed files to", outdir)
    print("Saved preprocessing pipeline to", models_dir)

if __name__ == "__main__":
    main()
