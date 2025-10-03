# src/train_rf.py
import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42

def main(train_path="data/processed/train.csv", val_path="data/processed/val.csv",
         models_dir="models", reports_dir="reports"):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    X_train = train.drop("Outcome", axis=1)
    y_train = train["Outcome"]
    X_val = val.drop("Outcome", axis=1)
    y_val = val["Outcome"]

    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)

    param_dist = {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.2, 0.5],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rnd = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=50, scoring='roc_auc',
                             n_jobs=-1, cv=cv, random_state=SEED, verbose=2)

    rnd.fit(X_train, y_train)

    best = rnd.best_estimator_
    y_val_proba = best.predict_proba(X_val)[:,1]
    val_auc = roc_auc_score(y_val, y_val_proba)

    joblib.dump(best, os.path.join(models_dir, "rf_best.joblib"))

    results = {
        "best_params": rnd.best_params_,
        "val_roc_auc": float(val_auc)
    }
    with open(os.path.join(reports_dir, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # save cv results for inspection
    cv_df = pd.DataFrame(rnd.cv_results_)
    cv_df.to_csv(os.path.join(reports_dir, "cv_results.csv"), index=False)

    print("Training finished. Best params saved and validation AUC:", val_auc)

if __name__ == "__main__":
    main()
