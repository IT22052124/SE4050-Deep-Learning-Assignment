# ======================================
# Member 3 - Random Forest Evaluation
# Saves results in reports/member3_rf/
# ======================================

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)

sns.set_style("whitegrid")

def main():
    member_name = "member3_rf"
    models_dir = f"models/{member_name}"
    reports_dir = f"reports/{member_name}"
    figures_dir = os.path.join(reports_dir, "figures")

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ---- load model & test data ----
    model_path = os.path.join(models_dir, "rf_best.joblib")
    test_path = "data/processed/test.csv"

    model = joblib.load(model_path)
    test = pd.read_csv(test_path)

    X_test = test.drop("Outcome", axis=1)
    y_test = test["Outcome"]

    # ---- predictions ----
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # ---- metrics ----
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    print(pd.DataFrame(report).transpose())
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Average Precision (PR AUC): {ap:.4f}")

    # ---- confusion matrix ----
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Member 3 (Random Forest)")
    plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"))
    plt.close()

    # ---- ROC curve ----
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Member 3 (Random Forest)")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "roc_curve.png"))
    plt.close()

    # ---- PR curve ----
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Member 3 (Random Forest)")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "pr_curve.png"))
    plt.close()

    # ---- feature importances ----
    importances = model.feature_importances_
    features = X_test.columns
    indices = np.argsort(importances)[::-1][:10]
    plt.figure(figsize=(6, 4))
    plt.barh(features[indices][::-1], importances[indices][::-1])
    plt.title("Top 10 Feature Importances - Member 3 (Random Forest)")
    plt.savefig(os.path.join(figures_dir, "feature_importances.png"))
    plt.close()

    # ---- save summary ----
    summary = {
        "roc_auc": float(auc),
        "average_precision": float(ap)
    }
    pd.DataFrame([summary]).to_csv(os.path.join(reports_dir, "results_table.csv"), index=False)

    print("Evaluation complete!")
    print(f"Saved all plots to {figures_dir}")
    print(f"Saved metrics to {reports_dir}")

if __name__ == "__main__":
    main()
