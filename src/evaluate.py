# src/evaluate.py
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score)

sns.set_style("whitegrid")

def main(test_path="data/processed/test.csv", models_dir="models", reports_dir="reports"):
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.join(reports_dir, "figures"), exist_ok=True)

    model = joblib.load(os.path.join(models_dir, "rf_best.joblib"))
    test = pd.read_csv(test_path)

    X_test = test.drop("Outcome", axis=1)
    y_test = test["Outcome"]

    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics
    print("Classification Report:\n", classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print("ROC AUC:", auc)
    print("Average Precision (PR AUC):", ap)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(reports_dir, "figures", "confusion_matrix.png"))
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--', label='random')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
    plt.savefig(os.path.join(reports_dir, "figures", "roc_curve.png"))
    plt.close()

    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend()
    plt.savefig(os.path.join(reports_dir, "figures", "pr_curve.png"))
    plt.close()

    # Feature importances
    importances = model.feature_importances_
    feat = X_test.columns
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(6,4))
    plt.barh(feat[indices][:10][::-1], importances[indices][:10][::-1])
    plt.title("Top Feature Importances")
    plt.savefig(os.path.join(reports_dir, "figures", "feature_importances.png"))
    plt.close()

    # Save summary table
    summary = {
        "roc_auc": float(auc),
        "average_precision": float(ap)
    }
    pd.DataFrame([summary]).to_csv(os.path.join(reports_dir, "results_table.csv"), index=False)
    print("Evaluation saved to", reports_dir)

if __name__ == "__main__":
    main()
