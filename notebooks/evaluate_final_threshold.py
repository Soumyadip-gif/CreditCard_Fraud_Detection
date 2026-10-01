import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

X_test = pd.read_csv(
    "dataset/processed/X_test.csv"
)

y_test = pd.read_csv(
    "dataset/processed/y_test.csv"
)["is_fraud"]

model = joblib.load(
    "model/hist_gradient_boosting.pkl"
)

threshold = 0.90

y_probability = model.predict_proba(
    X_test
)[:, 1]

y_pred = (
    y_probability >= threshold
).astype(int)

print("\n========== FINAL THRESHOLD EVALUATION ==========")

print("\nThreshold:")
print(threshold)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nROC-AUC:")
print(
    round(
        roc_auc_score(
            y_test,
            y_probability
        ),
        4
    )
)

print("\nPR-AUC:")
print(
    round(
        average_precision_score(
            y_test,
            y_probability
        ),
        4
    )
)

print("\n==============================================")
print("Final Threshold Evaluation Completed")
print("==============================================")