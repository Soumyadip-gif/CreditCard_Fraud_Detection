import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

X_train = pd.read_csv(
    "dataset/processed/X_train.csv"
)

X_test = pd.read_csv(
    "dataset/processed/X_test.csv"
)

y_train = pd.read_csv(
    "dataset/processed/y_train.csv"
)["is_fraud"]

y_test = pd.read_csv(
    "dataset/processed/y_test.csv"
)["is_fraud"]

print("\n========== LOGISTIC REGRESSION ==========")

print("\nTraining Shape:")
print(X_train.shape)

print("\nTest Shape:")
print(X_test.shape)

model = LogisticRegression(
    class_weight="balanced",
    solver="saga",
    max_iter=300,
    random_state=42
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

print("\n========== MODEL EVALUATION ==========")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print("\nROC-AUC:")
print(round(roc_auc, 4))

print("\nPR-AUC:")
print(round(pr_auc, 4))

joblib.dump(
    model,
    "model/logistic_regression.pkl"
)

print("\nModel saved successfully.")

print("\n========================================")
print("Logistic Regression Training Completed")
print("========================================")