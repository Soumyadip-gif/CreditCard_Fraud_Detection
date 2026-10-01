import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight

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

print("\n========== HISTOGRAM GRADIENT BOOSTING ==========")

print("\nTraining Shape:")
print(X_train.shape)

print("\nTest Shape:")
print(X_test.shape)

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

model = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_iter=150,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=42
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

print("Training completed.")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]

print("\n========== MODEL EVALUATION ==========")

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
    "model/hist_gradient_boosting.pkl"
)

print("\nModel saved successfully.")

print("\n================================================")
print("Histogram Gradient Boosting Training Completed")
print("================================================")