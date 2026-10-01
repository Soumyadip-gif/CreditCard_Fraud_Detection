import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
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

y_probability = model.predict_proba(
    X_test
)[:, 1]

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

results = []

for threshold in thresholds:

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append(
        {
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    )

results_df = pd.DataFrame(results)

print("\n========== THRESHOLD ANALYSIS ==========")

print(
    results_df.to_string(
        index=False
    )
)

best_row = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n========== BEST F1 THRESHOLD ==========")

print(
    best_row.to_string()
)

print("\n========================================")
print("Threshold Analysis Completed")
print("========================================")