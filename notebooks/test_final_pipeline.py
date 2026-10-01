import pandas as pd
import joblib


print("\n========== FINAL PIPELINE TEST ==========")


# Load files
encoder = joblib.load(
    "model/catboost_encoder.pkl"
)

model = joblib.load(
    "model/final_fraud_model.pkl"
)

threshold = joblib.load(
    "model/fraud_threshold.pkl"
)


print("\nEncoder loaded successfully.")
print("Model loaded successfully.")
print("Threshold:", threshold)


# Load processed test data
X_test = pd.read_csv(
    "dataset/processed/X_test.csv"
)

y_test = pd.read_csv(
    "dataset/processed/y_test.csv"
)["is_fraud"]


# Select one transaction
sample = X_test.iloc[[0]]


# Make prediction
probability = model.predict_proba(
    sample
)[0][1]


prediction = int(
    probability >= threshold
)


print("\n========== SAMPLE PREDICTION ==========")

print("Fraud Probability:")
print(round(probability, 6))

print("\nThreshold:")
print(threshold)

print("\nPrediction:")
print(prediction)

if prediction == 1:
    print("Result: FRAUD")
else:
    print("Result: LEGITIMATE")


print("\nActual Label:")
print(y_test.iloc[0])


print("\n========================================")
print("Final Pipeline Verification Completed")
print("========================================")