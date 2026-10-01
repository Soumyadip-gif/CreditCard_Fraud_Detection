import pandas as pd
import numpy as np
import joblib


print("\n========== RAW PIPELINE TEST ==========")


# ==============================
# Load Model Components
# ==============================

encoder = joblib.load(
    "model/catboost_encoder.pkl"
)

model = joblib.load(
    "model/final_fraud_model.pkl"
)

threshold = joblib.load(
    "model/fraud_threshold.pkl"
)


# ==============================
# Load Raw Test Data
# ==============================

test = pd.read_csv(
    "dataset/test.csv"
)


# Select one raw transaction
sample = test.iloc[[0]].copy()

actual_label = sample["is_fraud"].iloc[0]


print("\nRaw Transaction Loaded.")
print("Actual Label:", actual_label)


# ==============================
# Remove Unnecessary Columns
# ==============================

columns_to_drop = [
    "Unnamed: 0",
    "first",
    "last",
    "street",
    "trans_num"
]

sample = sample.drop(
    columns=columns_to_drop
)


# ==============================
# Date Features
# ==============================

sample["trans_date_trans_time"] = pd.to_datetime(
    sample["trans_date_trans_time"]
)

sample["transaction_hour"] = (
    sample["trans_date_trans_time"].dt.hour
)

sample["transaction_day"] = (
    sample["trans_date_trans_time"].dt.day
)

sample["transaction_month"] = (
    sample["trans_date_trans_time"].dt.month
)

sample["transaction_day_of_week"] = (
    sample["trans_date_trans_time"].dt.dayofweek
)

sample["is_weekend"] = (
    sample["transaction_day_of_week"] >= 5
).astype(int)


# ==============================
# Customer Age
# ==============================

sample["dob"] = pd.to_datetime(
    sample["dob"]
)

sample["customer_age"] = (
    (
        sample["trans_date_trans_time"]
        - sample["dob"]
    ).dt.days / 365.25
)

sample["customer_age"] = (
    sample["customer_age"].round(1)
)


# ==============================
# Haversine Distance
# ==============================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return R * c


sample["customer_merchant_distance"] = (
    haversine_distance(
        sample["lat"],
        sample["long"],
        sample["merch_lat"],
        sample["merch_long"]
    )
)


# ==============================
# Remove Raw Date Columns
# ==============================

sample = sample.drop(
    columns=[
        "trans_date_trans_time",
        "dob"
    ]
)


# ==============================
# Remove Target + Card Number
# ==============================

sample = sample.drop(
    columns=[
        "is_fraud",
        "cc_num"
    ]
)


# ==============================
# Categorical Encoding
# ==============================

categorical_columns = [
    "merchant",
    "category",
    "gender",
    "city",
    "state",
    "job"
]

sample_encoded = encoder.transform(
    sample
)


# ==============================
# Prediction
# ==============================

probability = model.predict_proba(
    sample_encoded
)[0][1]

prediction = int(
    probability >= threshold
)


# ==============================
# Result
# ==============================

print("\n========== PREDICTION ==========")

print("Fraud Probability:")
print(round(probability, 6))

print("\nThreshold:")
print(threshold)

print("\nPrediction:")

if prediction == 1:
    print("FRAUD")
else:
    print("LEGITIMATE")

print("\nActual Label:")
print(actual_label)


print("\n================================")
print("Raw Pipeline Test Completed")
print("================================")