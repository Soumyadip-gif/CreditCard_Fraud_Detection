import pandas as pd
import numpy as np
import joblib
import os

import category_encoders as ce

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


print("\n========== FINAL PIPELINE TRAINING ==========")


# ==============================
# Load Dataset
# ==============================

train = pd.read_csv(
    "dataset/train.csv"
)

print("\nTrain Shape:")
print(train.shape)


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

train = train.drop(
    columns=columns_to_drop
)


# ==============================
# Date Features
# ==============================

train["trans_date_trans_time"] = pd.to_datetime(
    train["trans_date_trans_time"]
)

train["transaction_hour"] = (
    train["trans_date_trans_time"].dt.hour
)

train["transaction_day"] = (
    train["trans_date_trans_time"].dt.day
)

train["transaction_month"] = (
    train["trans_date_trans_time"].dt.month
)

train["transaction_day_of_week"] = (
    train["trans_date_trans_time"].dt.dayofweek
)

train["is_weekend"] = (
    train["transaction_day_of_week"] >= 5
).astype(int)


# ==============================
# Customer Age
# ==============================

train["dob"] = pd.to_datetime(
    train["dob"]
)

train["customer_age"] = (
    (
        train["trans_date_trans_time"]
        - train["dob"]
    ).dt.days / 365.25
)

train["customer_age"] = (
    train["customer_age"].round(1)
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


train["customer_merchant_distance"] = (
    haversine_distance(
        train["lat"],
        train["long"],
        train["merch_lat"],
        train["merch_long"]
    )
)


# ==============================
# Remove Raw Date Columns
# ==============================

train = train.drop(
    columns=[
        "trans_date_trans_time",
        "dob"
    ]
)


# ==============================
# Separate Target
# ==============================

target = "is_fraud"

X_train = train.drop(
    columns=[
        target,
        "cc_num"
    ]
)

y_train = train[target]


# ==============================
# Categorical Columns
# ==============================

categorical_columns = [
    "merchant",
    "category",
    "gender",
    "city",
    "state",
    "job"
]


# ==============================
# CatBoost Encoding
# ==============================

encoder = ce.CatBoostEncoder(
    cols=categorical_columns,
    random_state=42
)

X_train_encoded = encoder.fit_transform(
    X_train,
    y_train
)


print("\nEncoding completed.")


# ==============================
# Class Weights
# ==============================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# ==============================
# Final Model
# ==============================

model = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_iter=150,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=42
)


print("\nTraining final model...")


model.fit(
    X_train_encoded,
    y_train,
    sample_weight=sample_weights
)


print("Training completed.")


# ==============================
# Save Model + Encoder
# ==============================

os.makedirs(
    "model",
    exist_ok=True
)


joblib.dump(
    encoder,
    "model/catboost_encoder.pkl"
)

joblib.dump(
    model,
    "model/final_fraud_model.pkl"
)


# ==============================
# Save Threshold
# ==============================

threshold = 0.90

joblib.dump(
    threshold,
    "model/fraud_threshold.pkl"
)


print("\n========== FILES SAVED ==========")

print(
    "model/catboost_encoder.pkl"
)

print(
    "model/final_fraud_model.pkl"
)

print(
    "model/fraud_threshold.pkl"
)


print("\n=================================")
print("Final ML Pipeline Training Completed")
print("=================================")