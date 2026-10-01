import pandas as pd
import numpy as np
import category_encoders as ce
import os

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

print("Original Train Shape:", train.shape)
print("Original Test Shape:", test.shape)

columns_to_drop = [
    "Unnamed: 0",
    "first",
    "last",
    "street",
    "trans_num"
]

train = train.drop(columns=columns_to_drop)
test = test.drop(columns=columns_to_drop)

train["trans_date_trans_time"] = pd.to_datetime(
    train["trans_date_trans_time"]
)

test["trans_date_trans_time"] = pd.to_datetime(
    test["trans_date_trans_time"]
)

for df in [train, test]:
    df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
    df["transaction_day"] = df["trans_date_trans_time"].dt.day
    df["transaction_month"] = df["trans_date_trans_time"].dt.month
    df["transaction_day_of_week"] = (
        df["trans_date_trans_time"].dt.dayofweek
    )
    df["is_weekend"] = (
        df["transaction_day_of_week"] >= 5
    ).astype(int)

for df in [train, test]:
    df["dob"] = pd.to_datetime(df["dob"])

    df["customer_age"] = (
        (
            df["trans_date_trans_time"] - df["dob"]
        ).dt.days / 365.25
    )

    df["customer_age"] = df["customer_age"].round(1)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return R * c

for df in [train, test]:
    df["customer_merchant_distance"] = haversine_distance(
        df["lat"],
        df["long"],
        df["merch_lat"],
        df["merch_long"]
    )

train = train.drop(
    columns=["trans_date_trans_time", "dob"]
)

test = test.drop(
    columns=["trans_date_trans_time", "dob"]
)

target = "is_fraud"

X_train = train.drop(columns=[target, "cc_num"])
y_train = train[target]

X_test = test.drop(columns=[target, "cc_num"])
y_test = test[target]

categorical_columns = [
    "merchant",
    "category",
    "gender",
    "city",
    "state",
    "job"
]

encoder = ce.CatBoostEncoder(
    cols=categorical_columns,
    random_state=42
)

X_train_encoded = encoder.fit_transform(
    X_train,
    y_train
)

X_test_encoded = encoder.transform(
    X_test
)

print("\n========== ENCODING COMPLETE ==========")

print("\nEncoded Train Shape:")
print(X_train_encoded.shape)

print("\nEncoded Test Shape:")
print(X_test_encoded.shape)

print("\nData Types:")
print(X_train_encoded.dtypes)

print("\nMissing Values:")
print(X_train_encoded.isnull().sum().sum())

print("\nSample Encoded Data:")
print(X_train_encoded.head())

print("\n========================================")
print("Categorical Encoding Completed")
print("========================================")

os.makedirs("dataset/processed", exist_ok=True)

X_train_encoded.to_csv(
    "dataset/processed/X_train.csv",
    index=False
)

X_test_encoded.to_csv(
    "dataset/processed/X_test.csv",
    index=False
)

y_train.to_csv(
    "dataset/processed/y_train.csv",
    index=False
)

y_test.to_csv(
    "dataset/processed/y_test.csv",
    index=False
)

print("\n========== PROCESSED DATA SAVED ==========")

print("X_train saved successfully")
print("X_test saved successfully")
print("y_train saved successfully")
print("y_test saved successfully")

print("\n==========================================")
print("Step 8 Completed Successfully")
print("==========================================")