import pandas as pd
import requests


print("\n========== FLASK API TEST ==========")


# Load test transaction
test = pd.read_csv(
    "dataset/test.csv"
)

sample = test.iloc[0]


# Prepare transaction
data = {
    "trans_date_trans_time":
        sample["trans_date_trans_time"],

    "merchant":
        sample["merchant"],

    "category":
        sample["category"],

    "amt":
        float(sample["amt"]),

    "gender":
        sample["gender"],

    "city":
        sample["city"],

    "state":
        sample["state"],

    "zip":
        int(sample["zip"]),

    "lat":
        float(sample["lat"]),

    "long":
        float(sample["long"]),

    "city_pop":
        int(sample["city_pop"]),

    "job":
        sample["job"],

    "merch_lat":
        float(sample["merch_lat"]),

    "merch_long":
        float(sample["merch_long"]),

    "dob":
        sample["dob"],

    "unix_time":
        int(sample["unix_time"])
}


# Send request
response = requests.post(
    "http://127.0.0.1:5000/predict",
    json=data
)


print("\nHTTP Status:")
print(response.status_code)


print("\nAPI Response:")
print(response.json())


print("\nActual Label:")
print(int(sample["is_fraud"]))


print("\n===================================")
print("Flask API Test Completed")
print("===================================")