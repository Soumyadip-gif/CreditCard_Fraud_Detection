from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib


app = Flask(__name__)


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


# ==============================
# Home Route
# ==============================

@app.route("/")
def home():

    return render_template("index.html")


# ==============================
# Prediction Route
# ==============================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        if not data:

            return jsonify(
                {
                    "error": "No transaction data received"
                }
            ), 400


        # ==============================
        # Required Fields
        # ==============================

        required_fields = [
            "trans_date_trans_time",
            "merchant",
            "category",
            "amt",
            "gender",
            "city",
            "state",
            "zip",
            "city_pop",
            "job",
            "age",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
            "unix_time"
        ]


        missing_fields = [
            field
            for field in required_fields
            if field not in data
        ]


        if missing_fields:

            return jsonify(
                {
                    "error":
                        "Missing fields: "
                        + ", ".join(missing_fields)
                }
            ), 400


        # ==============================
        # Create DataFrame
        # ==============================

        df = pd.DataFrame(
            [data]
        )


        # ==============================
        # Date Features
        # ==============================

        df["trans_date_trans_time"] = (
            pd.to_datetime(
                df["trans_date_trans_time"]
            )
        )


        df["transaction_hour"] = (
            df["trans_date_trans_time"].dt.hour
        )


        df["transaction_day"] = (
            df["trans_date_trans_time"].dt.day
        )


        df["transaction_month"] = (
            df["trans_date_trans_time"].dt.month
        )


        df["transaction_day_of_week"] = (
            df["trans_date_trans_time"].dt.dayofweek
        )


        df["is_weekend"] = (
            df["transaction_day_of_week"] >= 5
        ).astype(int)


        # ==============================
        # Customer Age
        # ==============================

        df["customer_age"] = (
            pd.to_numeric(
                df["age"],
                errors="coerce"
            )
        )


        # ==============================
        # Customer-Merchant Distance
        # ==============================

        df["customer_merchant_distance"] = (
            haversine_distance(
                df["lat"],
                df["long"],
                df["merch_lat"],
                df["merch_long"]
            )
        )


        # ==============================
        # Remove Unused Input Fields
        # ==============================

        df = df.drop(
            columns=[
                "trans_date_trans_time",
                "age"
            ]
        )


        # ==============================
        # Categorical Encoding
        # ==============================

        df_encoded = encoder.transform(
            df
        )


        # ==============================
        # Match Training Feature Order
        # ==============================

        df_encoded = df_encoded.reindex(
            columns=model.feature_names_in_
        )


        # ==============================
        # Prediction
        # ==============================

        probability = model.predict_proba(
            df_encoded
        )[0][1]


        prediction = int(
            probability >= threshold
        )


        # ==============================
        # Risk Level
        # ==============================

        if probability >= threshold:

            risk_level = "HIGH"

        elif probability >= 0.50:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"


        # ==============================
        # Response
        # ==============================

        return jsonify(
            {
                "prediction":
                    "FRAUD"
                    if prediction == 1
                    else "LEGITIMATE",

                "fraud_probability":
                    round(
                        float(probability),
                        4
                    ),

                "risk_level":
                    risk_level,

                "threshold":
                    float(threshold)
            }
        )


    except Exception as e:

        return jsonify(
            {
                "error": str(e)
            }
        ), 500


# ==============================
# Run Application
# ==============================

if __name__ == "__main__":

    app.run(
        debug=True
    )