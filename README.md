# CreditGuard AI — Credit Card Fraud Detection

**CreditGuard AI** is a machine-learning-powered web application designed to detect potentially fraudulent credit card transactions.

The system processes transaction, customer, merchant, location, and time-related information and uses a trained **HistGradientBoostingClassifier** to classify transactions as **Fraudulent** or **Legitimate**.

The project combines machine learning, feature engineering, categorical encoding, class-imbalance handling, and a Flask-based web interface into an end-to-end fraud detection system.

---

## Project Overview

Credit card fraud is a major challenge in digital financial transactions. Traditional rule-based systems can struggle to identify complex fraud patterns.

CreditGuard AI uses machine learning to analyze transaction characteristics and generate a fraud prediction based on learned patterns.

### Main Workflow

```text
Transaction Input
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Categorical Encoding
       ↓
HistGradientBoosting Model
       ↓
Fraud Probability
       ↓
Threshold = 0.90
       ↓
Fraud / Legitimate
```

---

## Key Features

* AI-powered credit card fraud detection
* Real-time transaction analysis
* Machine-learning-based classification
* CatBoost target encoding for categorical features
* HistGradientBoosting classification
* Class-imbalance handling using balanced sample weights
* Transaction time feature extraction
* Customer age calculation
* Customer-to-merchant geographical distance calculation
* Configurable fraud probability threshold
* Professional dark fintech-style web interface
* Flask-based backend
* Saved model and encoder for prediction
* Training and evaluation notebooks included

---

## Machine Learning Pipeline

The final training pipeline follows several stages.

### 1. Dataset Loading

The model is trained using the transaction training dataset.

```python
train = pd.read_csv("dataset/train.csv")
```

The original large training dataset is **not included in the GitHub repository** because of its size.

It remains available locally for model retraining.

---

### 2. Removing Unnecessary Columns

The following columns are removed:

```text
Unnamed: 0
first
last
street
trans_num
```

These fields are not required by the final prediction pipeline.

---

### 3. Transaction Date Feature Engineering

The transaction timestamp is converted into useful numerical features:

```text
transaction_hour
transaction_day
transaction_month
transaction_day_of_week
is_weekend
```

This allows the model to learn patterns related to transaction timing.

---

### 4. Customer Age Calculation

The customer's date of birth and transaction date are used to calculate the customer's age.

```text
customer_age
```

The calculated age is rounded to one decimal place.

---

### 5. Customer-Merchant Distance

CreditGuard AI calculates the geographical distance between the customer's location and merchant location using the **Haversine formula**.

Input coordinates:

```text
Customer:
lat
long

Merchant:
merch_lat
merch_long
```

Generated feature:

```text
customer_merchant_distance
```

This feature can help identify unusual transactions occurring far from the customer's normal location.

---

### 6. Raw Date Removal

After feature extraction, the original date columns are removed:

```text
trans_date_trans_time
dob
```

---

### 7. Target Separation

The prediction target is:

```text
is_fraud
```

The credit card number is also excluded from the model features:

```text
cc_num
```

Therefore:

```text
X = transaction features
y = is_fraud
```

---

## Categorical Feature Encoding

The following categorical columns are encoded:

```text
merchant
category
gender
city
state
job
```

CreditGuard AI uses:

**CatBoostEncoder**

```python
encoder = ce.CatBoostEncoder(
    cols=categorical_columns,
    random_state=42
)
```

The trained encoder is saved as:

```text
model/catboost_encoder.pkl
```

Saving the encoder is important because the same transformation must be applied when the Flask application receives a new transaction.

---

## Class Imbalance Handling

Credit card fraud datasets are usually highly imbalanced, meaning legitimate transactions greatly outnumber fraudulent transactions.

CreditGuard AI uses balanced sample weights:

```python
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)
```

These weights are passed during model training so that the model gives appropri
