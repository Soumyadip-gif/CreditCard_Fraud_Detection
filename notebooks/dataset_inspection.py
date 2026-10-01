import pandas as pd

# ==========================================
# LOAD DATASETS
# ==========================================

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")


# ==========================================
# TRAIN DATA INSPECTION
# ==========================================

print("\n========== TRAIN DATA ==========")

print("\nShape:")
print(train.shape)

print("\nColumns:")
print(train.columns.tolist())

print("\nFirst 5 Rows:")
print(train.head())

print("\nData Types:")
print(train.dtypes)

print("\nMissing Values:")
print(train.isnull().sum())

print("\nDuplicate Rows:")
print(train.duplicated().sum())

print("\nFraud Distribution:")
print(train["is_fraud"].value_counts())

print("\nFraud Percentage:")
print(train["is_fraud"].value_counts(normalize=True) * 100)


# ==========================================
# TEST DATA INSPECTION
# ==========================================

print("\n\n========== TEST DATA ==========")

print("\nShape:")
print(test.shape)

print("\nColumns:")
print(test.columns.tolist())

print("\nFirst 5 Rows:")
print(test.head())

print("\nData Types:")
print(test.dtypes)

print("\nMissing Values:")
print(test.isnull().sum())

print("\nDuplicate Rows:")
print(test.duplicated().sum())


# ==========================================
# TEST FRAUD DISTRIBUTION
# ==========================================

if "is_fraud" in test.columns:

    print("\nTest Fraud Distribution:")
    print(test["is_fraud"].value_counts())

    print("\nTest Fraud Percentage:")
    print(test["is_fraud"].value_counts(normalize=True) * 100)

else:

    print("\nTest dataset does not contain 'is_fraud' column.")


# ==========================================
# END
# ==========================================

print("\n==========================================")
print("Dataset Inspection Completed Successfully")
print("==========================================")
