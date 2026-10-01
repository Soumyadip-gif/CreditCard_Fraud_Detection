import pandas as pd

X_train = pd.read_csv("dataset/processed/X_train.csv")
X_test = pd.read_csv("dataset/processed/X_test.csv")
y_train = pd.read_csv("dataset/processed/y_train.csv")
y_test = pd.read_csv("dataset/processed/y_test.csv")

print("\n========== PROCESSED DATA VERIFICATION ==========")

print("\nX_train Shape:")
print(X_train.shape)

print("\nX_test Shape:")
print(X_test.shape)

print("\ny_train Shape:")
print(y_train.shape)

print("\ny_test Shape:")
print(y_test.shape)

print("\nX_train Missing Values:")
print(X_train.isnull().sum().sum())

print("\nX_test Missing Values:")
print(X_test.isnull().sum().sum())

print("\ny_train Distribution:")
print(y_train["is_fraud"].value_counts())

print("\ny_test Distribution:")
print(y_test["is_fraud"].value_counts())

print("\nX_train Data Types:")
print(X_train.dtypes)

print("\n==============================================")
print("Processed Data Verification Completed")
print("==============================================")