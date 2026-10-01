import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

y_train = pd.read_csv(
    "dataset/processed/y_train.csv"
)["is_fraud"]

classes = np.array(sorted(y_train.unique()))

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(
    zip(classes, weights)
)

print("\n========== CLASS BALANCE ==========")

print("\nClass Distribution:")
print(y_train.value_counts())

print("\nClass Percentage:")
print(y_train.value_counts(normalize=True) * 100)

print("\nCalculated Class Weights:")
print(class_weights)

print("\n===================================")
print("Class Balance Analysis Completed")
print("===================================")