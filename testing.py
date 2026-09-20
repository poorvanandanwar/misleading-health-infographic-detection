import pandas as pd

train = pd.read_csv("C:/Users/aditi/Desktop/DL PROJECT/misleading-health-infographic-detection/data/processed/phase5/datasets/claim_train.csv")

print("Dataset shape:", train.shape)

print("\nAll label counts:")
print(train["binary_target"].value_counts(dropna=False))

print("\nUnique labels:")
print(train["binary_target"].unique())

print("\nData types:")
print(train["binary_target"].dtype)