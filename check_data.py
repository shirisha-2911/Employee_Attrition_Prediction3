import pandas as pd

df = pd.read_csv("data/Employee Attrition.csv")

print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nTarget Distribution:")
print(df["Attrition"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())