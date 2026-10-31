import pandas as pd

# Load datasets
rainfall = pd.read_csv("data/rainfall.csv")
reservoir = pd.read_csv("data/reservoir.csv")

print("\n===== RAINFALL DATA =====")
print(rainfall.head())

print("\nColumns:")
print(rainfall.columns)

print("\nShape:")
print(rainfall.shape)

print("\n===== RESERVOIR DATA =====")
print(reservoir.head())

print("\nColumns:")
print(reservoir.columns)

print("\nShape:")
print(reservoir.shape)