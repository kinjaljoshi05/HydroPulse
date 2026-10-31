import pandas as pd

# Load datasets
rainfall = pd.read_csv("data/rainfall.csv")
reservoir = pd.read_csv("data/reservoir.csv")

print("Original Rainfall Shape:", rainfall.shape)
print("Original Reservoir Shape:", reservoir.shape)

# ---------------------------
# Rainfall Dataset Cleaning
# ---------------------------

rainfall = rainfall.dropna()

# Keep only useful columns
rainfall = rainfall[
    [
        "SUBDIVISION",
        "YEAR",
        "ANNUAL"
    ]
]

print("\nRainfall After Cleaning:")
print(rainfall.head())

# ---------------------------
# Reservoir Dataset Cleaning
# ---------------------------

reservoir = reservoir.dropna()

# Keep only useful columns
reservoir = reservoir[
    [
        "Reservoir_name",
        "Date",
        "Storage",
        "Level"
    ]
]

print("\nReservoir After Cleaning:")
print(reservoir.head())

# Save cleaned files
rainfall.to_csv("data/clean_rainfall.csv", index=False)
reservoir.to_csv("data/clean_reservoir.csv", index=False)

print("\nCleaned datasets saved successfully!")