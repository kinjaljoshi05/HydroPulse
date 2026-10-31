import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load datasets
rainfall = pd.read_csv("data/clean_rainfall.csv")
reservoir = pd.read_csv("data/clean_reservoir.csv")

# Uttarakhand only
uk = rainfall[rainfall["SUBDIVISION"] == "UTTARAKHAND"]

# Average reservoir values
avg_storage = reservoir["Storage"].mean()
avg_level = reservoir["Level"].mean()

# Create training dataset
data = pd.DataFrame()

data["YEAR"] = uk["YEAR"]
data["RAINFALL"] = uk["ANNUAL"]

data["STORAGE"] = avg_storage
data["LEVEL"] = avg_level

# Estimated hydropower potential
data["POWER"] = (
    data["RAINFALL"] * 0.15
    + data["STORAGE"] * 100
    + data["LEVEL"] * 0.25
)

# Features
X = data[["YEAR", "RAINFALL", "STORAGE", "LEVEL"]]

# Target
y = data["POWER"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "models/hydropower_model.pkl")

print("Model trained successfully!")