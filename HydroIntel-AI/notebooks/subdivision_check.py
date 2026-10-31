import pandas as pd

rainfall = pd.read_csv("data/clean_rainfall.csv")

print(rainfall["SUBDIVISION"].unique())