import pandas as pd
import plotly.express as px

# Load cleaned rainfall data
rainfall = pd.read_csv("data/clean_rainfall.csv")

# Group by year
yearly_rainfall = rainfall.groupby("YEAR")["ANNUAL"].mean().reset_index()

# Create graph
fig = px.line(
    yearly_rainfall,
    x="YEAR",
    y="ANNUAL",
    title="Annual Rainfall Trend in India",
    markers=True
)

fig.show()