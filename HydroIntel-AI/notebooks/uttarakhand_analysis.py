import pandas as pd
import plotly.express as px

rainfall = pd.read_csv("data/clean_rainfall.csv")

# Filter Uttarakhand data
uk = rainfall[rainfall["SUBDIVISION"] == "UTTARAKHAND"]

print(uk.head())

fig = px.line(
    uk,
    x="YEAR",
    y="ANNUAL",
    title="Uttarakhand Annual Rainfall Trend",
    markers=True
)

fig.show()