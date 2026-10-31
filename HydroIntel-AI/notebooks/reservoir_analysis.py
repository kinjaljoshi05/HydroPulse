import pandas as pd
import plotly.express as px

reservoir = pd.read_csv("data/clean_reservoir.csv")

# Top 10 reservoirs by average storage
top_storage = (
    reservoir.groupby("Reservoir_name")["Storage"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig = px.bar(
    top_storage,
    x="Reservoir_name",
    y="Storage",
    title="Top 10 Reservoirs by Average Storage"
)

fig.show()