import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="HydroIntel AI",
    page_icon="🌊",
    layout="wide"
)

# ==================================
# CUSTOM STYLE
# ==================================

st.markdown("""
<style>
.block-container{
    padding-top:1rem;
}
</style>
""", unsafe_allow_html=True)

# ==================================
# HEADER
# ==================================

st.title("🌊 HydroIntel AI")
st.markdown("""
### 🎯 Executive Dashboard

This AI system analyzes rainfall and reservoir datasets,
estimates hydropower generation potential,
identifies operational risk,
and predicts future energy output using Machine Learning.
""")
st.subheader("Smart Hydropower Analytics & Forecasting System")

st.info(
    "Upload rainfall and reservoir datasets to automatically generate analytics, risk assessment and hydropower forecasts."
)

# ==================================
# SIDEBAR
# ==================================

st.sidebar.header("📂 Upload Datasets")

rainfall_file = st.sidebar.file_uploader(
    "Upload Rainfall CSV",
    type=["csv"]
)

reservoir_file = st.sidebar.file_uploader(
    "Upload Reservoir CSV",
    type=["csv"]
)

# ==================================
# FILE CHECK
# ==================================

if rainfall_file is None or reservoir_file is None:
    st.warning(
        "Please upload both Rainfall and Reservoir CSV files."
    )
    st.stop()

# ==================================
# LOAD DATA
# ==================================

rainfall = pd.read_csv(rainfall_file)
reservoir = pd.read_csv(reservoir_file)

# ==================================
# VALIDATION
# ==================================

required_rainfall = [
    "SUBDIVISION",
    "YEAR",
    "ANNUAL"
]

required_reservoir = [
    "Reservoir_name",
    "Storage",
    "Level"
]

missing_rainfall = [
    col for col in required_rainfall
    if col not in rainfall.columns
]

missing_reservoir = [
    col for col in required_reservoir
    if col not in reservoir.columns
]

if missing_rainfall:
    st.error(
        f"Rainfall CSV missing columns: {missing_rainfall}"
    )
    st.stop()

if missing_reservoir:
    st.error(
        f"Reservoir CSV missing columns: {missing_reservoir}"
    )
    st.stop()

st.success("✅ Datasets validated successfully")

# ==================================
# REGION SELECTOR
# ==================================

regions = sorted(
    rainfall["SUBDIVISION"]
    .dropna()
    .unique()
)

selected_region = st.sidebar.selectbox(
    "Select Region",
    regions
)

filtered_rainfall = rainfall[
    rainfall["SUBDIVISION"] == selected_region
]

# ==================================
# DATA QUALITY SCORE
# ==================================

total_cells = rainfall.size + reservoir.size

missing_cells = (
    rainfall.isnull().sum().sum()
    +
    reservoir.isnull().sum().sum()
)

quality_score = round(
    ((total_cells - missing_cells)
     / total_cells) * 100,
    2
)
# ==================================
# EXECUTIVE DECISION PANEL
# ==================================

st.header("🎯 Executive Decision Panel")

best_region = (
    rainfall.groupby("SUBDIVISION")["ANNUAL"]
    .mean()
    .sort_values(ascending=False)
    .index[0]
)

highest_risk_reservoir = (
    reservoir.sort_values("Storage")
    .iloc[0]["Reservoir_name"]
)

system_status = (
    "🟢 STABLE"
    if quality_score >= 95
    else "🟡 REVIEW REQUIRED"
)

e1, e2, e3, e4 = st.columns([1,1,2,2])

e1.metric(
    "System Status",
    system_status
)

e2.metric(
    "Data Quality",
    f"{quality_score}%"
)

e3.metric(
    "Best Rainfall Region",
    best_region
)

e4.metric(
    "Highest Risk Reservoir",
    highest_risk_reservoir
)

# ==================================
# DATASET OVERVIEW
# ==================================

st.header("📊 Dataset Overview")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Rainfall Records",
    len(rainfall)
)

c2.metric(
    "Reservoir Records",
    len(reservoir)
)

c3.metric(
    "Rainfall Columns",
    len(rainfall.columns)
)

c4.metric(
    "Reservoir Columns",
    len(reservoir.columns)
)

c5.metric(
    "Data Quality %",
    quality_score
)

with st.expander("View Rainfall Dataset Preview"):
    st.dataframe(
        rainfall.head(),
        use_container_width=True
    )

with st.expander("View Reservoir Dataset Preview"):
    st.dataframe(
        reservoir.head(),
        use_container_width=True
    )
    # ==================================
# DATASET HEALTH
# ==================================

st.header("🩺 Dataset Health")

missing_values = (
    rainfall.isnull().sum().sum()
    +
    reservoir.isnull().sum().sum()
)

duplicate_rows = (
    rainfall.duplicated().sum()
    +
    reservoir.duplicated().sum()
)

health1, health2 = st.columns(2)

health1.metric(
    "Missing Values",
    int(missing_values)
)

health2.metric(
    "Duplicate Rows",
    int(duplicate_rows)
)

if missing_values == 0:
    st.success("✅ No missing values detected")
else:
    st.warning(
        f"⚠️ {missing_values} missing values detected"
    )

if duplicate_rows == 0:
    st.success("✅ No duplicate rows detected")
else:
    st.warning(
        f"⚠️ {duplicate_rows} duplicate rows detected"
    )

# ==================================
# KPI CALCULATIONS
# ==================================

avg_rainfall = round(
    filtered_rainfall["ANNUAL"].mean(),
    2
)

avg_storage = round(
    reservoir["Storage"].fillna(0).mean(),
    2
)

avg_level = round(
    reservoir["Level"].fillna(0).mean(),
    2
)

estimated_generation = round(
    (avg_rainfall * 0.15)
    + (avg_storage * 100)
    + (avg_level * 0.25),
    2
)

# ==================================
# RISK
# ==================================

if avg_storage > 2:
    risk = "Low"
elif avg_storage > 0.8:
    risk = "Medium"
else:
    risk = "High"

# ==================================
# KPI DASHBOARD
# ==================================

st.header("📈 Key Performance Indicators")

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(
        "🌧 Avg Rainfall",
        f"{avg_rainfall:,.2f}"
    )

with k2:
    st.metric(
        "💧 Avg Storage",
        f"{avg_storage:,.2f}"
    )

with k3:
    st.metric(
        "🏞 Reservoir Level",
        f"{avg_level:,.2f}"
    )

with k4:
    st.metric(
        "⚠ Risk Status",
        risk
    )

with k5:
    st.metric(
        "⚡ Hydro Potential",
        f"{estimated_generation:,.2f} MW"
    )

# ==================================
# ANALYSIS SUMMARY
# ==================================

highest_row = filtered_rainfall.loc[
    filtered_rainfall["ANNUAL"].idxmax()
]

highest_year = int(
    highest_row["YEAR"]
)

highest_rainfall = round(
    highest_row["ANNUAL"],
    2
)


# ==================================
# REGION COMPARISON
# ==================================

st.header("⚔️ Region Comparison Dashboard")

col_a, col_b = st.columns(2)

region_1 = col_a.selectbox(
    "Select Region 1",
    regions,
    key="region1"
)

region_2 = col_b.selectbox(
    "Select Region 2",
    regions,
    index=min(1, len(regions)-1),
    key="region2"
)

data_1 = rainfall[
    rainfall["SUBDIVISION"] == region_1
]

data_2 = rainfall[
    rainfall["SUBDIVISION"] == region_2
]

avg_1 = round(data_1["ANNUAL"].mean(), 2)
avg_2 = round(data_2["ANNUAL"].mean(), 2)

max_1 = round(data_1["ANNUAL"].max(), 2)
max_2 = round(data_2["ANNUAL"].max(), 2)

compare_df = pd.DataFrame({
    "Metric": [
        "Average Rainfall",
        "Highest Rainfall"
    ],
    region_1: [
        avg_1,
        max_1
    ],
    region_2: [
        avg_2,
        max_2
    ]
})

st.dataframe(
    compare_df,
    use_container_width=True
)

comparison_chart = pd.DataFrame({
    "Region": [
        region_1,
        region_2
    ],
    "Average Rainfall": [
        avg_1,
        avg_2
    ]
})

fig_compare = px.bar(
    comparison_chart,
    x="Region",
    y="Average Rainfall",
    title="Average Rainfall Comparison"
)

st.plotly_chart(
    fig_compare,
    use_container_width=True
)
# ==================================
# RAINFALL ANALYTICS
# ==================================

st.header("🌧️ Rainfall Analytics")

fig1 = px.line(
    filtered_rainfall,
    x="YEAR",
    y="ANNUAL",
    title=f"Annual Rainfall Trend - {selected_region}",
    markers=True
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

fig_hist = px.histogram(
    filtered_rainfall,
    x="ANNUAL",
    nbins=20,
    title="Rainfall Distribution"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)

# ==================================
# RESERVOIR ANALYTICS
# ==================================

st.header("🏞️ Reservoir Analytics")

top_storage = (
    reservoir.groupby(
        "Reservoir_name"
    )["Storage"]
    .mean()
    .sort_values(
        ascending=False
    )
    .head(10)
    .reset_index()
)

fig2 = px.bar(
    top_storage,
    x="Reservoir_name",
    y="Storage",
    title="Top 10 Reservoirs by Average Storage"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)
# ==================================
# RESERVOIR RISK RANKING
# ==================================

st.header("🏆 Reservoir Risk Ranking")

risk_df = reservoir.groupby(
    "Reservoir_name"
).agg({
    "Storage": "mean",
    "Level": "mean"
}).reset_index()

def classify_risk(storage):

    if storage > 3:
        return "🟢 Low"

    elif storage > 1:
        return "🟡 Medium"

    else:
        return "🔴 High"

risk_df["Risk"] = risk_df["Storage"].apply(
    classify_risk
)

st.dataframe(
    risk_df.sort_values(
        "Storage",
        ascending=False
    ),
    use_container_width=True
)
# ==================================
# FORECAST SETTINGS
# ==================================

forecast_years = st.slider(
    "Forecast Years",
    3,
    20,
    6
)

# ==================================
# FORECASTING
# ==================================

st.header("🔮 Hydropower Forecast")

model = joblib.load(
    "models/hydropower_model.pkl"
)

future_years = np.arange(
    2025,
    2025 + forecast_years
)

future_df = pd.DataFrame({
    "YEAR": future_years,
    "RAINFALL": np.linspace(
        avg_rainfall,
        avg_rainfall * 1.05,
        len(future_years)
    ),
    "STORAGE": np.linspace(
        avg_storage,
        avg_storage * 1.05,
        len(future_years)
    ),
    "LEVEL": np.linspace(
        avg_level,
        avg_level * 1.05,
        len(future_years)
    )
})

future_df["Forecast_MW"] = model.predict(
    future_df[
        [
            "YEAR",
            "RAINFALL",
            "STORAGE",
            "LEVEL"
        ]
    ]
)

st.dataframe(
    future_df[
        [
            "YEAR",
            "Forecast_MW"
        ]
    ],
    use_container_width=True
)

csv = future_df.to_csv(
    index=False
)

st.download_button(
    "📥 Download Forecast Report",
    csv,
    "forecast_report.csv",
    "text/csv"
)

fig3 = px.line(
    future_df,
    x="YEAR",
    y="Forecast_MW",
    title="Projected Hydropower Potential",
    markers=True
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ==================================
# AI INSIGHT
# ==================================

growth = round(
    (
        (
            future_df["Forecast_MW"].iloc[-1]
            -
            future_df["Forecast_MW"].iloc[0]
        )
        /
        future_df["Forecast_MW"].iloc[0]
    ) * 100,
    2
)

st.header("🤖 AI Insight")

st.info(f"""
Hydropower potential is expected to increase by {growth}% over the forecast period.

Region under analysis: {selected_region}

Reservoir storage indicates {risk.lower()} operational risk.

Dataset quality score is {quality_score}%.

The uploaded datasets support future hydropower planning and decision making.
""")
# ==================================
# AI RECOMMENDATIONS
# ==================================

st.header("💡 AI Recommendations")

recommendations = []

if risk == "High":
    recommendations.append(
        "Increase reservoir storage capacity and improve water conservation measures."
    )

if quality_score < 95:
    recommendations.append(
        "Dataset quality should be improved before making critical decisions."
    )

if growth > 5:
    recommendations.append(
        "Future hydropower generation outlook is positive. Consider expansion planning."
    )

if avg_rainfall > 2000:
    recommendations.append(
        "Region receives high rainfall. Flood monitoring systems are recommended."
    )

if len(recommendations) == 0:
    recommendations.append(
        "Current system indicators are stable. Continue regular monitoring."
    )

for item in recommendations:
    st.success(item)
# ==================================
# PROJECT SUMMARY
# ==================================

st.header("📋 Project Summary")

st.write("""
HydroIntel AI is a smart hydropower analytics platform.

Users can upload rainfall and reservoir datasets,
select any region, validate data quality,
generate analytics, monitor risk levels
and forecast future hydropower potential.

Key Modules:

• Dataset Upload

• Region Selection

• Dataset Validation

• Data Quality Assessment

• Rainfall Analytics

• Reservoir Analytics

• Risk Assessment

• Machine Learning Forecasting

• Forecast Report Download

• AI Insight Generation
""")