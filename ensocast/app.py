import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import datetime
import tensorflow as tf
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

LABEL_MAP = {0: "La Niña", 1: "Neutral", 2: "El Niño"}
PHASE_COLORS = {"El Niño": "#f6416c", "Neutral": "#94a3b8", "La Niña": "#3b82f6"}
PHASE_EMOJIS = {"El Niño": "🔴", "Neutral": "⚪", "La Niña": "🔵"}


def render_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[2, 1, 0])
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted Phase", y="Actual Phase", color="Count"),
        x=["🔴 El Niño", "⚪ Neutral", "🔵 La Niña"],
        y=["🔴 El Niño", "⚪ Neutral", "🔵 La Niña"],
        color_continuous_scale="Reds",
    )
    fig_cm.update_layout(height=400, margin=dict(t=30, b=30))
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            fig_cm.add_annotation(
                x=j, y=i, text=str(cm[i][j]),
                showarrow=False, font_size=16,
                font_color="white" if cm.max() > 0 and cm[i][j] > cm.max() / 2 else "black",
            )
    st.plotly_chart(fig_cm, use_container_width=True)

    total = cm.sum()
    correct = cm.diagonal().sum()
    off_diag = total - correct
    if correct / total >= 0.85:
        insight = "Your model demonstrates strong classification performance across all ENSO phases, with most predictions falling along the diagonal."
    elif off_diag > correct:
        insight = "The confusion matrix reveals significant misclassifications between phases, indicating the complexity of distinguishing ENSO states in your selected time period."
    elif cm.max() > total * 0.5:
        insight = "The model shows concentrated performance in certain phase predictions, with some ENSO states being more predictable than others."
    else:
        insight = "The confusion matrix displays a balanced mix of correct and incorrect predictions, reflecting the inherent challenges in ENSO phase classification."
    st.markdown(f"""
    <div class="insight-card" style="text-align: center;">
        <h4>🧮 <strong>Accurate or confused?</strong></h4>
        <p>{insight}</p>
    </div>
    """, unsafe_allow_html=True)


def render_classification_report(y_test_phases, y_pred_phases, selected_phases):
    all_present_phases = sorted(set(y_test_phases + y_pred_phases))
    if len(y_test_phases) == 0:
        st.error("No test data available for classification report")
        st.stop()
    try:
        report = classification_report(
            y_test_phases, y_pred_phases,
            labels=all_present_phases, output_dict=True, zero_division=0,
        )
    except Exception as e:
        st.error(f"Could not generate classification report: {str(e)}")
        st.stop()

    if len(all_present_phases) == 1:
        st.info(f"ℹ️ Only one phase ({all_present_phases[0]}) appears in the test data. Limited metrics available.")
    elif len(all_present_phases) == 2:
        st.info(f"ℹ️ Only two phases appear in test data: {', '.join(all_present_phases)}")
    missing = set(selected_phases) - set(all_present_phases)
    if missing:
        st.warning(f"⚠️ Selected phase(s) not in test data: {', '.join(missing)}")

    for phase in [p for p in ["El Niño", "Neutral", "La Niña"] if p in selected_phases]:
        metrics = report.get(phase)
        color = PHASE_COLORS[phase]
        emoji = PHASE_EMOJIS[phase]
        if metrics:
            precision = metrics["precision"]
            recall = metrics["recall"]
            f1 = metrics["f1-score"]
            support = int(metrics["support"])
            month_label = "month" if support == 1 else "months"
            was_were = "was" if support == 1 else "were"
            st.markdown(f"""
            <div class="insight-card" style="background: linear-gradient(135deg, {color}20, {color}40);">
                <h4>{emoji} {phase} Performance</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                    <div><strong>Precision:</strong> {precision:.0%}<br>
                        <small>When the model predicts "{phase}", it is correct {precision:.0%} of the time.</small></div>
                    <div><strong>Recall:</strong> {recall:.0%}<br>
                        <small>The model correctly identifies {recall:.0%} of actual {phase} months.</small></div>
                    <div><strong>F1-Score:</strong> {f1:.0%}<br>
                        <small>This is a balanced metric combining precision and recall.</small></div>
                    <div><strong>Sample Size:</strong> {support}<br>
                        <small>In the testing dataset, {support} {month_label} {was_were} labeled as {phase}.</small></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-card" style="background: linear-gradient(135deg, #64748b20, #64748b40);">
                <h4>{emoji} {phase} Performance</h4>
                <p style="text-align: center; color: #64748b; font-style: italic;">
                    No {phase} samples found in the test data for this time period and selection.
                </p>
            </div>
            """, unsafe_allow_html=True)

st.set_page_config(
    page_title="ENSOcast",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .story-card {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        font-size: 18px;
        text-align: center;
        color: black;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .story-card p {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    
    .description {
        text-align: left;
        padding-left: 2rem;
    }
    
    
    .phase-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .el-nino { background: linear-gradient(135deg, #ff9a8b 0%, #f6416c 100%); color: white; }
    .la-nina { background: linear-gradient(135deg, #a8edea 0%, #3b82f6 100%); color: white; }
    .neutral { background: linear-gradient(135deg, #d299c2 0%, #fef9d3 100%); color: #333; }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .insight-card {
        text-align: center;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .metric-container { 
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);    
        padding: 1rem; 
        border-radius: 10px; 
        text-align: center; 
        margin: 0.5rem; 
        }
    
    .model-card {
        background: linear-gradient(135deg, #667eea20, #764ba240);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .enhancement-badge {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

def create_story_intro():
    st.markdown("""
    <div class="main-header">
        <h1>🌊 ENSOcast</h1>
        <h3>Decoding El Niño–Southern Oscillation</h3>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("merged_enso.csv", parse_dates=["Date"])
        return df
    except FileNotFoundError:
        st.error("Data files not found. Please ensure merged_enso.csv and ENSOcast_model.pkl are in the correct directory.")
        return None

@st.cache_data
def load_sst_dataset():
    try:
        file_sst = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
        ds = xr.open_dataset(file_sst)
        ds["time"] = pd.to_datetime(ds["time"].values)
        return ds
    except:
        st.warning("Could not connect to live SST data. Using cached data for demonstration.")
        return None

def create_feature_for_date(target_date, df, feature_cols):
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    elif isinstance(target_date, datetime.date):
        target_date = pd.Timestamp(target_date)

    df_sorted = df.sort_values('Date')
    time_diffs = (df_sorted['Date'] - target_date).dt.days.abs()
    closest_idx = time_diffs.idxmin()
    closest_position = df_sorted.index.get_loc(closest_idx)

    month = target_date.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    start_idx = max(0, closest_position - 12)
    end_idx = closest_position + 1
    recent_data = df_sorted.iloc[start_idx:end_idx]

    if len(recent_data) == 0:
        feature_values = df[feature_cols].mean()
    else:
        feature_values = recent_data[feature_cols].mean()

    if 'month_sin' in feature_cols:
        feature_values['month_sin'] = month_sin
    if 'month_cos' in feature_cols:
        feature_values['month_cos'] = month_cos

    return feature_values.values.reshape(1, -1)

data_loaded = True
try:
    df = load_data()
    if df is None:
        data_loaded = False
    else:
        sst_ds = load_sst_dataset()
        df = df.sort_values("Date").reset_index(drop=True)
        DATA_MAX_YEAR = int(df["Date"].dt.year.max())
        split_idx = int(0.8 * len(df))

        feature_cols = [
            "SST_Anomaly", "SOI", "SOI_lag_1", "SOI_lag_2", "SOI_lag_3",
            "SST_Anomaly_lag_1", "SST_Anomaly_lag_2", "SST_Anomaly_lag_3",
            "month_sin", "month_cos"
        ]
        X = df[feature_cols]
        y = df["ENSO_Label"]

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        training_data_size = len(y_train)
        total_predictions = len(y_test)
        correct_predictions = int(accuracy * total_predictions)

        label_map = LABEL_MAP

        df.loc[df.index[split_idx:], "Predicted_Phase"] = [label_map[i] for i in y_pred]

        df.loc[df.index[split_idx:], "True_Phase"] = [label_map[i] for i in y_test]
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

create_story_intro()

if not data_loaded:
    st.error("⚠️Unable to load required data files. Please check your setup.")
    st.stop()

st.sidebar.title("🌊 ENSOcast")
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Tab Navigation")
page = st.sidebar.radio(
    "",
    ["🌍 Understanding ENSO",
     "🎯 Project Purpose",
     "🌡️ Ocean Temperatures",
     "📊 Explore Past Patterns",
     "🔬 Model Overview",
     "🔧 Train Baseline Model",
     "🛠️ Train Advanced Models"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown("*Created by Dylan Dsouza*")

if page == "🌍 Understanding ENSO":

    st.markdown("""
    <div class="story-card">
        <h2>🌍 ENSO: The Pulse of the Pacific</h2>
        <p>Imagine the Pacific Ocean as the Earth's beating heart. Every few years, its rhythm shifts. Ocean temperatures rise or fall, winds change direction, and rainfall patterns grow unpredictable. 
        Storms intensify in some places and disappear in others. Crops either thrive or fail.</p> 
        <p>This natural cycle, called the <b>El Niño–Southern Oscillation (ENSO)</b>, is driven by fluctuations 
        in sea surface temperature and atmospheric pressure in the Pacific Ocean. It moves through three phases:</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        color = "#f6416c"
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
            <h3>🔴 El Niño</h3>
            <h4><em><strong>"The Little Boy"</strong></em></h4>
            <p>Ocean warms up, trade winds weaken, and rainfall increases in the eastern Pacific. This often brings destructive floods to California and droughts from Australia to South Asia.</p>
            <small>Every 2–7 years</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        color = "#94a3b8"
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
            <h3>⚪ Neutral</h3>
            <h4><em><strong>"The Quiet Phase"</strong></em></h4>
            <p>Ocean temperatures stay near average, and weather patterns remain stable—neither extreme nor unusual. This brings typical seasonal weather worldwide.</p>
            <small>Most frequent state</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        color = "#3b82f6"
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
            <h3>🔵 La Niña</h3>
            <h4><em><strong>"The Little Girl"</strong></em></h4>
            <p>Ocean cools down, and trade winds strengthen. This typically causes droughts in South America and fuel intense hurricanes in the Atlantic.</p>
            <small>Often follows El Niño</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="story-card">
        <h2>💥 The Global Impact of ENSO</h2>
        <p>ENSO may begin in the Pacific, but its ripple effects reach around the world:</p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div style="background: #6c9faf; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4>🌾 Agriculture</h4>
                <p>Variations in rainfall patterns significantly affect crop production, food supply stability, and agricultural economies.</p>
            </div>
            <div style="background: #d77a7a; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4>🌤️ Weather</h4>
                <p>ENSO alters the distribution of extreme weather events, resulting in increased storms in some regions and decreased rainfall in others.</p>
            </div>
            <div style="background: #9a7ca8; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4>💰 Economy</h4>
                <p>Fluctuations in crop yields and energy demand influence financial markets worldwide and sway international trade.</p>
            </div>
            <div style="background: #e0c275; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4>🐠 Marine Life</h4>
                <p>Ocean temperature changes drive shifts in fish populations, degrade coral reef ecosystems, and disrupt marine food webs.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "🎯 Project Purpose":
    st.markdown("""
    <div class="story-card">
        <h2>🎯 Why ENSOcast Exists</h2>
        <p>The primary aim of ENSOcast is to provide a rudimentary and simplified bridge between climate science expertise and public understanding by making ENSO forecasting methodology <strong>interactive and accessible</strong>. By taking complex indices like the Oceanic Niño Index (ONI) and the Southern Oscillation Index (SOI) and breaking them down, ENSOcast aims to make the underlying logic of these indicators more <strong>visible and intuitive</strong>. Users can engage in hands-on experimentation, explore simplified models, and see firsthand how variations in Sea Surface Temperature (SST) and atmospheric pressure relate to different ENSO states.</p>
        <h3>🌊 Reverse-Engineering Climate Expertise</h3>
        <p>ENSOcast uses established climate indicators to help users discover <em>why</em> certain features (like 2-3 month temperature lags) consistently drive predictions. This approach bridges statistical methods with physical understanding, revealing how raw oceanographic observations become the standardized indices used in practical ENSO forecasting.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        color = "#6c9faf"
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); height: 180px;">
            <h3>🎓 Climate Education</h3>
            <p>Experience how climate predictions emerge from data, from rigorous feature engineering to specific model selection. Build intuition about probabilistic forecasting rather than being overwhelmed by jargon.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        color = "#d77a7a"
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); height: 180px;">
            <h3>📊 Hands-On Learning</h3>
            <p>Intuitively visualize temporal patterns emerging through interactive data exploration. Experiment with datasets to uncover trends and anomalies, gaining a understanding of how data drives climate insights.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="story-card">
    <h2>🔬 Five Models, Five Perspectives</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div style="background: #6c9faf; color: white; padding: 1rem; border-radius: 8px;">
                <h4>🌳 Random Forest</h4>
                <p>Captures non-linear feature interactions in tabular SST/SOI data with interpretability through feature importance, ideal for explainable ENSO classification.</p>
            </div>
            <div style="background: #d77a7a; color: white; padding: 1rem; border-radius: 8px;">
                <h4>🚀 XGBoost</h4>
                <p>Boosts decision trees to optimize phase prediction accuracy from engineered climate indicators, handling feature noise and missing data with resilience.</p>
            </div>
            <div style="background: #9a7ca8; color: white; padding: 1rem; border-radius: 8px;">
                <h4>🧠 1D CNN</h4>
                <p>Learns local temporal patterns in rolling SST/SOI sequences using convolutional layers, ideal for detecting short-term ENSO signal transitions.</p>
            </div>
            <div style="background: #e0c275; color: white; padding: 1rem; border-radius: 8px;">
                <h4>🕰️ LSTM</h4>
                <p>Models long-range temporal trends in SST/SOI sequences, making it well-suited for forecasting ENSO phases driven by multi-month climatic trends.</p>
            </div>
        </div>
        <div style="display: flex; justify-content: center; margin-top: 1rem;">
            <div style="background: #7aa89f; color: white; padding: 1rem; border-radius: 8px; max-width: 800px; width: 100%;">
                <h4>🤝 Ensemble Learning</h4>
                <p>Combines <strong>Random Forest</strong> and <strong>XGBoost</strong> predictions to leverage complementary strengths, improving overall ENSO phase prediction robustness and reducing model-specific biases.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "🌡️ Ocean Temperatures":
    st.markdown("""
    <div class="story-card">
        <h2>🌡️ Ocean Temperatures at a Glance</h2>
        <p>Global sea surface temperatures tell us a story. In the heart of the Pacific, <b>Niño 3.4</b> reveals the pulse of ENSO. Defined by the coordinates <em>5°N-5°S</em> latitude and <em>170°W-120°W</em> longitude, this rectangular region is where subtle changes in warmth tell us if El Niño or La Niña is forming.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.slider("Select Year", min_value=1982, max_value=DATA_MAX_YEAR, value=2010, help="Drag to select the year you want to explore", key="sst_year_slider")
    with col2:
        month_dict = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        selected_month = st.selectbox("Select Month",
                                    list(month_dict.keys()), index=7, help="Select the month you want to explore")

    month_num = month_dict[selected_month]

    if sst_ds is not None:
        try:
            with st.spinner(f"Downloading data for {selected_month} {selected_year}..."):
                sst_slice = sst_ds.sel(time=(sst_ds['time.year'] == selected_year) &
                                     (sst_ds['time.month'] == month_num))['sst']

                fig, ax = plt.subplots(figsize=(12, 6))

                im = sst_slice.plot(ax=ax, cmap='coolwarm',
                                  cbar_kwargs={"label": "Sea Surface Temperature (°C)", "shrink": 1})
                ax.add_patch(patches.Rectangle((190, -5), 50, 10, edgecolor='black', facecolor='none', linewidth=1, linestyle='--'))
                ax.text(189, 8, 'Niño 3.4 Region', color='black')

                ax.set_title(f'Sea Surface Temperature - {selected_month} {selected_year}',
                           fontsize=13, pad=10)
                ax.set_xlabel("Longitude (°E)", fontsize=12)
                ax.set_ylabel("Latitude (°N)", fontsize=12)

                max_temp = float(sst_slice.max())
                avg_temp = float(sst_slice.mean())
                min_temp = float(sst_slice.min())

                st.pyplot(fig)

                col1, col2, col3 = st.columns(3)
                with col1:
                    color = "#f6416c"
                    st.markdown(f"""
                    <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40);">
                        <h3>🔥 Hottest</h3>
                        <h2>{max_temp:.1f}°C</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    color = "#94a3b8"
                    st.markdown(f"""
                    <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40);">
                        <h3>🌡️ Average</h3>
                        <h2>{avg_temp:.1f}°C</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    color = "#3b82f6"
                    st.markdown(f"""
                    <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40);">
                        <h3>🧊 Coolest</h3>
                        <h2>{min_temp:.1f}°C</h2>
                    </div>
                    """, unsafe_allow_html=True)

                date_context = df[df['Date'].dt.year == selected_year]
                if len(date_context) > 0:
                    monthly_data = date_context[date_context['Date'].dt.month == month_num]
                    if len(monthly_data) > 0:
                        phase = monthly_data.iloc[0]['ENSO_Phase']
                        oni_value = monthly_data.iloc[0]['ONI']

                        phase_colors = {"El Niño": "#f6416c", "La Niña": "#3b82f6", "Neutral": "#94a3b8"}
                        phase_emojis = {"El Niño": "🔴", "La Niña": "🔵", "Neutral": "⚪"}

                        article = "an" if phase.startswith("El") else "a"

                        st.markdown(f"""
                            <div class="insight-card" style="background: {phase_colors[phase]}20;">
                                <h3>{phase_emojis[phase]} {selected_month} {selected_year} was {article} <strong>{phase}</strong> month with an ONI value of <strong>{oni_value:.2f}</strong></h3>
                            </div>
                            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"🌊 Unable to load ocean data for {selected_month} {selected_year}: {e}")
            st.info("💡 Try selecting a different date, or check your internet connection for satellite data.")
    else:
        st.warning("🛰️ Live satellite data unavailable.")

elif page == "📊 Explore Past Patterns":
    st.markdown("""
    <div class="story-card">
        <h2>📊 The Data Behind ENSO</h2>
        <p>To understand the future of ENSO, we first need to examine its past. These charts show how ENSO phases have shifted over time, highlighting patterns, trends, and seasonal tendencies.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        years = st.slider(
            "Select Year Range", 1982, DATA_MAX_YEAR, (2000, min(2020, DATA_MAX_YEAR)),
            help="Drag to select the time period you want to explore",
            key="explore_years_slider"
        )

    with col2:
        selected_phases = st.multiselect(
            "Select ENSO Phases",
            ["El Niño", "Neutral", "La Niña"],
            default=["El Niño", "Neutral", "La Niña"],
            help="Select which ENSO phases to include in your analysis",
            key="select_phases_slider"
        )

    df_date_filtered = df[
        (df["Date"].dt.year >= years[0]) &
        (df["Date"].dt.year <= years[1])
    ]

    total_months = len(df_date_filtered)

    all_phase_counts = df_date_filtered["ENSO_Phase"].value_counts()

    if total_months == 0:
        st.warning("No data available for your selected time period.")
        st.stop()

    available_phases = set(df_date_filtered["ENSO_Phase"].unique())
    selected_phases_available = [phase for phase in selected_phases if phase in available_phases]

    if len(selected_phases_available) == 0:
        st.warning("None of the selected ENSO phases are available for your selected time period.")
        st.stop()
    years_span = years[1] - years[0] + 1

    if years[0] == years[1]:
        st.markdown(f"""
            <div class="story-card">
                <h2>🕰️ Your Selected Timeline:</h2>
                <h3>January {years[0]} – December {years[1]}</h3>
                <p>In this <strong>1 year</strong>, here's how ENSO spent its time:</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="story-card">
                <h2>🕰️ Your Selected Timeline:</h2>
                <h3>January {years[0]} – December {years[1]}</h3>
                <p>In these <strong>{years_span} years</strong>, here's how ENSO spent its time:</p>
            </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    for i, phase in enumerate(selected_phases_available):
        count = all_phase_counts.get(phase, 0)

        percentage = (count / total_months) * 100
        color = "#f6416c" if phase == "El Niño" else "#3b82f6" if phase == "La Niña" else "#94a3b8"
        emoji = "🔴" if phase == "El Niño" else "🔵" if phase == "La Niña" else "⚪"

        years_duration = count // 12
        months = count % 12

        if years_duration > 0 and months > 0:
            time_str = f"{years_duration} year{'s' if years_duration > 1 else ''} {months} month{'s' if months > 1 else ''}"
        elif years_duration > 0:
            time_str = f"{years_duration} year{'s' if years_duration > 1 else ''}"
        else:
            time_str = f"{months} month{'s' if months > 1 else ''}"

        phase_columns = {"El Niño": col1, "Neutral": col2, "La Niña": col3}

        col = phase_columns[phase]

        with col:
            st.markdown(f"""
            <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40);">
                <h3>{emoji} {phase}</h2>
                <h4>{time_str}</h4>
                <p>{percentage:.1f}% of the time</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    df_filtered = df_date_filtered[df_date_filtered["ENSO_Phase"].isin(selected_phases)]

    phase_counts = df_filtered["ENSO_Phase"].value_counts()
    total_months = len(df_filtered)

    st.markdown("### 🌡️ Sea Surface Temperature (SST): The First ENSO Signal")

    st.markdown("""
    Sea Surface Temperature (SST) refers to the temperature of water measured at the ocean's surface. In the Niño 3.4 region, deviations from the climatological mean (the expected long-term seasonal average) provide early indications of ENSO development.""")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["SST"],
            name="Ocean Temperature",
            line=dict(color='skyblue', width=2),
            hovertemplate="<b>%{x}</b><br>Temperature: %{y:.2f}°C<extra></extra>"
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["SST_Climatology"],
            name="Expected Temperature",
            line=dict(color='orange', dash='dot', width=2),
            hovertemplate="<b>%{x}</b><br>Expected: %{y:.2f}°C<extra></extra>"
        ),
        secondary_y=False,
    )

    fig.update_layout(
        xaxis_title="Date",
        template="plotly_white",
        height=400,
        margin=dict(t=30, b=30),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.add_hrect(y0=27.75, y1=29.57, line_width=0, fillcolor="red", opacity=0.1)
    fig.add_hrect(y0=24.69, y1=26.45, line_width=0, fillcolor="blue", opacity=0.1)
    fig.add_hline(y=27.75, line_dash="dot", line_color="red", annotation_text="El Niño Conditions", annotation_position="top right")
    fig.add_hline(y=26.45, line_dash="dot", line_color="blue", annotation_text="La Niña Conditions", annotation_position="bottom right")
    fig.update_yaxes(title_text="Sea Surface Temperature (°C)", secondary_y=False)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-card" style="text-align: center;">
        <h4>🌡️ <strong>Warm or cool waters?</strong></h4>
        <p>SST anomalies above the climatological mean indicate the onset of El Niño, while those below suggest La Niña.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🌀 Southern Oscillation Index (SOI): Pressure Patterns Driving ENSO")

    st.markdown("""
    The Southern Oscillation Index (SOI) is a standardized scale which tracks sea level pressure differences across the Pacific, specifically between Tahiti and Darwin, Australia. These pressure differences steer wind patterns and can trigger or suppress ENSO events.
    """)

    fig_soi = px.line(df_filtered, x="Date", y="SOI")
    fig_soi.update_layout(
        yaxis_title="Southern Oscillation Index",
        template="plotly_dark",
        height=400,
        margin=dict(t=30, b=30)
    )
    fig_soi.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="La Niña Conditions", annotation_position="top right")
    fig_soi.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="El Niño Conditions", annotation_position="bottom right")
    fig_soi.add_hrect(y0=0, y1=3.2, line_width=0, fillcolor="blue", opacity=0.1)
    fig_soi.add_hrect(y0=-3.2, y1=0, line_width=0, fillcolor="red", opacity=0.1)

    fig_soi.add_trace(
        go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["SOI"],
            name="SOI",
            showlegend=False,
            line=dict(color='skyblue', width=2),
            hovertemplate="<b>%{x}</b><br>SOI: %{y:.2f}<extra></extra>"
        ),
        secondary_y=False,
    )

    st.plotly_chart(fig_soi, use_container_width=True)

    st.markdown("""
    <div class="insight-card" style="text-align: center;">
        <h4>🌀 <strong>High or low pressure?</strong></h4>
        <p>A deeply negative SOI value generally supports El Niño conditions, while strong positive values align with the onset of La Niña.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📈 Oceanic Niño Index (ONI): Quantifying ENSO")

    st.markdown("""
    The Oceanic Niño Index (ONI) tracks the three-month moving average of SST anomalies in the Niño 3.4 region. Widely regarded as the gold standard, it serves as the primary benchmark for identifying and classifying ENSO events.
    """)
    fig_oni = go.Figure()

    fig_oni.add_trace(
        go.Scatter(
            x=df_filtered["Date"],
            y=df_filtered["ONI"],
            name="ONI",
            line=dict(color='skyblue', width=2),
            hovertemplate="<b>%{x|%b %Y}</b><br>ONI: %{y:.2f}<extra></extra>"
        )
    )

    fig_oni.update_layout(
        xaxis_title="Date",
        yaxis_title="Oceanic Niño Index",
        height=400,
        yaxis_range=[-2.7, 2.7],
        template="plotly_dark",
        margin=dict(t=30, b=30)
    )

    fig_oni.add_hline(y=0.5, line_dash="dot", line_color="red",
                      annotation_text="El Niño Threshold", annotation_position="top right")
    fig_oni.add_hline(y=-0.5, line_dash="dot", line_color="blue",
                      annotation_text="La Niña Threshold", annotation_position="bottom right")

    fig_oni.add_hrect(y0=0.5, y1=2.7, line_width=0, fillcolor="red", opacity=0.1)
    fig_oni.add_hrect(y0=-2.7, y1=-0.5, line_width=0, fillcolor="blue", opacity=0.1)

    st.plotly_chart(fig_oni, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <h4>📈 <strong>Positive or negative anomaly?</strong></h4> 
        <p>If ONI exceeds +0.5 for months, El Niño is underway; if it remains below -0.5, La Niña is active. Values between these thresholds indicate neutral conditions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    phase_numeric = df_filtered["ENSO_Phase"].map({"El Niño": 1, "Neutral": 0, "La Niña": -1})

    fig_phases = go.Figure()

    colors = {"La Niña": "#3b82f6", "Neutral": "gray", "El Niño": "#ef4444"}
    st.markdown("### 🗓️ Seasonal Behavior of ENSO")

    st.markdown("""
    Some months are more likely to host El Niño or La Niña events. This bar chart reveals the seasonal rhythm of each ENSO phase.
    """)
    df_filtered['Month'] = df_filtered['Date'].dt.month
    seasonal_patterns = df_filtered.groupby(['Month', 'ENSO_Phase']).size().unstack(fill_value=0)

    fig_seasonal = px.bar(
        seasonal_patterns.reset_index().melt(id_vars='Month', var_name='Phase', value_name='Count'),
        x='Month', y='Count', color='Phase',
        color_discrete_map=colors
    )

    fig_seasonal.update_layout(
        margin=dict(t=30, b=30),
    )

    fig_seasonal.update_xaxes(
        tickmode='array',
        tickvals=list(range(1, 13)),
        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    )

    st.plotly_chart(fig_seasonal, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <h4>🗓️ <strong>Which seasons matter most?</strong></h4> 
        <p>ENSO seems to show asymmetric seasonal patterns — El Niño events cluster in late fall/winter months, while La Niña strengthens in winter and tends to persist longer. Neutral conditions are more common during spring/summer transition periods when phase changes typically occur.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "🔬 Model Overview":
    st.markdown("""
    <div class="story-card">
       <h2>🔬 Understanding the Baseline Model</h2>
       <p>ENSOcast is a climate prediction tool that leverages a <strong>Random Forest Classifier</strong> to forecast monthly ENSO phases based on patterns in ocean temperatures, atmospheric pressure, and seasonal cycles.</p>
    
    <p class="description"><strong>Model Targets —</strong> ENSO phase classifications derived from the <strong>Oceanic Niño Index (ONI)</strong>:</p>
    
    <ul class="description" style="text-align: left;">
        <li class="description"><strong>El Niño</strong></li>
        <li class="description"><strong>La Niña</strong></li>
        <li class="description"><strong>Neutral</strong></li>
    </ul>
    
    <p class="description"><strong>Key Input Features —</strong> Critical climate indicators that capture ocean-atmosphere dynamics:</p>
    
    <ul class="description" style="text-align: left;">
       <li><strong>Sea Surface Temperatures (SST)</strong> in the Niño 3.4 region, the primary oceanic signal of ENSO variability</li>
       <li class="description"><strong>Southern Oscillation Index (SOI)</strong>, measuring atmospheric pressure differences across the Pacific</li>
    </ul>
    
    <p class="description"><strong>Feature Engineering —</strong> To capture temporal patterns and seasonal dependencies, ENSOcast applies several preprocessing techniques:</p>
    
    <ul class="description" style="text-align: left;">
       <li>Calculating <strong>SST anomalies</strong> by eliminating the long-term climatological baseline</li>
       <li>Creating <strong>lagged variables</strong> (1–3 months) for both SOI and SST anomalies to capture predictive relationships</li>
       <li>Encoding <strong>seasonal cycles</strong> using sine and cosine transformations of calendar months</li>
    </ul>
    
    <p class="description"><strong>Validation Approach —</strong> ENSOcast is trained on the earliest 80% of historical ENSO data and achieves <strong>82% accuracy</strong> when predicting the most recent 20% of observations — compared to a majority-class baseline of ~40%, representing a 2× improvement over naive prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col3:
        color = "#94a3b8"
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
            <h3>🎯 Overall Accuracy</h3>
            <h2>{accuracy:.0%}</h2>
            <p>Correct predictions</p>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
            <h3>📚 Training Data</h3>
            <h2>{training_data_size:,}</h2>
            <p>Months trained on</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
            <h3>🧪 Testing Data</h3>
            <h2>{total_predictions:,}</h2>
            <p>Months tested on</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🧮 Confusion Matrix")

    st.markdown("""
    The confusion matrix reveals how accurately the baseline model identifies each ENSO phase. Diagonal cells show correct predictions for El Niño, Neutral, and La Niña months, while off-diagonal values highlight where the model confuses one phase for another.
    """)

    cm = confusion_matrix(df["True_Phase"], df["Predicted_Phase"], labels=["El Niño", "Neutral", "La Niña"])

    fig_cm = px.imshow(cm,
                       labels=dict(x="Predicted Phase", y="Actual Phase", color="Count"),
                       x=["🔴 El Niño", "⚪ Neutral", "🔵 La Niña"],
                       y=["🔴 El Niño", "⚪ Neutral", "🔵 La Niña"],
                       color_continuous_scale="Reds")

    fig_cm.update_layout(
        height=400,
        margin=dict(t=30, b=30)
    )

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            fig_cm.add_annotation(x=j, y=i, text=str(cm[i][j]),
                                showarrow=False, font_size=16, font_color="white" if cm[i][j] > cm.max()/2 else "black")

    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("""
    <div class="insight-card" style="text-align: center;">
        <h4>🧮 <strong>Accurate or confused?</strong></h4>
        <p>The baseline model predicts ENSO phases well overall. Neutral is classified most accurately, while La Niña shows some confusion with Neutral.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Detailed Classification Report")
    st.markdown("""
    To further evaluate baseline model performance, the classification report breaks down precision, recall, and F1-scores for each ENSO phase. This helps quantify how well the model identifies El Niño, Neutral, and La Niña conditions individually.
    """)

    report = classification_report(df["True_Phase"], df["Predicted_Phase"], output_dict=True)

    phases = ["El Niño", "Neutral", "La Niña"]
    phase_colors = {"El Niño": "#f6416c", "Neutral": "#94a3b8", "La Niña": "#3b82f6"}
    phase_emojis = {"El Niño": "🔴", "Neutral": "⚪", "La Niña": "🔵"}

    for phase in phases:
        if phase in report:
            precision = report[phase]['precision']
            recall = report[phase]['recall']
            f1 = report[phase]['f1-score']
            support = int(report[phase]['support'])

            st.markdown(f"""
            <div class="insight-card" style="background: linear-gradient(135deg, {phase_colors[phase]}20, {phase_colors[phase]}40);">
                <h4>{phase_emojis[phase]} {phase} Performance</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                    <div>
                        <strong>Precision:</strong> {precision:.0%}<br>
                        <small>When the model predicts "{phase}", it is correct {precision:.0%} of the time.</small>
                    </div>
                    <div>
                        <strong>Recall:</strong> {recall:.0%}<br>
                        <small>The model correctly identifies {recall:.0%} of actual {phase} months.</small>
                    </div>
                    <div>
                        <strong>F1-Score:</strong> {f1:.0%}<br>
                        <small>This is a balanced metric combining precision and recall.</small>
                    </div>
                    <div>
                        <strong>Sample Size:</strong> {support}<br>
                        <small>In the testing dataset, {support} months were labeled as {phase}.</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### ⚖️ Feature Importance")

    st.markdown("""
    Feature importance assigns weights to the influence of each input variable on the baseline model's ENSO predictions.
    """)

    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig_importance = px.bar(importance_df, y="Feature", x="Importance",
                              orientation="h")
        fig_importance.update_layout(
            height=400,
            margin=dict(t=30, b=30)
        )

        st.plotly_chart(fig_importance, use_container_width=True)

        st.markdown("""
        <div class="insight-card" style="text-align: center;">
            <h4>⚖️ <strong>Which features matter most?</strong></h4>
            <p>The baseline model is driven most by lagged SST anomalies, especially those from 2, 3, and 1 month ago. Seasonal signals and SOI data offer valuable supporting context.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔧 Train Baseline Model":
    st.markdown("""
    <div class="story-card">
        <h2>🔧 Customize Baseline Model Training</h2>
        <p>Leverage the feature engineering of ENSOcast to experiment with ENSO phases and historical time periods as you train a custom <strong>Random Forest</strong> model and evaluate its predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        years = st.slider("Select Year Range", 1982, 2024, (2000, 2020),
                         help="Drag to select the time period for training",
                          key="custom_model_years_slider")

    with col2:
        selected_phases = st.multiselect(
            "Select ENSO Phases",
            ["La Niña", "Neutral", "El Niño"],
            default=["La Niña", "Neutral", "El Niño"],
            help="Select which ENSO phases to include in training",
            key="analysis_select_slider"
        )

    label_map = LABEL_MAP

    df["True_Phase"] = df["ENSO_Label"].map(label_map)

    if st.button("🔬️ Train Your Model", type="primary", use_container_width=True):

        filtered_df = df[
            (df["Date"].dt.year >= years[0]) &
            (df["Date"].dt.year <= years[1]) &
            (df["True_Phase"].isin(selected_phases))
        ]

        if len(filtered_df) < 30:
            st.warning("⚠️ Not enough data for reliable training. Try expanding your date range or including more phases.")
            st.stop()

        with st.spinner("🧠 Training in progress..."):
            filtered_df = filtered_df.sort_values("Date").reset_index(drop=True)

            # Use 80/20 split, but ensure at least 10 test samples
            test_size = max(10, int(0.2 * len(filtered_df)))
            split_idx = len(filtered_df) - test_size

            X_custom = filtered_df[feature_cols]
            y_custom = filtered_df["ENSO_Label"]

            X_train_custom = X_custom.iloc[:split_idx]
            y_train_custom = y_custom.iloc[:split_idx]
            X_test_custom = X_custom.iloc[split_idx:]
            y_test_custom = y_custom.iloc[split_idx:]

            custom_model = RandomForestClassifier(random_state=42, n_estimators=100)
            custom_model.fit(X_train_custom, y_train_custom)
            y_pred_custom = custom_model.predict(X_test_custom)

            custom_accuracy = accuracy_score(y_test_custom, y_pred_custom)

            label_map = LABEL_MAP
            y_test_phases = [label_map[i] for i in y_test_custom]
            y_pred_phases = [label_map[i] for i in y_pred_custom]

            st.markdown("""
            <div class="story-card">
                <h2>🎉 Training Complete!</h2>
                <p>Your custom ENSO model has completed training. Check out its performance on unseen data.</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            color = "#94a3b8"

            with col3:
                st.markdown(f"""
                <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
                    <h3>🎯 Overall Accuracy</h3>
                    <h2>{custom_accuracy:.0%}</h2>
                    <p>Correct predictions</p>
                </div>
                """, unsafe_allow_html=True)

            with col1:
                training_size = len(X_train_custom)
                st.markdown(f"""
                <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
                    <h3>📚 Training Data</h3>
                    <h2>{training_size:,}</h2>
                    <p>Months trained on</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                test_size = len(X_test_custom)
                st.markdown(f"""
                <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40); padding: 1rem; border-radius: 10px;">
                    <h3>🧪 Testing Data</h3>
                    <h2>{test_size:,}</h2>
                    <p>Months tested on</p>
                </div>
                """, unsafe_allow_html=True)

            baseline_accuracy = accuracy

            if abs(custom_accuracy - baseline_accuracy) <= 0.015:
                comparison = f"Your custom model reached <b>{custom_accuracy:.0%} accuracy</b>, closely matching the ENSOcast baseline performance of {baseline_accuracy:.0%}. This demonstrates consistent predictive capability across different training configurations."
                comparison_emoji = "🔬️ "
            elif custom_accuracy > baseline_accuracy:
                comparison = f"Your custom model achieved <b>{custom_accuracy:.0%} accuracy</b>, surpassing the ENSOcast baseline model at {baseline_accuracy:.0%}. This suggests your selected time period and ENSO phases provided particularly strong training signals for the Random Forest algorithm."
                comparison_emoji = "🔬️ "
            else:
                comparison = f"Your custom model achieved <b>{custom_accuracy:.0%} accuracy</b> compared to the ENSOcast baseline's {baseline_accuracy:.0%}. This difference often reflects the challenging nature of your selected time period or phase combinations, offering valuable insights into ENSO predictability patterns."
                comparison_emoji = "🔬️ "

            st.markdown(f"""
            <div class="insight-card" style="text-align: center;">
                <h4>{comparison_emoji} <strong>Model Performance Analysis</strong></h4>
                <p>{comparison}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            ***NOTE:** 
            The baseline ENSOcast model (82% accuracy) is trained on the complete 40+ year dataset (1982-2024) and includes climate variability, which can lower overall accuracy compared to models trained on more focused time periods.*
            """)

            st.markdown("---")

            st.markdown("### 🧮 Confusion Matrix")
            st.markdown("""
            The confusion matrix reveals how well your custom model distinguishes each ENSO phase. Diagonal cells represent correct predictions for El Niño, Neutral, and La Niña months, while off-diagonal entries highlight misclassifications.
            """)
            render_confusion_matrix(y_test_custom, y_pred_custom)

            st.markdown("---")
            st.markdown("### 📋 Detailed Classification Report")
            st.markdown("""
            To help better understand how your model functions, this classification report breaks down precision, recall, and F1-scores for each ENSO phase. It shows how well your model identifies El Niño, Neutral, and La Niña conditions individually.
            """)
            render_classification_report(y_test_phases, y_pred_phases, selected_phases)

            st.markdown("---")

            if hasattr(custom_model, 'feature_importances_'):
                st.markdown("### ⚖️ Feature Importance")

                st.markdown("""
                Feature importance reveals which climate indicators your custom model relies upon most heavily for making ENSO predictions.
                """)

                importance_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": custom_model.feature_importances_
                }).sort_values("Importance", ascending=True)

                fig_importance = px.bar(importance_df, y="Feature", x="Importance",
                              orientation="h")
                fig_importance.update_layout(
                    height=400,
                    margin=dict(t=30, b=30)
                )

                st.plotly_chart(fig_importance, use_container_width=True)

                top_feature = importance_df.iloc[-1]
                st.markdown(f"""
                <div class="insight-card" style="text-align: center;">
                    <h4>⚖️ <strong>Which features matter most?</strong></h4>
                    <p>Your Random Forest model found <strong>{top_feature['Feature']}</strong> to be the most important factor, accounting for {top_feature['Importance']:.0%} of its decision-making process.</p>
                </div>
                """, unsafe_allow_html=True)

elif page == "🛠️ Train Advanced Models":
    st.markdown("""
    <div class="story-card">
       <h2>🛠️ Build and Compare Advanced Models</h2>
       <p>Experiment with the advanced feature-engineering of ENSOcast to train and assess a range of models, including <strong>Enhanced Random Forest</strong>, <strong>XGBoost</strong>, <strong>1D CNNs</strong>, <strong>LSTMs</strong>, and <strong>Ensemble Learning</strong>, on their ability to predict ENSO phases across historical time periods.</p>
    
    <p class="description"><strong>Advanced Feature Engineering —</strong> To capture deeper temporal trends, seasonal patterns, and interaction terms, ENSOcast employs additional feature engineering steps:</p>
    
    <ul class="description" style="text-align: left;">
       <li>Calculating <strong>rolling SST and SOI averages</strong> over 3 and 6 months to smooth short- and medium-term variability</li>
       <li>Computing <strong>SST and SOI trends</strong> to capture month-to-month changes indicative of ENSO phase transitions</li>
       <li>Creating an <strong>SST × SOI interaction</strong> feature to capture combined ocean–atmosphere effects</li>
       <li>Adding <strong>seasonal signals</strong> for December–February and June–August to reflect key ENSO-related seasons</li>
       <li>Including <strong>OHC anomalies</strong> to incorporate subsurface thermal energy signals predictive of El Niño or La Niña</li>
    </ul>
    """, unsafe_allow_html=True)

    try:
        advanced_models_available = True
    except ImportError as e:
        st.error(f"🚨 Advanced models require additional libraries: {str(e)}")
        st.info("💡 Please install: pip install xgboost tensorflow")
        advanced_models_available = False

    if not advanced_models_available:
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        years = st.slider("Select Year Range", 1982, 2024, (2000, 2020),
                         help="Drag to select the time period for training")

    with col2:
        selected_phases = st.multiselect(
            "Select ENSO Phases",
            ["La Niña", "Neutral", "El Niño"],
            default=["La Niña", "Neutral", "El Niño"],
            help="Select which ENSO phases to include in training"
        )

    with col3:
        model_type = st.selectbox(
            "Select ML Algorithm",
            ["Enhanced Random Forest", "XGBoost", "1D CNN", "LSTM", "Ensemble Learning"],
            help="Choose the machine learning algorithm to train"
        )

    with st.expander("⚙️ Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Testing Data Size (%)", 10, 40, 20,
                                help="Percentage of data to use for testing")
        with col2:
            sequence_length = st.slider("Sequence Length (months)", 3, 18, 6,
                                      help="Number of months to look back for CNN/LSTM")

    if st.button("🔬 Train Advanced Model", type="primary", use_container_width=True):

        label_map = LABEL_MAP
        df["True_Phase"] = df["ENSO_Label"].map(label_map)

        filtered_df = df[
            (df["Date"].dt.year >= years[0]) &
            (df["Date"].dt.year <= years[1]) &
            (df["True_Phase"].isin(selected_phases))
        ]

        if len(filtered_df) < 50:
            st.warning("⚠️ Insufficient data for advanced models. Try expanding your date range or including more phases.")
            st.stop()

        with st.spinner("🧠 Training advanced model..."):

            filtered_df = filtered_df.sort_values("Date").reset_index(drop=True)

            feature_cols = [
                "SST_Anomaly", "SOI", "SOI_lag_1", "SOI_lag_2", "SOI_lag_3",
                "SST_Anomaly_lag_1", "SST_Anomaly_lag_2", "SST_Anomaly_lag_3",
                "month_sin", "month_cos"
            ]

            enhanced_features = [
                "SST_Anomaly_rolling_3m", "SOI_rolling_3m", "SST_Anomaly_rolling_6m",
                "SST_SOI_interaction", "SST_trend", "SOI_trend", "is_djf", "is_jja"
            ]

            for feat in enhanced_features:
                if feat in filtered_df.columns and not filtered_df[feat].isna().all():
                    feature_cols.append(feat)

            if "OHC_Anomaly" in filtered_df.columns and not filtered_df["OHC_Anomaly"].isna().all():
                feature_cols.append("OHC_Anomaly")

            X_custom = filtered_df[feature_cols].ffill().bfill()
            y_custom = filtered_df["ENSO_Label"]

            split_idx = int((100 - test_size) / 100 * len(filtered_df))
            split_idx = max(split_idx, int(0.6 * len(filtered_df)))

            X_train_custom = X_custom.iloc[:split_idx]
            y_train_custom = y_custom.iloc[:split_idx]
            X_test_custom = X_custom.iloc[split_idx:]
            y_test_custom = y_custom.iloc[split_idx:]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_custom)
            X_test_scaled = scaler.transform(X_test_custom)

            model_results = {}

            def create_sequences(X, y, seq_len):
                sequences, labels = [], []
                for i in range(seq_len, len(X)):
                    sequences.append(X[i-seq_len:i])
                    labels.append(y.iloc[i] if hasattr(y, 'iloc') else y[i])
                return np.array(sequences), np.array(labels)

            if model_type == "Enhanced Random Forest":
                model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
                model.fit(X_train_custom, y_train_custom)
                y_pred = model.predict(X_test_custom)
                model_results["Random Forest"] = accuracy_score(y_test_custom, y_pred)
                final_predictions = y_pred

            elif model_type == "XGBoost":
                model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                model.fit(X_train_scaled, y_train_custom, verbose=False)
                y_pred = model.predict(X_test_scaled)
                model_results["XGBoost"] = accuracy_score(y_test_custom, y_pred)
                final_predictions = y_pred

            elif model_type == "1D CNN":
                if len(X_train_scaled) < sequence_length + 10:
                    st.warning(f"⚠️ Insufficient data for sequence length {sequence_length}. Using sequence length of {min(6, len(X_train_scaled)//4)}")
                    sequence_length = min(6, len(X_train_scaled)//4)

                X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_custom, sequence_length)
                X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_custom, sequence_length)

                model = Sequential([
                    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, len(feature_cols))),
                    BatchNormalization(),
                    Conv1D(filters=32, kernel_size=3, activation='relu'),
                    Dropout(0.3),
                    GlobalMaxPooling1D(),
                    Dense(50, activation='relu'),
                    Dropout(0.5),
                    Dense(len(selected_phases), activation='softmax')
                ])

                model.compile(optimizer=Adam(learning_rate=0.001),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
                ]

                model.fit(X_train_seq, y_train_seq,
                         validation_data=(X_test_seq, y_test_seq),
                         epochs=100, batch_size=16, callbacks=callbacks, verbose=0)

                y_pred = np.argmax(model.predict(X_test_seq), axis=1)
                model_results["1D CNN"] = accuracy_score(y_test_seq, y_pred)
                final_predictions = y_pred
                y_test_custom = y_test_seq

            elif model_type == "LSTM":
                if len(X_train_scaled) < sequence_length + 10:
                    st.warning(f"⚠️ Insufficient data for sequence length {sequence_length}. Using sequence length of {min(6, len(X_train_scaled)//4)}")
                    sequence_length = min(6, len(X_train_scaled)//4)

                X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_custom, sequence_length)
                X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_custom, sequence_length)

                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(sequence_length, len(feature_cols))),
                    Dropout(0.3),
                    LSTM(50, return_sequences=False),
                    Dropout(0.3),
                    Dense(25, activation='relu'),
                    Dense(len(selected_phases), activation='softmax')
                ])

                model.compile(optimizer=Adam(learning_rate=0.001),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
                ]

                model.fit(X_train_seq, y_train_seq,
                         validation_data=(X_test_seq, y_test_seq),
                         epochs=100, batch_size=16, callbacks=callbacks, verbose=0)

                y_pred = np.argmax(model.predict(X_test_seq), axis=1)
                model_results["LSTM"] = accuracy_score(y_test_seq, y_pred)
                final_predictions = y_pred
                y_test_custom = y_test_seq

            elif model_type == "Ensemble Learning":
                ensemble_predictions = []

                rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
                rf_model.fit(X_train_custom, y_train_custom)
                rf_pred = rf_model.predict(X_test_custom)
                model_results["Random Forest"] = accuracy_score(y_test_custom, rf_pred)
                ensemble_predictions.append(rf_pred)

                xgb_model = xgb.XGBClassifier(objective='multi:softprob', n_estimators=200, random_state=42)
                xgb_model.fit(X_train_scaled, y_train_custom, verbose=False)
                xgb_pred = xgb_model.predict(X_test_scaled)
                model_results["XGBoost"] = accuracy_score(y_test_custom, xgb_pred)
                ensemble_predictions.append(xgb_pred)

                from scipy.stats import mode
                ensemble_pred = mode(np.array(ensemble_predictions), axis=0)[0].flatten()
                model_results["Ensemble"] = accuracy_score(y_test_custom, ensemble_pred)
                final_predictions = ensemble_pred

            custom_accuracy = accuracy_score(y_test_custom, final_predictions)

            st.markdown("""
            <div class="story-card">
                <h2>🎉 Training Complete!</h2>
                <p>Your custom ENSO model has completed training. Check out its performance on unseen data.</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            color = "#94a3b8"

            with col1:
                training_size = len(X_train_custom)
                st.markdown(f"""
                <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40);">
                    <h3>📚 Training Data</h3>
                    <h2>{training_size:,}</h2>
                    <p>Months trained on</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                test_size = len(y_test_custom)
                st.markdown(f"""
                <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40);">
                    <h3>🧪 Testing Data</h3>
                    <h2>{test_size:,}</h2>
                    <p>Months tested on</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-container" style="background: linear-gradient(135deg, {color}20, {color}40);">
                    <h3>🎯 Overall Accuracy</h3>
                    <h2>{custom_accuracy:.0%}</h2>
                    <p>Correct predictions</p>
                </div>
                """, unsafe_allow_html=True)

            baseline_accuracy = accuracy

            if abs(custom_accuracy - baseline_accuracy) <= 0.015:
                comparison = f"Your custom model reached <b>{custom_accuracy:.0%} accuracy</b>, closely matching the ENSOcast baseline performance of {baseline_accuracy:.0%}. This demonstrates consistent predictive capability across different training configurations."
                comparison_emoji = "🔬️ "
            elif custom_accuracy > baseline_accuracy:
                comparison = f"Your custom model achieved <b>{custom_accuracy:.0%} accuracy</b>, surpassing the ENSOcast baseline model at {baseline_accuracy:.0%}. This suggests your selected time period and ENSO phases provided particularly strong training signals for the Random Forest algorithm."
                comparison_emoji = "🔬️ "
            else:
                comparison = f"Your custom model achieved <b>{custom_accuracy:.0%} accuracy</b> compared to the ENSOcast baseline's {baseline_accuracy:.0%}. This difference often reflects the challenging nature of your selected time period or phase combinations, offering valuable insights into ENSO predictability patterns."
                comparison_emoji = "🔬️ "

            st.markdown(f"""
            <div class="insight-card" style="text-align: center;">
                <h4>{comparison_emoji} <strong>Model Performance Analysis</strong></h4>
                <p>{comparison}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            ***NOTE:** 
            The CNN and LSTM models predict ENSO phases using multi-month sequences. Accuracy is computed on the last month of each sequence, reflecting sequence-level predictions rather than all individual months, which slightly reduces the number of test points and shifts alignment.*
            """)

            st.markdown("---")
            st.markdown("### 🧮 Confusion Matrix")
            st.markdown("""
            The confusion matrix reveals how well your custom model distinguishes each ENSO phase. Diagonal cells represent correct predictions for El Niño, Neutral, and La Niña months, while off-diagonal entries highlight misclassifications.
            """)
            y_test_phases = [LABEL_MAP[i] for i in y_test_custom]
            y_pred_phases = [LABEL_MAP[i] for i in final_predictions]
            render_confusion_matrix(y_test_custom, final_predictions)

            st.markdown("---")
            st.markdown("### 📋 Detailed Classification Report")
            st.markdown("""
            To help better understand how your model functions, this classification report breaks down precision, recall, and F1-scores for each ENSO phase. It shows how well your model identifies El Niño, Neutral, and La Niña conditions individually.
            """)
            render_classification_report(y_test_phases, y_pred_phases, selected_phases)

            if model_type in ["Enhanced Random Forest", "XGBoost"] or model_type == "Ensemble Learning":
                st.markdown("---")
                st.markdown("### ⚖️ Feature Importance")

                st.markdown("""
                Feature importance reveals which climate indicators your custom model relies upon most heavily for making ENSO predictions.
                """)

                if model_type == "Enhanced Random Forest":
                    importance = model.feature_importances_
                elif model_type == "XGBoost":
                    importance = model.feature_importances_
                else:
                    importance = (rf_model.feature_importances_ + xgb_model.feature_importances_) / 2

                importance_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": importance
                }).sort_values("Importance", ascending=True)

                fig_importance = px.bar(importance_df, y="Feature", x="Importance", orientation="h")
                fig_importance.update_layout(height=400, margin=dict(t=30, b=30))
                st.plotly_chart(fig_importance, use_container_width=True)

                top_feature = importance_df.iloc[-1]
                st.markdown(f"""
                <div class="insight-card" style="text-align: center;">
                    <h4>⚖️ <strong>Which features matter most?</strong></h4>
                    <p>Your {model_type} model found <strong>{top_feature['Feature']}</strong> to be the most important factor, accounting for {top_feature['Importance']:.0%} of its decision-making process.</p>
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
           border-radius: 10px; color: white; margin-top: 2rem;">
    <h3>🌊 ENSOcast</h3>
    <p>Decoding El Niño–Southern Oscillation</p>
    <p><em>Created by Dylan Dsouza</em></p>
    <small>Data source: NOAA | Built with Streamlit, plotly, and scikit-learn</small>
</div>
""", unsafe_allow_html=True)
