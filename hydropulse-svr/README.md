# HydroPulse-SVR
## El Niño-Based KRS Reservoir Inflow Deficit Prediction Using Support Vector Regression

**College Project | Machine Learning for Hydropower Risk Assessment**  
Cauvery Basin, India | KRS Reservoir | 2011–2020

---

## Project Objective

Predict **next-month KRS reservoir inflow deficit (%)** as a hydropower-risk indicator
using ENSO climate signals (ONI, SOI) and current hydrological observations.

The study establishes the climate chain:
> **El Niño → Weakened South-West Monsoon → Reduced Cauvery Basin Inflows → Inflow Deficit → Hydropower Generation Risk**

KRS is primarily an irrigation dam (4.8 MW installed capacity). Actual hydropower
generation data is not available, so inflow deficit serves as a scientifically
defensible surrogate for downstream generation risk (Shivanasamudra 96 MW, Mettur 200 MW).

---

## Methodology

| Step | Description |
|------|-------------|
| **Data collection** | Merged 3 open-source datasets: ENSO indices (ENSOcast), KRS daily reservoir hydrology (IndiaWRI), monthly catchment rainfall (World Bank ERA5) |
| **Data processing** | Aggregated daily reservoir data to monthly; combined with monthly ENSO + rainfall |
| **Target variable** | Next-month inflow deficit (%) computed from training-year climatological normals |
| **Feature engineering** | ONI lags (2, 3 months), lagged inflow, circular seasonal encoding |
| **Model** | SVR with RBF kernel; hyperparameters tuned via GridSearchCV + TimeSeriesSplit |
| **Split** | Strict chronological split — no shuffling, no data leakage |
| **Evaluation** | RMSE, MAE, R² compared against Linear Regression, Ridge, Random Forest, and Persistence baselines |

---

## How to Run

### Prerequisites
Python 3.9+ with pip.

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model
```bash
python train_model.py
```
This runs the full pipeline (data loading → feature engineering → SVR training →
evaluation) and saves the model and results to `models/`.

### Step 3 — Launch the dashboard
```bash
streamlit run dashboard.py
```
Open **http://localhost:8501** in your browser.

---

## Project Workflow

```
3 Reference Repos (read-only)
        |
        |-- ENSOcast/merged_enso.csv            (ENSO indices: ONI, SOI, SST)
        |-- reservoir-water-level-forecasting/  (KRS WRI daily data + WB rainfall)
        |
        v
data_pipeline.py
        |-- load_enso()          Load 516-month ENSO index series
        |-- load_reservoir()     Load 3,653-day KRS hydrology, aggregate to monthly
        |-- load_wb_rainfall()   Load World Bank monthly catchment rainfall
        |-- merge_all()          Inner join on monthly period key (120 months)
        |-- engineer_features()  ENSO lags, inflow lag, seasonal encoding
        |-- build_target()       Inflow deficit % -- normals from TRAINING YEARS ONLY
        |-- clean_dataset()      Drop NaN rows, winsorize extreme values
        |
        v  (116 usable monthly rows)
        |
train_model.py
        |-- Chronological split  (train 57 / val 24 / test 35)
        |-- StandardScaler       Fitted on training set only
        |-- SVR (RBF kernel)     GridSearchCV + TimeSeriesSplit CV
        |-- Baselines            Linear Regression, Ridge, Random Forest, Persistence
        |-- Evaluate             RMSE, MAE, R² on held-out test set
        |-- Save                 svr_model.pkl, scaler.pkl, metrics.csv, predictions.csv
        |
        v
dashboard.py (Streamlit)
        |-- ENSO Monitor         ONI timeline, phase distribution, ONI vs deficit scatter
        |-- Hydrology            Inflow, storage, rainfall time series
        |-- Model Performance    Actual vs predicted, metrics table, residuals
        |-- Deficit Forecast     What-if sliders for scenario exploration
        |-- Risk Assessment      Risk gauge, heatmap, ENSO phase box plots
```

---

## Dataset Summary

| Item | Detail |
|------|--------|
| Total months | 120 (Jan 2011 – Dec 2020) |
| Usable (X, y) pairs | 116 (after removing lag NaN rows and last month) |
| Training set | 57 months (Apr 2011 – Dec 2015) |
| Validation set | 24 months (Jan 2016 – Dec 2017) |
| Test set | 35 months (Jan 2018 – Nov 2020) |
| ENSO source | ENSOcast merged_enso.csv (NOAA-based; 1982–2024) |
| Reservoir source | IndiaWRI KRS daily data (2011–2020) |
| Rainfall source | World Bank ERA5 monthly (1991–2020) |

---

## Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | ONI | Oceanic Niño Index (°C) — primary ENSO signal |
| 2 | SOI | Southern Oscillation Index — atmospheric ENSO signal |
| 3 | ONI_lag2 | ONI 2 months ago |
| 4 | ONI_lag3 | ONI 3 months ago (peak correlation +0.29 with inflow deficit) |
| 5 | rainfall_mm | Monthly catchment rainfall (World Bank ERA5) |
| 6 | inflow_mcm | Current-month KRS inflow (MCM) |
| 7 | inflow_lag1_mcm | Previous-month inflow (MCM) |
| 8 | storage_pct_frl | Reservoir storage as % of Full Reservoir Level |
| 9 | month_sin | sin(2π·month/12) — circular seasonal encoding |
| 10 | month_cos | cos(2π·month/12) — circular seasonal encoding |
| 11 | deficit_pct | Current-month inflow deficit % (autoregressive predictor) |

**Target:** `deficit_pct_next` = next-month inflow deficit (%)  
Positive → below-normal inflow → hydropower risk.  Negative → surplus.

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Kernel | RBF |
| C | Tuned via GridSearchCV (range: 0.5–500) |
| Epsilon | Tuned via GridSearchCV (range: 0.1–5.0) |
| Gamma | Tuned via GridSearchCV ("scale" or "auto") |
| CV strategy | TimeSeriesSplit (4 folds, gap=1 month) |
| Scaler | StandardScaler (fit on training set ONLY) |

---

## Results

### Test Set Performance (Jan 2018 – Nov 2020, n = 35)

| Model | RMSE (%) | MAE (%) | R² |
|-------|---------|---------|-----|
| SVR-RBF *(proposed)* | **65.5** | **53.0** | **0.29** |
| Linear Regression | 60.5 | 49.3 | 0.40 |
| Ridge | 60.6 | 49.5 | 0.39 |
| Random Forest | 69.1 | 57.6 | 0.21 |
| Persistence (naive) | 77.5 | 48.7 | 0.01 |

**Key observations:**
- SVR-RBF beats the naive persistence baseline (R² 0.29 vs 0.01), confirming that ENSO + hydrology features carry genuine predictive signal.
- Linear/Ridge models achieve slightly better RMSE on this test set. With only 57 training samples and a noisy hydrology target (std ≈ 60%), this is expected — linear models are more data-efficient in this regime.
- All trained models significantly outperform persistence on RMSE.
- Model correctly captures the El Niño direction: ONI=+1.2 → +38.8% predicted deficit; ONI=−1.0 → −59.8% (surplus).

---

## ENSO Signal Validation

| ENSO Phase | Mean Inflow Deficit | Interpretation |
|------------|--------------------|-|
| El Niño | +33.9% | Below-normal inflow — hydropower risk |
| Neutral | +4.7% | Near-normal inflow |
| La Niña | −12.6% | Above-normal inflow — surplus |

---

## Limitations

1. **Small dataset**: Only 57 training samples limit SVR to a small feature set.
2. **No generation data**: KRS has 4.8 MW installed capacity; inflow deficit is a surrogate — not actual generation.
3. **Winsorization**: Extreme flood months (Aug 2018: 14× normal inflow) are clipped at the 2nd/98th percentile.
4. **Single reservoir**: Results are specific to KRS; generalisation requires multi-reservoir data.
5. **10-year window**: 2011–2020 includes only one major El Niño event (2015–16); more events would improve robustness.

---

## Future Scope

- Expand to multi-reservoir Cauvery cascade (Harangi, Kabini, Hemavathi)
- Obtain actual monthly generation data from KPCL for direct validation
- Use 30+ years of ENSO + WRI data (1990–2024) to increase training samples
- Explore seasonal (3-month) forecasting lead times
- Compare with LSTM or Random Forest with proper validation on a larger dataset

---

## Conclusion

This project demonstrates a complete machine-learning pipeline for climate-driven
hydropower risk assessment. An SVR model trained on ENSO indices and KRS reservoir
hydrology successfully identifies months at risk of inflow deficit, achieving
R² = 0.29 on the unseen test set — significantly better than the naive persistence
baseline (R² = 0.01). The ENSO → monsoon → reservoir inflow chain is empirically
confirmed, with El Niño months averaging +33.9% inflow deficit versus La Niña months
at −12.6% surplus, consistent with published literature on ENSO effects on the
Cauvery Basin.

---

## Project Structure

```
hydropulse-svr/
├── config.py          Central configuration (paths, features, parameters)
├── data_pipeline.py   Data loading, merging, feature engineering, target builder
├── train_model.py     SVR + baselines: training, evaluation, model saving
├── dashboard.py       Streamlit interactive dashboard (5 pages)
├── requirements.txt   Python dependencies
├── README.md          This file
└── models/            Auto-created on first run
    ├── svr_model.pkl  Trained SVR model
    ├── scaler.pkl     StandardScaler (fit on training set)
    ├── master_data.csv  Merged and cleaned dataset (116 rows × 25 columns)
    ├── predictions.csv  Predictions across all splits
    └── metrics.csv    Model comparison table
```

---

## References / Data Sources

| Source | URL / Path | Description |
|--------|-----------|-------------|
| ENSOcast | `../ensocast/merged_enso.csv` | NOAA-based ENSO monthly indices (1982–2024) |
| IndiaWRI | `../reservoir-water-level-forecasting/...` | KRS daily reservoir data (WRI) |
| World Bank | Climate Change Knowledge Portal | ERA5 monthly catchment rainfall |
| NOAA | oceanic-nino-index | ONI definition and El Niño threshold (±0.5°C) |

---

*Submitted as part of a college ML project. Code is open-source and reproducible.*
