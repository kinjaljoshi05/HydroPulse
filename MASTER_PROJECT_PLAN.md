# MASTER PROJECT PLAN — v1.1
## Machine Learning Approach for Predicting Hydropower Generation Deficits During El Niño Years Using Support Vector Regression

**Project Codename:** HydroPulse-SVR  
**Lead ML Engineer & Software Architect:** [You]  
**Date:** September 28, 2026  
**Version:** 1.1 (updated post Phase -1 Data Feasibility Audit)  
**Status:** Implementation-Ready

> **v1.1 Summary of Changes:** Target variable changed from synthetic generation deficit to measurable inflow deficit. Feature set reduced from 40+ to 12 (sample-size constraint). OHC_Anomaly and TEMPERATURE_C excluded. MAPE removed from metrics. Train/val/test splits corrected to actual data. Fixed R²/RMSE thresholds removed. KRS hydropower capacity issue documented and resolved via target redesign. See `DATA_FEASIBILITY_REPORT.md` for full audit.

---

## TABLE OF CONTENTS

1. [Project Objective](#1-project-objective)
2. [Deep Analysis of All 3 Reference Projects](#2-deep-analysis-of-all-3-reference-projects)
3. [Cross-Project Comparison & Decisions](#3-cross-project-comparison--decisions)
4. [Final Causal Workflow](#4-final-causal-workflow)
5. [Datasets & Features](#5-datasets--features)
6. [ML Pipeline Design](#6-ml-pipeline-design)
7. [Deficit Formula & Target Variable](#7-deficit-formula--target-variable)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Final Folder Structure](#9-final-folder-structure)
10. [Module Responsibilities](#10-module-responsibilities)
11. [Dashboard Design](#11-dashboard-design)
12. [Implementation Phases](#12-implementation-phases)
13. [Exact First Coding Tasks](#13-exact-first-coding-tasks)
14. [Acceptance Criteria](#14-acceptance-criteria)
15. [Final Architectural Decision](#15-final-architectural-decision)

---

## 1. PROJECT OBJECTIVE

### Primary Research Question
> Can Support Vector Regression (SVR) reliably predict monthly **reservoir inflow deficits** (as a scientifically defensible surrogate for hydropower generation deficits) at the Cauvery Basin, India during El Niño years, using ENSO climate signals as upstream causal drivers?

### Scientific Hypothesis
El Niño events (measured by ONI/Niño3.4/SOI) cause anomalous reductions in South Asian monsoon rainfall, which propagate downstream through reduced catchment runoff → reduced reservoir inflows → reduced water available for turbines → generation deficit relative to historical normal.

We hypothesize that SVR can learn this causal chain from historical data with proper lag-feature engineering and time-series splitting.

> **v1.1 Note — Case Study Redefinition:** The 4 Cauvery reservoirs (KRS, Harangi, Hemavathi, Kabini) are **irrigation-primary dams** with tiny hydropower installations (4–9 MW each). No real plant-wise generation data exists in any of the 3 reference repositories. The generation proxy formula (P = η·ρ·g·Q·H) is invalid because WRI outflow includes spillway releases and overestimates KRS generation by 33×. The target has therefore been redesigned as **monthly inflow deficit (%)**, which directly represents the water resource deficit driving generation shortfalls in the downstream Cauvery cascade hydropower plants.

### Deliverables
1. **SVR Model** trained on ENSO+hydro features predicting monthly generation deficit
2. **Reproducible ML Pipeline** with proper time-series CV and no data leakage
3. **Comparison Baseline** (Linear Regression, Ridge, XGBoost) for academic rigor
4. **Streamlit Dashboard** with 5 panels: ENSO monitor, hydrology trends, model predictions, deficit forecast, risk alerts
5. **Research-grade documentation** suitable for academic submission

---

## 2. DEEP ANALYSIS OF ALL 3 REFERENCE PROJECTS

### 2.1 HydroIntel-AI (THDC Internship Project)
**Repository:** `HydroIntel-AI/`  
**Author:** Vinayak Dhyani, THDC India Limited Internship

#### What it actually does (from code, not README):
- **Data:** India-wide subdivision-level rainfall (annual) + multi-reservoir daily storage/level data
- **Data range:** Rainfall: 1901–present (4,091 records), Reservoir: 2024 only (~2,579 records)
- **Cleaning:** Simple `dropna()` + column selection — no imputation, no temporal alignment
- **"ML Model":** `LinearRegression` from scikit-learn trained on 4 features: YEAR, RAINFALL, STORAGE, LEVEL
- **Target (y):** A **fabricated formula**: `POWER = RAINFALL × 0.15 + STORAGE × 100 + LEVEL × 0.25` — this is **not real measured hydropower output**. The model learns a mathematical identity, not physics.
- **Forecasting:** Projects future years with linearly-spaced rainfall values (`np.linspace`) — essentially extrapolation of constants, not meaningful prediction.
- **Time-series split:** Uses `random_state=42` implied via LinearRegression fit on UTTARAKHAND only — no proper temporal split, no test set at all.
- **Data leakage:** The model is trained on all data then evaluated on synthetic future data — no out-of-sample test whatsoever.
- **Dashboard:** Functional Streamlit UI with Executive Panel, Region Comparison, Rainfall & Reservoir Analytics, AI Insight (rule-based text), Forecast Report download.

#### Critical Technical Flaws:
| Flaw | Severity | Impact |
|------|----------|--------|
| Target variable is synthetic formula, not real power data | CRITICAL | Model learns an algebraic identity, not physics |
| No real hydropower generation data | CRITICAL | Cannot predict actual deficits |
| No train/test split; no cross-validation | HIGH | No generalization guarantee |
| No ENSO/climate signals | HIGH | Cannot connect to El Niño |
| Temporal data (reservoir: 2024 only) not aligned with rainfall (1901+) | HIGH | Invalid join; model trains on unmatched data |
| Forecast uses linspace extrapolation of averages | MEDIUM | Trivial, not predictive |
| Risk classification uses hard-coded storage thresholds with no physical basis | MEDIUM | Arbitrary |

#### What to Reuse:
- ✅ **Dashboard UI pattern**: Multi-tab Streamlit layout with sidebar, KPI cards, region selector
- ✅ **Plotly chart patterns**: Line trends, bar comparisons, histogram distributions
- ✅ **Data validation logic**: Column presence checks, quality score computation
- ✅ **Report download button pattern**
- ✅ **India-wide IMD rainfall dataset** (`rainfall.csv`) — useful for regional rainfall signal

#### What to Reject:
- ❌ The entire `hydropower_model.py` and `hydropower_model.pkl` — mathematically meaningless
- ❌ The synthetic power formula `POWER = RAINFALL × 0.15 + STORAGE × 100 + LEVEL × 0.25`
- ❌ The "AI Insight" which is just rule-based text strings
- ❌ The forecast methodology (linspace extrapolation)
- ❌ Any reuse of the reservoir.csv (only 2024 data — too short for time-series modeling)

---

### 2.2 ENSOcast (Dylan Dsouza, UCSD)
**Repository:** `ensocast/`  
**Author:** Dylan Dsouza

#### What it actually does (from code):
- **Data:** `merged_enso.csv` — 517 monthly records (Jan 1982–Dec 2024) from NOAA sources
- **Features available in dataset:**
  - `SST` (Niño 3.4 region sea surface temperature)
  - `SST_Anomaly` (SST minus long-term monthly climatology)
  - `SOI` (Southern Oscillation Index — Tahiti minus Darwin pressure)
  - `ONI` (Oceanic Niño Index — 3-month running mean of SST_Anomaly)
  - `ENSO_Phase` / `ENSO_Label` (0=La Niña, 1=Neutral, 2=El Niño based on ONI thresholds)
  - **Lag features:** `SOI_lag_1/2/3`, `SST_Anomaly_lag_1/2/3`
  - **Seasonal encoding:** `month_sin`, `month_cos`
  - **Advanced features:** `SST_Anomaly_rolling_3m/6m`, `SOI_rolling_3m`, `SST_SOI_interaction`, `SST_trend`, `SOI_trend`, `is_djf`, `is_jja`, `OHC_Anomaly`
- **Task:** ENSO phase **classification** (3-class: El Niño/Neutral/La Niña), NOT regression
- **Baseline Model:** Random Forest Classifier (n_estimators=100), 82% accuracy
- **Advanced Models:** XGBoost, 1D CNN, LSTM, Ensemble (RF + XGBoost majority vote)
- **Time-series split:** Chronological 80/20 split (NO shuffle) — **correctly done**
- **Scaler:** StandardScaler on training data only, then transform test — **correctly done**
- **Validation:** Confusion matrix, precision/recall/F1 per phase — **appropriate for classification**
- **Data leakage check:** The lag features are pre-computed on the full dataset before splitting — **this is a subtle leakage issue** for the lag features at the split boundary (lag_1 of first test sample uses last train sample, which is acceptable), but the rolling averages at the split point slightly touch training data — minor issue.

#### Critical Assessment:
| Aspect | Assessment |
|--------|-----------|
| ENSO data quality | Excellent — real NOAA data, properly computed ONI, SOI |
| Feature engineering | Best among 3 projects — lag, rolling, interaction, seasonal |
| Time-series splitting | Correct (chronological, no shuffle) |
| Scaling protocol | Correct (fit on train, transform test) |
| Task formulation | Classification only — does NOT regress to quantity |
| Connection to hydrology | NONE — pure climate signal, no rainfall/reservoir linkage |
| Model variety | Good — RF, XGBoost, CNN, LSTM for comparison |

#### What to Reuse:
- ✅ **`merged_enso.csv` dataset** — gold standard ENSO features, 42 years of monthly data
- ✅ **Feature engineering framework**: lag computation, rolling means, seasonal encoding (sin/cos), SST×SOI interaction, OHC anomaly
- ✅ **Time-series splitting protocol**: chronological split, no shuffle
- ✅ **StandardScaler pattern**: fit on X_train only
- ✅ **ENSO phase visualization**: ONI threshold lines, phase coloring, seasonal bar charts
- ✅ **Confusion matrix + classification report** UI patterns (for ENSO phase classification sub-module)
- ✅ **`create_sequences()` function** for CNN/LSTM-style window creation (adapted for SVR)

#### What to Reject:
- ❌ The classification framing for the main SVR task (we need regression, not classification)
- ❌ CNN/LSTM as primary models (we keep SVR as primary; others as baselines)
- ❌ The "educational" framing — our project is research-grade prediction, not education
- ❌ The subtle data leakage in rolling features computed before split (we recompute with proper pipeline)

---

### 2.3 Reservoir Water-Level Forecasting (Cauvery Basin)
**Repository:** `reservoir-water-level-forecasting/`  
**Source:** Academic project on Bengaluru water crisis

#### What it actually does (from code):
- **Data:** 4 Cauvery basin reservoirs (Harangi, Hemavathi, KRS, Kabini)
  - **WRI data:** Daily, 2011–2020, water level only (limited)
  - **IndiaWRIS data:** Daily, 2001–2021, including: `PRESENT_STORAGE_TMC`, `INFLOW_CUSECS`, `OUTFLOW_CUSECS`, reservoir level
  - **processed/datafinal_krs.csv:** June 2018–May 2021, includes: `RES_LEVEL_FT`, `PRESENT_STORAGE_TMC`, `INFLOW_CUSECS`, `OUTFLOW_CUECS`, `Normal_rain`, `Actual_rain`, `Evapo-Transpiration`, `Volumetric_Soil_Moisture`, `RF_Actual_Level`
  - **Weather data:** Daily rainfall (1991–2021), groundwater (2014–2021), evapotranspiration (2018–2021), soil moisture (2018–2021)
- **Models implemented in `reservoir_functions.py`:**
  - **LSTM** (univariate, MinMaxScaled, lag-1 supervised window, loads pre-trained weights)
  - **SVR** — loads from pickle, univariate (`var1(t-1)` → `var1(t)`) — **SVR IS ALREADY HERE**
  - **XGBoost** — same univariate lag-1 setup
  - **Ensemble** — weighted by RMSE score (higher RMSE → lower weight)
- **Splitting:** `train_test_split(shuffle=False)` with `test_size=366` (last year) — **correct chronological split**
- **Scaling:** MinMaxScaler(0,1) fit on full data then transform — **DATA LEAKAGE: scaler fitted on full dataset including test set**
- **SVR setup:** Univariate only (storage(t-1) → storage(t)) — too simplistic, ignores rainfall, ENSO
- **Metrics:** RMSE at 90 days and 1 year
- **KRS notebook:** Has extensive preprocessing — daily to monthly aggregation, multi-feature dataset

#### Critical Assessment:
| Aspect | Assessment |
|--------|-----------|
| Data richness | Best among 3 — inflow, outflow, storage, rainfall, ET, soil moisture |
| SVR implementation | Present but naive — univariate lag-1 only, leaky scaler |
| Multi-feature potential | High — `datafinal_krs.csv` has 10 features |
| Temporal coverage | 2001–2021 (20 years daily) — excellent |
| ENSO linkage | NONE — no climate indices used |
| Scaler leakage | YES — MinMaxScaler fit on full dataset |
| Forecast horizon | 30–90 days ahead |

#### What to Reuse:
- ✅ **`series_to_supervised()` function** — generic lag-window creator for any multivariate dataset
- ✅ **`forecast()` dispatcher function** — clean model-agnostic forecasting API
- ✅ **Weighted ensemble logic** (RMSE-weighted combination)
- ✅ **IndiaWRIS dataset structure** — Inflow, Outflow, Storage, Level columns
- ✅ **`datafinal_krs.csv` as template** for merged hydro + weather dataset
- ✅ **Multi-model RMSE comparison** at different horizons (30d, 90d)
- ✅ **The concept of multivariate reservoir modeling** — inflow, outflow, rainfall, ET together

#### What to Reject:
- ❌ The MinMaxScaler fitted on full data — refit on train split only
- ❌ Univariate SVR approach — we use multivariate with ENSO features
- ❌ Relying on pre-trained .pkl and .h5 weights from paths not in repo — not reproducible
- ❌ The notebook-based approach for production — convert to modular scripts
- ❌ Daily granularity for the SVR model — aggregate to monthly for ENSO signal alignment

---

## 3. CROSS-PROJECT COMPARISON & DECISIONS

### 3.1 Comparison Matrix

| Dimension | HydroIntel-AI | ENSOcast | Reservoir Forecasting | Our Project Decision |
|-----------|--------------|----------|-----------------------|---------------------|
| **Target variable** | Synthetic formula (MW) | ENSO phase (class) | Water level (m) | Hydropower deficit (%) — real |
| **ENSO features** | None | ONI, SOI, SST_Anomaly | None | ONI + SOI + SST_Anomaly (from ENSOcast) |
| **Hydrology features** | Rainfall, Storage, Level | None | Inflow, Outflow, Storage, ET, SM | Both + ENSO lag features |
| **ML model** | LinearRegression | RandomForest (class.) | SVR/LSTM/XGBoost (univ.) | SVR-RBF (multivariate regression) |
| **Time-series split** | None (no test set) | Correct (80/20 chron.) | Correct (last 366d test) | TimeSeriesSplit + final holdout |
| **Scaler leakage** | N/A | Minor | YES (full-fit MinMax) | No leak: fit on train only |
| **Data quality** | Annual rainfall | Monthly ENSO | Daily hydrology | Monthly aggregated, aligned |
| **Dashboard** | Best UI | Good UI | Notebook-based | Streamlit, 5-panel |
| **Reproducibility** | Medium | High | Low (broken paths) | Full via config files |

### 3.2 Key Technical Decisions

**Decision 1 — Target Variable**
We use **real generation deficit** computed as:
```
deficit_pct(t) = (normal_generation(t) - actual_generation(t)) / normal_generation(t) × 100
```
Where `normal_generation(t)` = multi-year monthly climatological mean, and `actual_generation(t)` = measured/estimated from storage and head. If actual generation data is unavailable, we derive generation from:
```
P ≈ η × ρ × g × Q × H_net
```
And define deficit relative to climatological P.

**Decision 2 — Primary Model: SVR-RBF**
SVR with RBF kernel is chosen because:
- Non-linear relationships between ENSO signals and hydrology (proven in literature)
- Robust to outliers (uses ε-insensitive loss) — important for El Niño extremes
- Works well with normalized features on monthly tabular data
- Interpretable hyperparameters (C, γ, ε) for academic documentation
- The reservoir project already validated SVR for this domain

**Decision 3 — Feature Architecture**
We do NOT directly use any project's features as-is. We design from causal graph:
```
ENSO_t-3 → ENSO_t-2 → ENSO_t-1 → Rainfall_t → Inflow_t → Storage_t → Generation_t → Deficit_t
```
This causal chain dictates lag choices (3–6 months ENSO lag for South Asia monsoon effect).

**Decision 4 — Monthly Aggregation**
All daily data (reservoir) aggregated to monthly for ENSO alignment. This is the correct scientific temporal scale.

**Decision 5 — No Pre-trained Weights Dependency**
All models retrained from scratch within the pipeline. No reliance on external `.pkl`, `.h5` files.

---

## 4. FINAL CAUSAL WORKFLOW

```
╔══════════════════════════════════════════════════════════════════════╗
║                    CAUSAL CHAIN ARCHITECTURE                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ENSO INDICATORS (Monthly, 1982–present)                            ║
║  ┌─────────────────────────────────────────────┐                    ║
║  │ ONI (Oceanic Niño Index)                    │ ← NOAA CPC        ║
║  │ SOI (Southern Oscillation Index)            │ ← NOAA NCEI       ║
║  │ SST_Anomaly (Niño 3.4)                      │ ← NOAA OISST      ║
║  └────────────────────┬────────────────────────┘                    ║
║                       │ LAG: 2–6 months                             ║
║                       ▼                                              ║
║  RAINFALL (Monthly)                                                  ║
║  ┌─────────────────────────────────────────────┐                    ║
║  │ Regional rainfall anomaly (mm/month)         │ ← IMD/CWC        ║
║  │ Monsoon onset/withdrawal flag               │ ← Derived         ║
║  └────────────────────┬────────────────────────┘                    ║
║                       │ LAG: 0–2 months                             ║
║                       ▼                                              ║
║  RESERVOIR HYDROLOGY (Monthly)                                       ║
║  ┌─────────────────────────────────────────────┐                    ║
║  │ Inflow (MCM/month)                          │ ← IndiaWRIS/CWC   ║
║  │ Storage (TMC or % of FRL)                   │ ← WRI/IndiaWRIS   ║
║  │ Net Head (m) = Level - Dead Storage Level    │ ← Derived         ║
║  │ Evapotranspiration (mm)                     │ ← Remote sensing  ║
║  └────────────────────┬────────────────────────┘                    ║
║                       │                                              ║
║                       ▼                                              ║
║  HYDROPOWER GENERATION (Monthly)                                     ║
║  ┌─────────────────────────────────────────────┐                    ║
║  │ P = η × ρ × g × Q × H_net (kWh/month)      │                    ║
║  │ OR: Actual generation from plant records     │ ← CEA/THDC        ║
║  └────────────────────┬────────────────────────┘                    ║
║                       │                                              ║
║                       ▼                                              ║
║  GENERATION DEFICIT (Target Variable)                                ║
║  ┌─────────────────────────────────────────────┐                    ║
║  │ deficit_pct = (P_normal - P_actual)/P_normal × 100              ║
║  │ Range: -∞ to 100% (positive = shortfall)   │                    ║
║  └─────────────────────────────────────────────┘                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

          ML MODEL: SVR-RBF
          Input:  [ENSO features (t-3..t-0)] +
                  [Rainfall features (t-2..t-0)] +
                  [Reservoir features (t-1..t-0)] +
                  [Seasonal encoding] +
                  [El Niño intensity flag]
          Output: generation_deficit_pct(t)
          
          Horizon: 1-month-ahead (primary), 3-month-ahead (stretch)
```

### Data Flow Between Modules
```
raw/enso/      → src/data/enso_loader.py        → data/processed/enso_monthly.csv
raw/rainfall/  → src/data/rainfall_loader.py    → data/processed/rainfall_monthly.csv
raw/reservoir/ → src/data/reservoir_loader.py   → data/processed/reservoir_monthly.csv
                                                         ↓
                 src/features/feature_engineer.py → data/features/master_features.csv
                                                         ↓
                 src/models/train_svr.py          → models/svr_pipeline.pkl
                 src/models/train_baselines.py    → models/baselines/*.pkl
                                                         ↓
                 src/evaluation/evaluate.py       → reports/metrics/evaluation_report.json
                                                         ↓
                 dashboard/app.py                 → Streamlit Dashboard
```

---

## 5. DATASETS & FEATURES

### 5.1 Data Sources

| Dataset | Source | Granularity | Coverage | Variables |
|---------|--------|-------------|----------|-----------|
| ENSO merged | ENSOcast `merged_enso.csv` | Monthly | 1982–2024 | SST, SOI, ONI, ENSO_Phase, lags |
| IMD Rainfall | HydroIntel `rainfall.csv` | Annual by subdivision | 1901–2015 | SUBDIVISION, YEAR, ANNUAL |
| CWC/IndiaWRIS Reservoir | Reservoir project `IndiaWRIS_data/` | Daily 2001–2021 | 4 reservoirs | Level, Storage, Inflow, Outflow |
| Hydro+Weather merged | Reservoir project `datafinal_krs.csv` | Daily 2018–2021 | KRS reservoir | +ET, SM, rainfall |
| CEA Generation data | To be sourced: data.gov.in / CEA | Monthly | 2001–2024 | Plant-wise generation (MU) |

> **Note on Generation Data:** If CEA actual generation data is not available during implementation Phase 1, we proxy-estimate generation using the hydrological formula (see Section 7), which is scientifically defensible and used in published literature.

### 5.2 Master Feature Set (v1.1 — Revised for Sample Size Constraint)

> **v1.1 CRITICAL CHANGE:** Original plan had 40+ features. Actual training samples = 57. SVR safe feature limit (n_train/10) = 5–6; acceptable limit (n_train/5) = 11–12. Feature set reduced to **12 features**. Excluded: OHC_Anomaly (66.7% missing), TEMPERATURE_C (constant fill value in WRI), ET and soil moisture (only 2018-2021 coverage — insufficient overlap), SST_Anomaly raw lags (redundant with ONI lags which are a smoothed version of the same signal).

**Final 12-Feature Set**

| # | Feature | Source | Unit | Notes |
|---|---------|--------|------|-------|
| 1 | `ONI` | merged_enso.csv | °C | Primary ENSO indicator |
| 2 | `ONI_lag_2` | Computed | °C | 2-month lag — best ENSO-India lag |
| 3 | `ONI_lag_3` | Computed | °C | 3-month lag — peak correlation (+0.29) |
| 4 | `SST_Anomaly_rolling_3m` | Recomputed per split | °C | Smoothed ocean signal; recompute to avoid boundary leakage |
| 5 | `SOI` | merged_enso.csv | std | Atmospheric pressure differential |
| 6 | `rainfall_mm` | WRI `RAINFALL_MM` | mm/month | Monthly catchment rainfall (sum of daily) |
| 7 | `rainfall_anomaly_pct` | Computed | % | (actual - normal_m) / normal_m × 100; normal from training years only |
| 8 | `storage_pct_frl` | WRI (KRS) | % | Storage as % of FRL (124.8 ft) |
| 9 | `inflow_mcm` | WRI (KRS) | MCM | Monthly sum of cusecs × 86400 × 0.0283168 / 1e6 |
| 10 | `inflow_lag_1_mcm` | Computed | MCM | Previous month inflow |
| 11 | `month_sin` | Derived | - | sin(2π × month / 12) |
| 12 | `month_cos` | Derived | - | cos(2π × month / 12) |

**Excluded features (v1.1):**
- `OHC_Anomaly` — **66.7% missing** (344/516 rows); excluded
- `TEMPERATURE_C` (WRI) — **constant 23.2°C fill value**; not a real measurement; excluded
- `evapotranspiration_mm`, `soil_moisture_pct` — only available Jun 2018–May 2021 (insufficient overlap); excluded
- `SST_Anomaly_lag_1/2/3` — highly correlated with ONI lags (ONI = rolling mean of SST_Anomaly); causes multicollinearity; excluded pending VIF analysis
- `is_monsoon`, `is_djf`, `is_jja` — month_sin/cos already encodes seasonality; redundant; excluded
- `is_el_nino` binary flag — used **only for evaluation stratification**, NOT as a training feature (to avoid circularity with ONI)
- `outflow_mcm` — excluded as feature for safety (if target formula is ever changed to include outflow, this would create leakage)

**Target Variable (v1.1 — corrected)**
```python
# CORRECT target: inflow deficit at t+1, using features at t

target = 'inflow_deficit_pct_next_month'

# Formula:
# inflow_mcm(t) = sum(INFLOW_CUSECS_daily * 86400 * 0.0283168) / 1e6  [MCM]
# normal_inflow_m = mean(inflow_mcm for training years, same calendar month m)
# deficit_pct(t) = (normal_inflow_m - inflow_mcm(t)) / normal_inflow_m * 100
# target(t) = deficit_pct(t+1)  [1-month-ahead prediction]

# Positive = below-normal inflow = water/power deficit
# Negative = above-normal inflow = surplus

# NOTE: normal_inflow_m MUST be computed from training years ONLY (2011-2016)
# This computation happens inside Pipeline.fit(), not pre-computed on full data.
```

### 5.3 Feature Engineering Anti-Leakage Protocol

> ⚠️ **CRITICAL**: All features MUST be computed on training data only, then applied to test data. No future information must contaminate training.

```
Leakage Risk Assessment:
├── Lag features (ONI_lag_N): SAFE — shift creates NaN at start of series
├── Rolling means (rolling_3m): SAFE if computed per-split; UNSAFE if pre-computed
├── StandardScaler: FIT on X_train only; TRANSFORM X_test separately
├── Monthly climatological means for deficit: COMPUTE on TRAINING YEARS only
├── ENSO phase flag (is_el_nino): Based on pre-published NOAA classification — SAFE
└── OHC_Anomaly: From ENSOcast dataset (real NOAA values) — SAFE as input feature
```

**Pipeline Implementation Strategy:**
```python
# CORRECT approach using sklearn Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Fits on X_train ONLY inside Pipeline.fit()
    ('svr', SVR(kernel='rbf', C=..., gamma=..., epsilon=...))
])

# TimeSeriesSplit for cross-validation
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=1)  # gap=1 prevents leakage at fold boundaries
```

---

## 6. ML PIPELINE DESIGN

### 6.1 Pipeline Architecture (v1.1)

```
RAW DATA INGESTION
     │
     ▼
DATA VALIDATION & QUALITY CHECK
├── Schema validation (expected columns per source)
├── Date range check (2011-01 to 2020-12 for WRI)
├── Missing value audit
└── Outlier detection (IQR-based; flag extreme inflow months for review)
     │
     ▼
DATA PREPROCESSING
├── WRI daily → monthly aggregation:
│   ├── inflow/outflow: sum of daily cusecs × 86400 × 0.0283168 / 1e6 = MCM
│   ├── storage: end-of-month (last daily value)
│   ├── level: end-of-month (last daily value)
│   └── rainfall: sum of daily mm
├── ENSO: already monthly; parse Date, sort, set PeriodIndex
├── Temporal alignment: merge all on (year, month) key
└── Target: compute deficit_pct then shift by +1 month
     │
     ▼
FEATURE ENGINEERING (INSIDE PIPELINE — training slice only)
├── 1. Lag features: ONI_lag_2, ONI_lag_3, inflow_lag_1 (computed on X_train only)
├── 2. Rolling features: SST_Anomaly_rolling_3m (recomputed per split boundary)
├── 3. Rainfall anomaly: (actual - normal_m) / normal_m × 100
│        └── normal_m = training-year mean per calendar month (NO leakage)
├── 4. Storage % FRL: storage_tmc / max_storage_tmc × 100
├── 5. Seasonal encoding: month_sin, month_cos
├── 6. VIF check: remove any feature with VIF > 5 (prevent SVR kernel issues)
└── DROP rows with any NaN (after lags: first 3 rows dropped)
     │
     ▼
TIME-SERIES SPLIT (v1.1 — corrected from actual data)
├── Training: Apr 2011 – Dec 2015 (57 supervised pairs)
├── Validation: Jan 2016 – Dec 2017 (24 pairs; contains 2015-16 strong El Niño)
└── Test: Jan 2018 – Nov 2020 (35 pairs; NEVER touch until final evaluation)
     │
     ▼
HYPERPARAMETER TUNING (training set only, 4-fold TimeSeriesSplit)
├── Method: TimeSeriesSplit CV (n_splits=4, gap=1 month)  ← v1.1: 4 splits on 57 samples
├── Search: GridSearchCV over C, gamma, epsilon
├── Parameters:
│   ├── C: [0.1, 1, 10, 100, 1000]
│   ├── gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
│   └── epsilon: [0.05, 0.1, 0.2, 0.5]  ← removed 0.01 (too tight for noisy target)
└── Scoring: neg_RMSE (primary), neg_MAE (secondary)
     │
     ▼
FINAL MODEL TRAINING
└── Retrain best SVR on full train+validation set
     │
     ▼
TEST SET EVALUATION (only once, at the very end)
├── RMSE, MAE, MAPE, R²
├── Residual analysis (ACF/PACF of residuals for autocorrelation)
├── El Niño years vs. Non-El Niño years breakdown
└── Deficit magnitude accuracy (within ±5%, ±10%, ±20% bands)
     │
     ▼
BASELINE COMPARISON
├── Linear Regression (OLS)
├── Ridge Regression (L2 regularized)
├── XGBoost Regressor
├── Persistence Model (naive: deficit(t) = deficit(t-1))
└── Climatological Mean Model (deficit = historical mean per month)
     │
     ▼
SAVING ARTIFACTS
├── models/svr_pipeline.pkl (full sklearn Pipeline object)
├── models/baselines/{model_name}.pkl
├── reports/metrics/evaluation_results.json
├── reports/figures/residual_plots.png
└── data/features/master_features.csv (for dashboard)
```

### 6.2 SVR Configuration Details

```python
# Primary SVR configuration
SVR_CONFIG = {
    'kernel': 'rbf',         # RBF: handles non-linear ENSO-hydro relationships
    'C': 100,                # Regularization (tune via GridSearchCV)
    'gamma': 'scale',        # 1/(n_features × X.var()) — adapts to data scale
    'epsilon': 0.1,          # ε-tube width for insensitive loss
    'cache_size': 500,       # MB for kernel cache (speed)
    'max_iter': 10000        # Convergence guarantee
}

# Note on kernel choice:
# - RBF is preferred over linear because ENSO-to-hydrology relationships
#   involve complex lags and non-linearities
# - Polynomial kernel overfits on limited monthly data (~300-400 samples)
# - Sigmoid kernel is not universal approximator — avoid
```

### 6.3 Anti-Leakage Checklist (v1.1 — extended)

```
[ ] StandardScaler fitted ONLY on X_train
[ ] Rolling/lag features computed ONLY on training slice
[ ] Monthly inflow normals for deficit: COMPUTE on TRAINING YEARS (2011-2016) ONLY
[ ] Rainfall anomaly normals: COMPUTE from training period rainfall ONLY
[ ] No shuffling of time-series data at any stage
[ ] TimeSeriesSplit gap=1 between folds
[ ] Final test set (Jan 2018 – Nov 2020) evaluation performed EXACTLY ONCE
[ ] Hyperparameter search uses training folds only (NOT validation or test)
[ ] Target shifted by +1 month: y(t) = deficit_pct(t+1); no future info in X(t)
[ ] OHC_Anomaly EXCLUDED (67% missing — cannot impute safely at this density)
[ ] TEMPERATURE_C EXCLUDED (constant 23.2°C — WRI fill value, not real observation)
[ ] Outflow EXCLUDED as feature (safety: prevents leakage if target formula ever changes)
[ ] ENSO phase labels used ONLY for evaluation stratification, NOT as training features
[ ] VIF computed on X_train only; same drops applied to X_val and X_test
```

---

## 7. TARGET VARIABLE (v1.1 — FULLY REVISED)

> **v1.1 COMPLETE REVISION:** The original Generation Proxy Formula (P = η·ρ·g·Q·H) is **INVALID** for the KRS reservoir (4.8 MW installed capacity vs. 119,226 MWh/month proxy output — a 33× overestimate). No real generation data exists. Target is redesigned as **inflow deficit percentage**.

### 7.1 Target Formula (Inflow Deficit)

```python
# Step 1: Monthly inflow in MCM from daily cusecs (WRI INFLOW_CUSECS)
inflow_mcm(t) = sum_over_days(INFLOW_CUSECS_daily × 86400 × 0.0283168) / 1_000_000
#                                                    ^s/day   ^m³/ft³     ^MCM

# Step 2: Climatological normal per calendar month (TRAINING YEARS ONLY)
normal_inflow_m = mean(inflow_mcm where year ∈ [2011, 2012, 2013, 2014, 2015, 2016]
                       and calendar_month == m)
                       
# Step 3: Deficit percentage
deficit_pct(t) = (normal_inflow_m − inflow_mcm(t)) / normal_inflow_m × 100
# Positive → below normal → water/power deficit
# Negative → above normal → surplus

# Step 4: Shift target by +1 for clean prediction task
y(t) = deficit_pct(t+1)   ← predicted at time t
X(t) = all features observed at END of month t
```

### 7.2 Target Statistics (Confirmed from Data)

| Statistic | Value |
|-----------|-------|
| Count | 120 months |
| Mean | +4.7% |
| Std | 63.3% |
| Min | −211.5% (extreme flood) |
| Max | +94.6% (near-drought) |
| Median | +21.6% |

**Outlier handling:** Winsorize at 2nd–98th percentile (removes extreme flood outliers > ±150%). Evaluate both Winsorized and raw targets; report both results.

### 7.3 Deficit Classification Thresholds (for reporting only, not used in SVR training)

| Deficit Range | Category | Color Code | Action |
|--------------|----------|------------|--------|
| < -10% | Surplus | 🟢 Green | No action |
| -10% to 0% | Near-normal | 🔵 Blue | Monitor |
| 0% to 10% | Mild deficit | 🟡 Yellow | Alert |
| 10% to 25% | Moderate deficit | 🟠 Orange | Warning |
| 25% to 50% | Severe deficit | 🔴 Red | Crisis |
| > 50% | Extreme deficit | ⚫ Black | Emergency |

---

## 8. EVALUATION FRAMEWORK (v1.1)

### 8.1 Primary Metrics (Regression)

> **v1.1 CHANGE:** MAPE **REMOVED**. Target `inflow_deficit_pct` can equal zero (normal-inflow months), causing undefined MAPE = |y − ŷ|/|y| × 100. No fixed R²/RMSE success thresholds — establish empirically.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(y_true, y_pred, model_name):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),  # Same units as target (%)
        'MAE': mean_absolute_error(y_true, y_pred),            # Interpretable: "avg X% off"
        'R2': r2_score(y_true, y_pred),                       # Variance explained
        'MBE': np.mean(y_pred - y_true),                      # Systematic bias direction
        'within_10pct': np.mean(np.abs(y_pred - y_true) <= 10),  # Operational accuracy
        'within_20pct': np.mean(np.abs(y_pred - y_true) <= 20),
        'within_30pct': np.mean(np.abs(y_pred - y_true) <= 30),
        # NOTE: MAPE excluded — target can be zero
    }
```

### 8.2 Scientific Validity Tests

1. **El Niño Year Subset Test**: SVR must perform ≥ as well on El Niño years as overall — verify the model specifically improves deficit prediction during ENSO events (the primary use case)
2. **Residual Autocorrelation Test**: Ljung-Box test on residuals; high autocorrelation → model missing temporal structure
3. **Diebold-Mariano Test**: Statistical comparison of SVR vs. XGBoost forecast accuracy
4. **ENSO Feature Ablation**: Train SVR with and without ENSO features; validate that ENSO features significantly improve accuracy (this is the core scientific contribution)

### 8.3 Benchmark Targets — v1.1 REVISED

> **v1.1 CHANGE:** Fixed R²/RMSE success thresholds **REMOVED**. With 35 test samples (std ≈ 63%), all metrics carry high uncertainty. Benchmarks below are **illustrative ranges** based on literature, not pass/fail criteria. True success = SVR outperforms all baselines AND ENSO ablation test confirms ENSO features add measurable value.

| Model | Illustrative RMSE range (%) | Notes |
|-------|----------------------------|-------|
| Persistence (naive) | 50–80 | Expected poor; high inflow variance |
| Climatological mean | 40–60 | Seasonal baseline |
| Linear Regression | 30–55 | Linear ENSO-inflow relationship |
| Ridge Regression | 28–50 | Regularized linear |
| XGBoost | 25–45 | Tree ensemble baseline |
| **SVR-RBF (ours)** | **TBD after training** | **Target: beat all baselines on RMSE + MAE** |

**True success criteria (v1.1):**
1. SVR RMSE < all 4 baselines on test set
2. ENSO ablation: model with ENSO features has significantly lower RMSE than model without them
3. El Niño months subset: model predicts positive deficits more often than chance (> 60% directional accuracy)

---

## 9. FINAL FOLDER STRUCTURE

```
hydropulse-svr/
│
├── README.md                          # Project overview, setup, usage
├── requirements.txt                   # All dependencies (pinned versions)
├── setup.py                           # Package setup
├── config.yaml                        # Central configuration (paths, params)
│
├── data/
│   ├── raw/
│   │   ├── enso/
│   │   │   └── merged_enso.csv        # From ENSOcast (1982–2024)
│   │   ├── rainfall/
│   │   │   └── india_rainfall.csv     # From HydroIntel (IMD 1901–2015)
│   │   └── reservoir/
│   │       ├── KRS_indiawris_jan2001_may2021.csv
│   │       ├── Harangi_indiawris_jan2001_may2021.csv
│   │       ├── Hemavathi_indiawris_jan2001_may2021.csv
│   │       └── Kabini_indiawris_jan2001_may2021.csv
│   ├── processed/
│   │   ├── enso_monthly.csv           # Cleaned, validated ENSO features
│   │   ├── rainfall_monthly.csv       # Aggregated monthly catchment rainfall
│   │   ├── reservoir_monthly.csv      # Monthly aggregated hydrology (4 reservoirs)
│   │   └── generation_monthly.csv     # Monthly generation (proxy or real)
│   └── features/
│       ├── master_features.csv        # Final merged feature matrix (all reservoirs)
│       └── feature_metadata.json      # Column descriptions, units, sources
│
├── notebooks/
│   ├── 01_eda_enso.ipynb             # ENSO data exploration
│   ├── 02_eda_hydrology.ipynb        # Reservoir + rainfall exploration
│   ├── 03_feature_engineering.ipynb  # Feature creation & validation
│   ├── 04_model_training.ipynb       # SVR training & tuning
│   ├── 05_evaluation.ipynb           # Full evaluation suite
│   └── 06_el_nino_analysis.ipynb     # El Niño specific analysis
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── enso_loader.py            # Load & validate ENSO data
│   │   ├── rainfall_loader.py        # Load & aggregate rainfall
│   │   ├── reservoir_loader.py       # Load & aggregate reservoir data
│   │   ├── generation_loader.py      # Load or compute generation
│   │   └── data_merger.py            # Align all datasets by date
│   ├── features/
│   │   ├── __init__.py
│   │   ├── lag_features.py           # Lag, rolling feature computation
│   │   ├── enso_features.py          # ENSO-specific features (phase flags, intensity)
│   │   ├── hydro_features.py         # Hydrological feature derivation
│   │   └── feature_selector.py       # Correlation analysis, VIF, feature ranking
│   ├── models/
│   │   ├── __init__.py
│   │   ├── svr_model.py              # SVR pipeline (primary model)
│   │   ├── baseline_models.py        # LR, Ridge, XGBoost baselines
│   │   ├── hyperparameter_tuning.py  # GridSearchCV with TimeSeriesSplit
│   │   └── model_registry.py         # Save/load model artifacts
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # RMSE, MAE, MAPE, R², MBE, within-bands
│   │   ├── el_nino_analysis.py       # El Niño year specific evaluation
│   │   ├── residual_analysis.py      # Autocorrelation, Ljung-Box, residual plots
│   │   └── model_comparison.py       # Diebold-Mariano test, comparison table
│   └── utils/
│       ├── __init__.py
│       ├── time_series_utils.py      # series_to_supervised, TimeSeriesSplit helpers
│       ├── plot_utils.py             # Shared plotting functions
│       └── config_loader.py          # Config YAML reader
│
├── models/
│   ├── svr_pipeline.pkl              # Trained SVR (StandardScaler + SVR)
│   └── baselines/
│       ├── linear_regression.pkl
│       ├── ridge_regression.pkl
│       └── xgboost_regressor.pkl
│
├── reports/
│   ├── metrics/
│   │   └── evaluation_results.json   # All model metrics in structured format
│   └── figures/
│       ├── learning_curves.png
│       ├── residual_plots.png
│       ├── feature_importance.png
│       └── el_nino_comparison.png
│
├── dashboard/
│   ├── app.py                        # Main Streamlit entry point
│   ├── components/
│   │   ├── enso_panel.py             # Panel 1: ENSO Monitor
│   │   ├── hydrology_panel.py        # Panel 2: Hydrology Trends
│   │   ├── prediction_panel.py       # Panel 3: Model Predictions
│   │   ├── deficit_panel.py          # Panel 4: Deficit Forecast
│   │   └── risk_panel.py             # Panel 5: Risk Alerts
│   └── assets/
│       ├── style.css                 # Custom styling
│       └── logo.png                  # Project logo
│
└── tests/
    ├── test_data_loaders.py
    ├── test_feature_engineering.py
    ├── test_model_pipeline.py
    └── test_no_data_leakage.py       # Automated leakage check
```

---

## 10. MODULE RESPONSIBILITIES

### `src/data/enso_loader.py`
- Load `merged_enso.csv` (from ENSOcast)
- Parse dates to `pd.DatetimeIndex` with monthly frequency
- Validate: no missing ONI/SOI/SST_Anomaly in core date range
- Output: `pd.DataFrame` with columns: `[date, ONI, SOI, SST_Anomaly, ENSO_Phase, OHC_Anomaly]`

### `src/data/reservoir_loader.py`
- Load IndiaWRIS daily data per reservoir
- Convert daily → monthly (mean for level/head, sum for inflow/outflow, end-of-month for storage)
- Align to month-start DatetimeIndex
- Handle missing values: flag gaps > 2 months; interpolate short gaps
- Output: `pd.DataFrame` per reservoir in standard schema

### `src/data/generation_loader.py`
- If real generation data exists: parse CEA plant-wise generation CSV
- If not: compute proxy using `P = η × ρ × g × Q × H × T`
- Compute monthly climatological normal from training years
- Compute deficit_pct: `(normal - actual) / normal * 100`
- Output: `pd.DataFrame` with `[date, actual_generation_mu, normal_generation_mu, deficit_pct]`

### `src/features/lag_features.py`
- **CRITICAL**: Takes a `DataSplitter` object to ensure lags computed per split
- `compute_lags(df, cols, lags=[1,2,3])` → adds lag columns
- `compute_rolling(df, cols, windows=[3,6])` → adds rolling mean columns
- `compute_interactions(df, col1, col2)` → adds product interaction column
- Drops NaN rows created by lag operations

### `src/features/enso_features.py`
- `compute_enso_intensity(oni_series)` → `abs(ONI)` continuous
- `compute_el_nino_flag(oni_series, threshold=0.5, min_months=3)` → binary flag
- `compute_enso_phase_shifts(df)` → months since last ENSO phase transition
- `classify_enso_event_type(oni_series)` → weak/moderate/strong/very_strong/super

### `src/models/svr_model.py`
```python
class SVRPipeline:
    def __init__(self, config: dict):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(**config))
        ])
    
    def fit(self, X_train, y_train): ...
    def predict(self, X): ...
    def save(self, path): ...
    
    @classmethod
    def load(cls, path): ...
```

### `src/models/hyperparameter_tuning.py`
- `tune_svr(X_train, y_train, param_grid, n_splits=5)` → best_params dict
- Uses `TimeSeriesSplit(n_splits=5, gap=1)`
- Logs all trial results to `reports/metrics/tuning_log.csv`

### `src/evaluation/el_nino_analysis.py`
- `split_by_enso_phase(df, enso_col)` → dict of phase-specific DataFrames
- `evaluate_el_nino_years(y_true, y_pred, enso_phase)` → metrics per phase
- `plot_deficit_by_enso(df)` → box plot of deficits per ENSO phase

### `dashboard/app.py`
- 5-panel Streamlit multi-page layout
- Imports `src` modules for live predictions
- Loads `models/svr_pipeline.pkl` for inference
- Sidebar: Reservoir selector, date range, ENSO threshold

---

## 11. DASHBOARD DESIGN

### Overall Design System
- **Theme:** Dark mode with deep blue/teal palette (oceanographic feel)
- **Font:** Inter (Google Fonts) for all text
- **Charts:** Plotly (interactive, hover tooltips)
- **Layout:** Wide layout, 5 sidebar navigation tabs

### Panel 1: 🌊 ENSO Monitor
**Purpose:** Show current and historical ENSO state
- **KPI Cards:** Current ONI value, Current Phase, Months in current phase, Niño3.4 SST anomaly
- **Chart 1:** ONI time series with ±0.5°C threshold lines (El Niño = red zone, La Niña = blue zone)
- **Chart 2:** SOI vs ONI scatter colored by ENSO phase
- **Chart 3:** Seasonal distribution of ENSO phases (bar chart, from ENSOcast)
- **Insight Box:** Current ENSO forecast narrative (rule-based from ONI value)

### Panel 2: 💧 Hydrology Trends
**Purpose:** Show rainfall and reservoir behavior vs. ENSO
- **KPI Cards:** Current storage (% FRL), Monthly inflow, Rainfall anomaly (%), Evapotranspiration
- **Chart 1:** Multi-reservoir storage time series with ENSO shading (El Niño years highlighted)
- **Chart 2:** Rainfall anomaly vs. ONI scatter (correlation visualization)
- **Chart 3:** Inflow time series with historical range band
- **Chart 4:** Storage waterfall chart (inflow - outflow - ET = net change)

### Panel 3: 🤖 Model Predictions
**Purpose:** Show SVR model performance and comparison
- **KPI Cards:** SVR RMSE, R², MAE, El Niño year accuracy
- **Chart 1:** Actual vs. Predicted deficit (time series, overlaid)
- **Chart 2:** Model comparison bar chart (SVR vs. baselines, grouped by metric)
- **Chart 3:** Residuals plot (residual vs. time, with autocorrelation)
- **Chart 4:** Feature importance proxy (using permutation importance for SVR)
- **Table:** El Niño year breakdown — predicted vs. actual deficit per event

### Panel 4: 🔮 Deficit Forecast
**Purpose:** Forward-looking predictions
- **KPI Cards:** Next-month predicted deficit, 3-month outlook, Confidence interval
- **Chart 1:** 3-month forecast with uncertainty bands
- **Slider:** "What-If" ONI scenario slider (user adjusts ONI, see predicted deficit)
- **Table:** Monthly forecast table (month, predicted ONI, predicted deficit %, risk category)
- **Download Button:** Export forecast as CSV

### Panel 5: ⚠️ Risk Assessment
**Purpose:** Operational risk for power planning
- **KPI Cards:** Current risk level (color-coded), Expected shortfall (MU), Risk trend
- **Chart 1:** Risk category gauge (speedometer style)
- **Chart 2:** Historical El Niño deficits heatmap (year × month grid)
- **Chart 3:** Generation deficit distribution (histogram by ENSO phase)
- **Alerts Panel:** Color-coded alert system with recommended actions
- **Report Download:** PDF-ready risk summary

---

## 12. IMPLEMENTATION PHASES

### Phase 0: Environment Setup (Day 1)
- Create `hydropulse-svr/` folder structure
- Set up `requirements.txt` with pinned versions
- Configure `config.yaml` with all data paths and model parameters
- Write `tests/test_no_data_leakage.py` (test-first approach for leakage)
- Initialize git repository

### Phase 1: Data Pipeline (Days 2–5)
- `src/data/enso_loader.py`: Load and validate ENSOcast data (flag and drop OHC_Anomaly)
- `src/data/reservoir_loader.py`: Daily → monthly aggregation for 4 reservoirs (inflow to MCM, storage EOM)
- `src/data/target_builder.py`: Compute inflow_deficit_pct; shift by +1 month (NO generation proxy)
- `src/data/data_merger.py`: Align all on monthly PeriodIndex
- Notebooks: `01_eda_enso.ipynb`, `02_eda_hydrology.ipynb`
- **Milestone:** `data/processed/` folder populated; 116 supervised (X,y) pairs ready; target statistics verified

### Phase 2: Feature Engineering (Days 6–8)
- `src/features/lag_features.py`: All lag and rolling features
- `src/features/enso_features.py`: ENSO intensity, phase flags, event classification
- `src/features/hydro_features.py`: Net head, storage % FRL, inflow anomaly
- `src/features/feature_selector.py`: Correlation matrix, VIF filter, mutual info
- Notebook: `03_feature_engineering.ipynb`
- **Milestone:** `data/features/master_features.csv` ready

### Phase 3: SVR Model Development (Days 9–13)
- `src/models/svr_model.py`: SVRPipeline class
- `src/models/hyperparameter_tuning.py`: GridSearchCV with TimeSeriesSplit
- `src/models/baseline_models.py`: LR, Ridge, XGBoost wrappers
- Notebook: `04_model_training.ipynb`
- **Run hyperparameter search** and document best parameters
- **Milestone:** `models/svr_pipeline.pkl` saved and verified

### Phase 4: Evaluation (Days 14–16)
- `src/evaluation/metrics.py`: Full metric suite
- `src/evaluation/el_nino_analysis.py`: Phase-specific evaluation
- `src/evaluation/residual_analysis.py`: Autocorrelation, Ljung-Box
- `src/evaluation/model_comparison.py`: Diebold-Mariano test
- Notebooks: `05_evaluation.ipynb`, `06_el_nino_analysis.ipynb`
- **Milestone:** `reports/metrics/evaluation_results.json` with all model comparisons

### Phase 5: Dashboard (Days 17–22)
- `dashboard/app.py`: Navigation skeleton
- `dashboard/components/enso_panel.py`: Panel 1
- `dashboard/components/hydrology_panel.py`: Panel 2
- `dashboard/components/prediction_panel.py`: Panel 3
- `dashboard/components/deficit_panel.py`: Panel 4
- `dashboard/components/risk_panel.py`: Panel 5
- **Milestone:** Full Streamlit dashboard running locally

### Phase 6: Documentation & Polish (Days 23–25)
- `README.md`: Setup instructions, data sources, methodology
- Docstrings in all `src/` modules
- Research-grade evaluation report (`reports/evaluation_report.md`)
- Code review for any remaining leakage or correctness issues
- **Milestone:** Project submission-ready

---

## 13. EXACT FIRST CODING TASKS

### Task 1: Create `config.yaml` (first file to write — v1.1)
```yaml
# config.yaml — v1.1
project:
  name: "HydroPulse-SVR"
  version: "1.1.0"

data:
  raw:
    enso: "data/raw/enso/merged_enso.csv"
    wri_reservoir_dir: "data/raw/reservoir/wri/"      # Has inflow + outflow (2011-2020)
    wb_rainfall_dir: "data/raw/rainfall/world_bank/"  # Monthly catchment (1991-2020)
  processed:
    enso: "data/processed/enso_monthly.csv"
    reservoir: "data/processed/reservoir_monthly.csv"  # Monthly aggregated WRI
    target: "data/processed/inflow_deficit_monthly.csv" # Target variable
  features:
    master: "data/features/master_features.csv"       # All 12 features + target

model:
  svr:
    kernel: "rbf"
    C: 100        # Starting point; tune via GridSearchCV
    gamma: "scale"
    epsilon: 0.1
  train_period: ["2011-04", "2015-12"]  # 57 pairs after lag removal
  val_period:   ["2016-01", "2017-12"]  # 24 pairs
  test_period:  ["2018-01", "2020-11"]  # 35 pairs
  cv_n_splits: 4
  cv_gap: 1

reservoir:
  primary: "KRS"           # Primary case study reservoir
  all: ["KRS", "Harangi", "Hemavathi", "Kabini"]
  frl_ft:
    KRS: 124.8
    Harangi: 2871.42  # MSL — convert to gauge separately
    Hemavathi: 2890.63
    Kabini: 2696.16
  dead_storage_ft:
    KRS: 62.8           # Approximate from WRI minimum observed level

enso:
  el_nino_threshold: 0.5
  min_consecutive_months: 5  # v1.1: corrected to NOAA standard (was 3)

features:
  lag_months: [2, 3]          # Only lags 2 and 3 for ONI (peak correlation confirmed)
  rolling_windows: [3]        # Only 3-month rolling for SST_Anomaly
  n_features: 12              # Hard limit; do not exceed without sample-size justification
  exclude_always:
    - "OHC_Anomaly"           # 67% missing
    - "TEMPERATURE_C"         # WRI fill value (constant 23.2)
    - "outflow_mcm"           # Safety: excluded to prevent future leakage scenarios

target:
  column: "inflow_deficit_pct_next_month"
  normal_years: [2011, 2012, 2013, 2014, 2015, 2016]  # Training years for normal computation
  winsorize_pct: [2, 98]      # Clip extreme outlier inflow values
```

### Task 2: `src/data/enso_loader.py`
```python
"""
Load and validate ENSO data from merged_enso.csv (ENSOcast source).
Key columns: Date, ONI, SOI, SST_Anomaly, ENSO_Phase, OHC_Anomaly
"""
import pandas as pd
import numpy as np
from pathlib import Path

REQUIRED_COLUMNS = ['Date', 'ONI', 'SOI', 'SST_Anomaly', 'ENSO_Phase']

def load_enso(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'])
    
    # Validate schema
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in ENSO data: {missing}")
    
    # Set monthly DatetimeIndex
    df = df.sort_values('Date').reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.to_period('M').dt.to_timestamp()
    df = df.set_index('Date')
    
    # Validate continuity
    expected_months = pd.date_range(df.index[0], df.index[-1], freq='MS')
    missing_months = expected_months.difference(df.index)
    if len(missing_months) > 0:
        print(f"WARNING: {len(missing_months)} missing months in ENSO data")
    
    return df[REQUIRED_COLUMNS + ['OHC_Anomaly'] if 'OHC_Anomaly' in df.columns else REQUIRED_COLUMNS]
```

### Task 3: `tests/test_no_data_leakage.py`
```python
"""
Automated test to verify no data leakage in the ML pipeline.
Run: pytest tests/test_no_data_leakage.py -v
"""
import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler

def test_scaler_not_fitted_on_test_data():
    """StandardScaler should be fitted only on training data."""
    from src.models.svr_model import SVRPipeline
    
    # Create dummy time-series data
    X_train = np.random.randn(100, 10)
    X_test = np.random.randn(20, 10)
    y_train = np.random.randn(100)
    
    pipeline = SVRPipeline(config={'kernel': 'rbf', 'C': 1.0})
    pipeline.fit(X_train, y_train)
    
    # Extract fitted scaler mean
    scaler_mean = pipeline.pipeline.named_steps['scaler'].mean_
    
    # Verify it matches training data statistics
    np.testing.assert_allclose(scaler_mean, X_train.mean(axis=0), rtol=1e-5)
    
    # Verify it does NOT match test+train combined
    combined_mean = np.concatenate([X_train, X_test]).mean(axis=0)
    assert not np.allclose(scaler_mean, combined_mean, rtol=1e-3)

def test_lag_features_no_lookahead():
    """Lag features must not use future values."""
    from src.features.lag_features import compute_lags
    import pandas as pd
    
    dates = pd.date_range('2000-01', periods=24, freq='MS')
    df = pd.DataFrame({'ONI': np.random.randn(24)}, index=dates)
    
    df_lagged = compute_lags(df, cols=['ONI'], lags=[1, 2, 3])
    
    # At time t, ONI_lag_1 should equal ONI at t-1
    for i in range(3, len(df)):
        expected_lag1 = df['ONI'].iloc[i-1]
        actual_lag1 = df_lagged['ONI_lag_1'].iloc[i]
        assert abs(expected_lag1 - actual_lag1) < 1e-10, f"Lag-1 lookahead at index {i}"
```

### Task 4: `src/features/lag_features.py`
```python
"""
Compute lag features for time-series data.
IMPORTANT: Must be called AFTER train/test split to avoid leakage.
Apply to X_train and X_test separately.
"""
import pandas as pd
import numpy as np
from typing import List, Optional

def compute_lags(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    """Add lag columns for each specified column."""
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def compute_rolling(df: pd.DataFrame, cols: List[str], 
                    windows: List[int], min_periods: int = 1) -> pd.DataFrame:
    """Add rolling mean columns."""
    df = df.copy()
    for col in cols:
        for window in windows:
            df[f'{col}_rolling_{window}m'] = (
                df[col]
                .rolling(window=window, min_periods=min_periods)
                .mean()
            )
    return df

def compute_interactions(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Add multiplicative interaction feature."""
    df = df.copy()
    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    return df

def series_to_supervised(data: pd.DataFrame, n_in: int = 1, 
                          n_out: int = 1, dropnan: bool = True) -> pd.DataFrame:
    """
    Adapted from reservoir-water-level-forecasting.
    Frame a time series as a supervised learning dataset.
    """
    n_vars = data.shape[1] if hasattr(data, 'shape') else 1
    df = pd.DataFrame(data)
    cols, names = [], []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'{col}(t-{i})' for col in df.columns]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        suffix = '(t)' if i == 0 else f'(t+{i})'
        names += [f'{col}{suffix}' for col in df.columns]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg
```

---

## 14. ACCEPTANCE CRITERIA

### Scientific Acceptance Criteria
- [ ] SVR model achieves R² ≥ 0.65 on the held-out test set (2020–present)
- [ ] SVR RMSE is lower than all 4 baseline models on the test set
- [ ] ENSO feature ablation confirms statistically significant improvement (p < 0.05, Diebold-Mariano test)
- [ ] El Niño year deficit predictions are within ±15% for ≥ 80% of El Niño months
- [ ] Residual autocorrelation at lag-1 is < 0.3 (Ljung-Box p > 0.05)
- [ ] No data leakage detected by automated tests (all `tests/test_no_data_leakage.py` pass)

### Technical Acceptance Criteria
- [ ] Full pipeline runs end-to-end: `python src/pipeline.py --config config.yaml`
- [ ] All `pytest tests/` pass with 0 failures
- [ ] `models/svr_pipeline.pkl` is reproducible (same random seed, same results)
- [ ] Dashboard runs locally with `streamlit run dashboard/app.py`
- [ ] Code coverage ≥ 80% for `src/` modules
- [ ] No hardcoded paths — all paths via `config.yaml`

### Dashboard Acceptance Criteria
- [ ] All 5 panels load without errors
- [ ] ENSO Monitor panel shows ONI time series with correct threshold lines
- [ ] Prediction panel shows actual vs. predicted overlay for test period
- [ ] What-If ONI slider produces different deficit predictions
- [ ] Deficit forecast table downloads correctly as CSV
- [ ] Risk panel color codes correctly for 6 risk levels

### Documentation Criteria
- [ ] `README.md` contains: setup, data sources, methodology, how to reproduce results
- [ ] All `src/` functions have docstrings with parameters and return types
- [ ] `feature_metadata.json` documents all 40+ features with units and sources
- [ ] `reports/evaluation_report.md` is publication-ready prose with tables and figures

---

## 15. FINAL ARCHITECTURAL DECISION

```
╔════════════════════════════════════════════════════════════════════════╗
║           FINAL ARCHITECTURAL DECISION: HOW 3 BECOME 1               ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  FROM ENSOCAST:                                                        ║
║  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │ ✅ merged_enso.csv (42 years NOAA data)                        │  ║
║  │ ✅ Feature engineering: lag, rolling, seasonal, interaction    │  ║
║  │ ✅ Chronological split protocol (no shuffle)                   │  ║
║  │ ✅ StandardScaler on training data only                        │  ║
║  │ ✅ ENSO phase visualization patterns (ONI chart, SOI chart)    │  ║
║  │ ❌ Classification framing (we need regression)                 │  ║
║  │ ❌ CNN/LSTM as primary (we use SVR-RBF)                        │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
║                              │                                         ║
║                              ▼ FEEDS INTO                              ║
║                                                                        ║
║  FROM RESERVOIR WATER-LEVEL FORECASTING:                              ║
║  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │ ✅ IndiaWRIS dataset (inflow, outflow, storage, level)         │  ║
║  │ ✅ series_to_supervised() function                             │  ║
║  │ ✅ SVR as validated model for reservoir domain                 │  ║
║  │ ✅ Weighted ensemble logic (RMSE-weighted)                     │  ║
║  │ ✅ Multi-feature reservoir modeling concept                    │  ║
║  │ ❌ Univariate SVR (replace with multivariate + ENSO)           │  ║
║  │ ❌ Leaky MinMaxScaler (fix: fit on train only)                 │  ║
║  │ ❌ Broken path dependencies (rebuild cleanly)                  │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
║                              │                                         ║
║                              ▼ FEEDS INTO                              ║
║                                                                        ║
║  FROM HYDROINTEL-AI:                                                   ║
║  ┌─────────────────────────────────────────────────────────────────┐  ║
║  │ ✅ Streamlit 5-tab dashboard architecture                      │  ║
║  │ ✅ Executive KPI card panel layout                             │  ║
║  │ ✅ Region selector + reservoir selector pattern                │  ║
║  │ ✅ Plotly line/bar/histogram chart patterns                    │  ║
║  │ ✅ Download button for reports                                 │  ║
║  │ ✅ India rainfall dataset (IMD subdivision data)               │  ║
║  │ ❌ Synthetic power formula (replaced with real proxy physics)  │  ║
║  │ ❌ LinearRegression model (replaced with SVR-RBF)              │  ║
║  │ ❌ Rule-based AI insights (replaced with model-driven output)  │  ║
║  └─────────────────────────────────────────────────────────────────┘  ║
║                              │                                         ║
║                              ▼ PRODUCES                                ║
║                                                                        ║
║  ╔═══════════════════════════════════════════════════════════════╗     ║
║  ║            HydroPulse-SVR: FINAL INTEGRATED SYSTEM           ║     ║
║  ╠═══════════════════════════════════════════════════════════════╣     ║
║  ║                                                               ║     ║
║  ║  ENSO Layer (ENSOcast DNA)                                    ║     ║
║  ║    ONI → SOI → SST_Anomaly → lags → flags                   ║     ║
║  ║         ↓                                                     ║     ║
║  ║  Hydrology Layer (Reservoir project DNA)                      ║     ║
║  ║    Inflow → Storage → Head → Generation Proxy                ║     ║
║  ║         ↓                                                     ║     ║
║  ║  ML Layer (SVR-RBF, sklearn Pipeline, TimeSeriesSplit)        ║     ║
║  ║    Input: 40+ features    Output: deficit_pct                 ║     ║
║  ║    Baselines: LR, Ridge, XGBoost for comparison              ║     ║
║  ║         ↓                                                     ║     ║
║  ║  Dashboard Layer (HydroIntel-AI DNA)                          ║     ║
║  ║    5 Panels: ENSO | Hydro | Predictions | Forecast | Risk    ║     ║
║  ║    Premium dark UI, Plotly charts, KPI cards, downloads      ║     ║
║  ║                                                               ║     ║
║  ╚═══════════════════════════════════════════════════════════════╝     ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
```

### What Makes This New Project Distinct from All Three Sources
1. **Only project with a real scientific research question** (El Niño → deficit)
2. **Only project with proper causal chain** (ENSO → rainfall → inflow → storage → generation → deficit)
3. **First project to combine ENSO indices with reservoir hydrology** in a single ML pipeline
4. **SVR is used correctly**: multivariate regression with proper feature engineering, not univariate lag-1
5. **Research-grade rigor**: Diebold-Mariano test, ENSO ablation study, residual analysis
6. **No data leakage by design**: automated tests, Pipeline API, chronological splits
7. **Scientifically defensible target variable**: either real generation data or physically-motivated proxy

---

*End of MASTER_PROJECT_PLAN.md*  
*Version: 1.0 | Date: 2026-09-28 | Status: Implementation-Ready*
