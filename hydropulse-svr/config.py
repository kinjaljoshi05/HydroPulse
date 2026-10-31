# config.py  --  HydroPulse-SVR: Central configuration
# All paths, constants, and ML parameters in one place.

import os

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))   # hydropulse-svr/
PARENT_DIR = os.path.dirname(BASE_DIR)                    # Hydropulseproject/

# ---------------------------------------------------------------------------
# Source data paths  (read-only -- the 3 reference repos are never modified)
# ---------------------------------------------------------------------------
ENSO_PATH = os.path.join(PARENT_DIR, "ensocast", "merged_enso.csv")

KRS_WRI_PATH = os.path.join(
    PARENT_DIR,
    "reservoir-water-level-forecasting",
    "data", "processed", "WRI",
    "KRS_wri_jan2011_dec2020.csv",
)

WB_RAINFALL_PATH = os.path.join(
    PARENT_DIR,
    "reservoir-water-level-forecasting",
    "data", "raw",
    "World Bank Monthly Temperature rainfall data",
    "KRS_pr_1991_2020_76.5724_12.4255.csv",
)

# ---------------------------------------------------------------------------
# Output paths  (created automatically on first run)
# ---------------------------------------------------------------------------
MODELS_DIR      = os.path.join(BASE_DIR, "models")
SVR_MODEL_PATH  = os.path.join(MODELS_DIR, "svr_model.pkl")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")
DATA_OUT_PATH   = os.path.join(MODELS_DIR, "master_data.csv")
METRICS_PATH    = os.path.join(MODELS_DIR, "metrics.csv")
PRED_PATH       = os.path.join(MODELS_DIR, "predictions.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Reservoir physical constants (KRS, WRI gauge)
# ---------------------------------------------------------------------------
KRS_FRL_FT  = 124.8   # Full Reservoir Level in feet
KRS_DEAD_FT = 62.8    # Approximate dead-storage level in feet (WRI observed minimum)

# ---------------------------------------------------------------------------
# Train / Validation / Test split boundaries
# ---------------------------------------------------------------------------
TRAIN_END   = "2015-12"              # Last month included in training
VAL_END     = "2017-12"              # Last month included in validation
# Test: everything after VAL_END  (2018-01 to 2020-11)

TRAIN_YEARS = list(range(2011, 2017))  # Used to compute inflow normals -- no leakage

# ---------------------------------------------------------------------------
# Feature columns  (11 features; safe for ~57 training samples)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "ONI",            # Oceanic Nino Index  (primary ENSO signal)
    "SOI",            # Southern Oscillation Index  (atmospheric signal)
    "ONI_lag2",       # ONI 2 months ago  (monsoon lag)
    "ONI_lag3",       # ONI 3 months ago  (peak correlation +0.29 with inflow deficit)
    "rainfall_mm",    # Monthly catchment rainfall  (World Bank / ERA5)
    "inflow_mcm",     # Current-month reservoir inflow  (MCM)
    "inflow_lag1_mcm",# Previous-month inflow  (persistence signal)
    "storage_pct_frl",# Reservoir storage as % of Full Reservoir Level
    "month_sin",      # sin(2*pi*month/12)  -- seasonal cycle
    "month_cos",      # cos(2*pi*month/12)  -- seasonal cycle
    "deficit_pct",    # Current-month inflow deficit %  (autoregressive predictor)
                      # This is the CURRENT month's deficit -- known before predicting t+1.
]

# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
TARGET_COL = "deficit_pct_next"
# = next-month KRS inflow deficit (%) used as a hydropower-risk indicator.
# Positive  => below-normal inflow  => water/power deficit risk.
# Negative  => above-normal inflow  => surplus.

# ---------------------------------------------------------------------------
# SVR default parameters  (overridden by GridSearchCV in train_model.py)
# ---------------------------------------------------------------------------
SVR_PARAMS = dict(kernel="rbf", C=50, gamma="scale", epsilon=0.5)
