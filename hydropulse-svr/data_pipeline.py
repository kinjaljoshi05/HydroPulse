# data_pipeline.py  --  HydroPulse-SVR: Data loading and feature engineering
#
# Reads from the 3 reference repositories (read-only) and produces a clean
# monthly master dataset ready for model training.
#
# Sources:
#   ENSOcast              -> merged_enso.csv          (ENSO climate indices)
#   Reservoir Forecasting -> KRS_wri_jan2011_dec2020  (daily KRS hydrology)
#   Reservoir Forecasting -> KRS_pr_1991_2020 (WB)    (monthly catchment rainfall)
#
# Output: 116-row monthly DataFrame with 11 features + target.

import pandas as pd
import numpy as np
from config import (
    ENSO_PATH, KRS_WRI_PATH, WB_RAINFALL_PATH,
    TRAIN_YEARS, DATA_OUT_PATH,
    FEATURE_COLS, TARGET_COL,
)


# ---------------------------------------------------------------------------
# 1. LOAD ENSO DATA
# ---------------------------------------------------------------------------
def load_enso():
    """Load monthly ENSO indices from ENSOcast merged_enso.csv."""
    df = pd.read_csv(ENSO_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["YM"] = df["Date"].dt.to_period("M")

    enso = df[["YM", "ONI", "SOI", "SST_Anomaly", "ENSO_Phase"]].copy()
    enso["ENSO_Phase"] = enso["ENSO_Phase"].str.strip()

    print("[ENSO] Loaded %d monthly rows  (%s to %s)" % (len(enso), enso.YM.min(), enso.YM.max()))
    return enso


# ---------------------------------------------------------------------------
# 2. LOAD KRS RESERVOIR DATA  (daily -> monthly)
# ---------------------------------------------------------------------------
def load_reservoir():
    """
    Load KRS WRI daily reservoir data and aggregate to monthly.
    Inflow and outflow are converted from cusecs-days to MCM:
        MCM = sum(daily_cusecs) * 86400 s/day * 0.0283168 m3/ft3 / 1e6
    """
    df = pd.read_csv(KRS_WRI_PATH, parse_dates=["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)
    df["YM"] = df["DATE"].dt.to_period("M")

    CUSECS_TO_MCM = 86400 * 0.0283168 / 1e6   # conversion factor

    monthly = (
        df.groupby("YM")
        .agg(
            inflow_mcm      = ("INFLOW_CUSECS",      lambda x: x.sum() * CUSECS_TO_MCM),
            outflow_mcm     = ("OUTFLOW_CUSECS",     lambda x: x.sum() * CUSECS_TO_MCM),
            storage_tmc_eom = ("PRESENT_STORAGE_TMC", "last"),   # end-of-month storage
            level_eom_ft    = ("RES_LEVEL_FT",       "last"),    # end-of-month level
            rainfall_wri_mm = ("RAINFALL_MM",        "sum"),     # fallback rainfall
            n_days          = ("DATE",               "count"),
        )
        .reset_index()
    )

    # Storage as % of observed maximum (used as FRL proxy)
    max_storage = df["PRESENT_STORAGE_TMC"].max()
    monthly["storage_pct_frl"] = monthly["storage_tmc_eom"] / max_storage * 100

    print("[Reservoir] Monthly rows: %d  (%s to %s)" % (len(monthly), monthly.YM.min(), monthly.YM.max()))
    print("  Max observed storage: %.2f TMC  (used as 100%% FRL reference)" % max_storage)
    return monthly


# ---------------------------------------------------------------------------
# 3. LOAD WORLD BANK MONTHLY RAINFALL
# ---------------------------------------------------------------------------
def load_wb_rainfall():
    """
    Load catchment-specific monthly rainfall for KRS from the World Bank
    Climate Data Portal (ERA5 reanalysis point estimate, 1991-2020).
    """
    df = pd.read_csv(WB_RAINFALL_PATH)
    df.columns = [c.strip() for c in df.columns]

    month_map = {
        "Jan Average": 1,  "Feb Average": 2,  "Mar Average": 3,
        "Apr Average": 4,  "May Average": 5,  "Jun Average": 6,
        "Jul Average": 7,  "Aug Average": 8,  "Sep Average": 9,
        "Oct Average": 10, "Nov Average": 11, "Dec Average": 12,
    }
    df["month"] = df["Statistics"].str.strip().map(month_map)
    df = df.dropna(subset=["month"])   # drop any unmapped rows
    df["YM"] = pd.PeriodIndex(
        pd.to_datetime({"year": df["Year"].astype(int), "month": df["month"], "day": 1}),
        freq="M",
    )
    wb = df[["YM", "Rainfall - (MM)"]].rename(columns={"Rainfall - (MM)": "rainfall_wb_mm"})

    print("[WB Rainfall] Monthly rows: %d  (%s to %s)" % (len(wb), wb.YM.min(), wb.YM.max()))
    return wb


# ---------------------------------------------------------------------------
# 4. MERGE ALL SOURCES
# ---------------------------------------------------------------------------
def merge_all(enso, reservoir, wb_rain):
    """Merge ENSO + Reservoir + WB Rainfall on monthly period key (YM)."""
    df = reservoir.merge(enso,    on="YM", how="inner")
    df = df.merge(wb_rain,        on="YM", how="left")

    # Use World Bank rainfall where available; fall back to WRI catchment rain
    df["rainfall_mm"] = df["rainfall_wb_mm"].combine_first(df["rainfall_wri_mm"])

    print("[Merge] Combined rows: %d  (%s to %s)" % (len(df), df.YM.min(), df.YM.max()))
    return df


# ---------------------------------------------------------------------------
# 5. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df):
    """
    Compute lag features and seasonal encoding.
    Data must be sorted chronologically before calling this function.
    No leakage: all shifts use only past values.
    """
    df = df.sort_values("YM").reset_index(drop=True)

    # ENSO lags
    df["ONI_lag2"] = df["ONI"].shift(2)
    df["ONI_lag3"] = df["ONI"].shift(3)

    # Previous-month inflow (autoregressive signal)
    df["inflow_lag1_mcm"] = df["inflow_mcm"].shift(1)

    # Circular seasonal encoding
    df["calendar_month"] = df["YM"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["calendar_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["calendar_month"] / 12)

    # Additional hydrologic features
    df["rainfall_lag1_mm"] = df["rainfall_mm"].shift(1)
    df["storage_change"] = df["storage_pct_frl"].diff()

    print("[Features] Lag + seasonal features added.")
    return df


# ---------------------------------------------------------------------------
# 6. BUILD TARGET VARIABLE
# ---------------------------------------------------------------------------
def build_target(df):
    """
    Compute monthly inflow deficit (%) and shift by +1 month to create
    the prediction target.

    Formula:
        normal_inflow_m  = mean inflow for calendar month m  (training years ONLY)
        deficit_pct(t)   = (normal_inflow_m - actual_inflow(t)) / normal_inflow_m * 100
        target(t)        = deficit_pct(t+1)   <-- what we predict at time t

    Interpretation:
        Positive => below-normal inflow => hydropower risk indicator (water deficit)
        Negative => above-normal inflow => surplus

    IMPORTANT: Monthly normals are computed ONLY from training years to prevent
    data leakage from future periods into the feature/target construction.
    """
    df["year"] = df["YM"].dt.year
    train_mask = df["year"].isin(TRAIN_YEARS)

    # Climatological normal per calendar month (training years only)
    normal_inflow = (
        df[train_mask]
        .groupby("calendar_month")["inflow_mcm"]
        .mean()
        .rename("normal_inflow_mcm")
    )
    df = df.merge(normal_inflow, on="calendar_month", how="left")
    # Crucial guard: ensure strict chronological ordering before shifting
    df = df.sort_values("YM").reset_index(drop=True)

    df["deficit_pct"] = (
        (df["normal_inflow_mcm"] - df["inflow_mcm"]) / df["normal_inflow_mcm"] * 100
    )

    # Shift deficit by +1 month: predict NEXT month's deficit using CURRENT features
    df[TARGET_COL] = df["deficit_pct"].shift(-1)

    dmin = df["deficit_pct"].min()
    dmax = df["deficit_pct"].max()
    print("[Target] deficit_pct range: %.1f%% to %.1f%%" % (dmin, dmax))
    return df


# ---------------------------------------------------------------------------
# 7. CLEAN: drop NaN rows and winsorize extreme target values
# ---------------------------------------------------------------------------
def clean_dataset(df):
    """
    Drop rows with NaN (first 3 rows from lags; last row has no t+1 target).
    Winsorize the target at 2nd/98th percentile to reduce influence of
    extreme flood months (e.g. Aug 2018 inflow was 14x normal).
    """
    required = FEATURE_COLS + [TARGET_COL]
    df = df.dropna(subset=required).copy()

    lo = df[TARGET_COL].quantile(0.02)
    hi = df[TARGET_COL].quantile(0.98)
    df[TARGET_COL] = df[TARGET_COL].clip(lo, hi)

    print("[Clean] Usable rows: %d  (target clipped to [%.1f%%, %.1f%%])" % (len(df), lo, hi))
    return df


# ---------------------------------------------------------------------------
# 8. MAIN PIPELINE ENTRY POINT
# ---------------------------------------------------------------------------
def build_master_dataset(save=True):
    """
    Run the complete pipeline and return a clean master DataFrame.
    Optionally saves the result to models/master_data.csv.
    """
    print("=" * 60)
    print("  HydroPulse-SVR  |  Data Pipeline")
    print("=" * 60)

    enso      = load_enso()
    reservoir = load_reservoir()
    wb_rain   = load_wb_rainfall()

    df = merge_all(enso, reservoir, wb_rain)
    df = engineer_features(df)
    df = build_target(df)
    df = clean_dataset(df)

    # Readable date column for charts
    df["date"] = df["YM"].dt.to_timestamp()

    if save:
        df.to_csv(DATA_OUT_PATH, index=False)
        print("\n[Saved] Master dataset -> %s" % DATA_OUT_PATH)

    print("\n[Done] Dataset shape: %s" % str(df.shape))
    return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = build_master_dataset()
    print("\nSample (last 6 rows):")
    cols = ["date", "ONI", "ENSO_Phase", "inflow_mcm", "storage_pct_frl",
            "deficit_pct", TARGET_COL]
    print(df[cols].tail(6).to_string(index=False))
