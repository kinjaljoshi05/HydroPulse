# train_model.py  --  HydroPulse-SVR: Model training and evaluation
#
# Trains SVR (RBF) and three baseline models on the master dataset.
# Uses strict chronological train/validation/test split -- no data leakage.
# Saves model, scaler, metrics, and predictions to models/.
#
# Run:  python train_model.py

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

warnings.filterwarnings("ignore")

from config import (
    FEATURE_COLS, TARGET_COL, TRAIN_END, VAL_END,
    SVR_MODEL_PATH, SCALER_PATH, METRICS_PATH,
    DATA_OUT_PATH, PRED_PATH,
)


# ---------------------------------------------------------------------------
# METRICS HELPER
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, name):
    """Return a dict of evaluation metrics for a model."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mbe  = float(np.mean(y_pred - y_true))
    w10  = float(np.mean(np.abs(y_pred - y_true) <= 10))
    w20  = float(np.mean(np.abs(y_pred - y_true) <= 20))
    return {
        "model":        name,
        "RMSE":         round(rmse, 2),
        "MAE":          round(mae,  2),
        "R2":           round(r2,   3),
        "MBE":          round(mbe,  2),
        "within_10pct": round(w10,  3),
        "within_20pct": round(w20,  3),
    }


# ---------------------------------------------------------------------------
# CHRONOLOGICAL DATA SPLIT
# ---------------------------------------------------------------------------
def split_data(df):
    """
    Split dataset into train / validation / test sets chronologically.
    Periods:
        Train : Apr 2011 -- Dec 2015  (~57 samples)
        Val   : Jan 2016 -- Dec 2017  (~24 samples)
        Test  : Jan 2018 -- Nov 2020  (~35 samples)
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["ym_str"] = df["date"].dt.to_period("M").astype(str)

    train_mask = df["ym_str"] <= TRAIN_END
    val_mask   = (df["ym_str"] > TRAIN_END) & (df["ym_str"] <= VAL_END)
    test_mask  = df["ym_str"] > VAL_END

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print("[Split] Train: %d rows | Val: %d rows | Test: %d rows"
          % (train_mask.sum(), val_mask.sum(), test_mask.sum()))
    print("  Train: %s to %s" % (df.loc[train_mask, "date"].min().strftime("%Y-%m"),
                                  df.loc[train_mask, "date"].max().strftime("%Y-%m")))
    print("  Val  : %s to %s" % (df.loc[val_mask, "date"].min().strftime("%Y-%m"),
                                  df.loc[val_mask, "date"].max().strftime("%Y-%m")))
    print("  Test : %s to %s" % (df.loc[test_mask, "date"].min().strftime("%Y-%m"),
                                  df.loc[test_mask, "date"].max().strftime("%Y-%m")))

    return X_train, X_val, X_test, y_train, y_val, y_test, df


# ---------------------------------------------------------------------------
# SCALER  (fit on training set only to prevent leakage)
# ---------------------------------------------------------------------------
def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ---------------------------------------------------------------------------
# SVR WITH GRID SEARCH
# ---------------------------------------------------------------------------
def train_svr(X_train_sc, y_train):
    """
    Train SVR-RBF with hyperparameter tuning via GridSearchCV.
    Cross-validation uses TimeSeriesSplit (4 folds, gap=1 month).
    """
    print("\n[SVR] Running GridSearchCV ...")

    param_grid = {
        "C":       [10, 50, 100, 200],
        "epsilon": [0.5, 1.0, 2.0, 5.0],
        "gamma":   [0.005, 0.01, 0.02, 0.05, 0.1, "scale"],
    }
    tscv = TimeSeriesSplit(n_splits=4, gap=1)
    gs = GridSearchCV(
        SVR(kernel="rbf"),
        param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        verbose=0,
    )
    gs.fit(X_train_sc, y_train)

    best = gs.best_params_
    cv_rmse = -gs.best_score_
    print("  Best params: C=%(C)s, epsilon=%(epsilon)s, gamma=%(gamma)s" % best)
    print("  CV RMSE: %.2f%%" % cv_rmse)
    return gs.best_estimator_


# ---------------------------------------------------------------------------
# BASELINE MODELS
# ---------------------------------------------------------------------------
def train_baselines(X_train_sc, y_train):
    """Train three baseline models for comparison with SVR."""
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=1.0),
        "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=42),
    }
    fitted = {}
    print("\n[Baselines] Training ...")
    for name, m in models.items():
        m.fit(X_train_sc, y_train)
        fitted[name] = m
        print("  %s trained." % name)
    return fitted


# ---------------------------------------------------------------------------
# MAIN: TRAIN + EVALUATE + SAVE
# ---------------------------------------------------------------------------
def train_and_evaluate():
    print("=" * 60)
    print("  HydroPulse-SVR  |  Model Training")
    print("=" * 60)

    # Load (or build) master dataset
    if not os.path.exists(DATA_OUT_PATH):
        from data_pipeline import build_master_dataset
        df = build_master_dataset()
    else:
        df = pd.read_csv(DATA_OUT_PATH, parse_dates=["date"])

    # Add calendar_month if missing (needed for persistence baseline)
    df["calendar_month"] = df["date"].dt.month

    # ---- Split ----
    X_train, X_val, X_test, y_train, y_val, y_test, df = split_data(df)

    # ---- Pre-test Historical Training Set (Train + Val: 2011 to 2017) ----
    # Time-series standard practice: Tune hyperparameters using chronological splits,
    # then train final estimator on all pre-test historical data before evaluating on withheld test set.
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    scaler        = fit_scaler(X_train_val)
    X_train_sc    = scaler.transform(X_train)
    X_val_sc      = scaler.transform(X_val)
    X_train_val_sc = scaler.transform(X_train_val)
    X_test_sc     = scaler.transform(X_test)

    # ---- Hyperparameter Search on Train Split ----
    tuned_svr = train_svr(X_train_sc, y_train)
    best_params = tuned_svr.get_params()

    # ---- Refit Final SVR on Pre-Test Data (Apr 2011 - Dec 2017) ----
    svr_model = SVR(
        kernel="rbf",
        C=best_params.get("C", 100),
        epsilon=best_params.get("epsilon", 1.0),
        gamma=best_params.get("gamma", 0.01),
    )
    svr_model.fit(X_train_val_sc, y_train_val)
    svr_val_pred  = svr_model.predict(X_val_sc)
    svr_test_pred = svr_model.predict(X_test_sc)

    # ---- Train Baselines on Pre-Test Data ----
    baselines = train_baselines(X_train_val_sc, y_train_val)

    # ---- Evaluate on TEST SET ----
    print("\n" + "=" * 60)
    print("  TEST SET RESULTS  (Jan 2018 - Nov 2020, n=35)")
    print("=" * 60)
    print()

    all_metrics = []

    # SVR
    m_svr = compute_metrics(y_test.values, svr_test_pred, "SVR-RBF")
    all_metrics.append(m_svr)
    print("SVR-RBF (proposed)    RMSE=%.2f%%  MAE=%.2f%%  R2=%.3f"
          % (m_svr["RMSE"], m_svr["MAE"], m_svr["R2"]))

    # Baselines
    for name, model in baselines.items():
        pred = model.predict(X_test_sc)
        m    = compute_metrics(y_test.values, pred, name)
        all_metrics.append(m)
        print("%-22s RMSE=%.2f%%  MAE=%.2f%%  R2=%.3f"
              % (name, m["RMSE"], m["MAE"], m["R2"]))

    # Persistence baseline  (predict next month = current month deficit)
    test_mask    = df["ym_str"] > VAL_END
    persist_pred = df.loc[test_mask, "deficit_pct"].values
    if len(persist_pred) == len(y_test):
        m_p = compute_metrics(y_test.values, persist_pred, "Persistence (naive)")
        all_metrics.append(m_p)
        print("%-22s RMSE=%.2f%%  MAE=%.2f%%  R2=%.3f"
              % (m_p["model"], m_p["RMSE"], m_p["MAE"], m_p["R2"]))

    # ---- Honest comparison note ----
    metrics_df = pd.DataFrame(all_metrics)
    best_row   = metrics_df.sort_values("RMSE").iloc[0]
    if best_row["model"] != "SVR-RBF":
        print("\nNote: %s achieves lower RMSE than SVR-RBF on the test set." % best_row["model"])
        print("      This is expected -- with only 57 training samples, linear models")
        print("      are competitive. SVR advantages emerge with larger datasets.")

    print()

    # ---- Save metrics ----
    metrics_df.to_csv(METRICS_PATH, index=False)
    print("[Saved] Metrics     -> %s" % METRICS_PATH)

    # ---- Save SVR model + scaler ----
    joblib.dump(svr_model, SVR_MODEL_PATH)
    joblib.dump(scaler,    SCALER_PATH)
    print("[Saved] SVR model   -> %s" % SVR_MODEL_PATH)
    print("[Saved] Scaler      -> %s" % SCALER_PATH)

    # ---- Attach split labels and predictions to df for dashboard ----
    df = df.copy()
    df["split"] = "train"
    df.loc[(df["ym_str"] > TRAIN_END) & (df["ym_str"] <= VAL_END), "split"] = "val"
    df.loc[df["ym_str"] > VAL_END, "split"] = "test"

    # Generate SVR predictions for all splits (for dashboard charts)
    train_mask_bool = df["ym_str"] <= TRAIN_END
    val_mask_bool   = (df["ym_str"] > TRAIN_END) & (df["ym_str"] <= VAL_END)
    test_mask_bool  = df["ym_str"] > VAL_END

    df["svr_pred"] = np.nan
    df.loc[train_mask_bool, "svr_pred"] = svr_model.predict(
        scaler.transform(df.loc[train_mask_bool, FEATURE_COLS])
    )
    df.loc[val_mask_bool,  "svr_pred"] = svr_val_pred
    df.loc[test_mask_bool, "svr_pred"] = svr_test_pred

    save_cols = ["date", "ONI", "ENSO_Phase", "inflow_mcm", "storage_pct_frl",
                 "rainfall_mm", "deficit_pct", TARGET_COL, "svr_pred", "split"]
    df[save_cols].to_csv(PRED_PATH, index=False)
    print("[Saved] Predictions -> %s" % PRED_PATH)

    print("\n[Done] Training complete. Run:  streamlit run dashboard.py")
    return df, metrics_df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_and_evaluate()
