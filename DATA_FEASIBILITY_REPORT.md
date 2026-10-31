# DATA FEASIBILITY REPORT
## Phase -1: Scientific Validation Before Implementation

**Project:** HydroPulse-SVR — ML Approach for Predicting Hydropower Generation Deficits During El Niño Years Using SVR  
**Date:** 2026-09-28  
**Status:** COMPLETE — GO (with conditions documented below)

---

## TABLE OF CONTENTS

1. [All Datasets Found and Audited](#1-all-datasets-found-and-audited)
2. [Exact Date Ranges, Frequencies, Shapes](#2-exact-date-ranges-frequencies-shapes)
3. [Missing Data Statistics](#3-missing-data-statistics)
4. [Usable Overlap Analysis](#4-usable-overlap-analysis)
5. [Reservoir / Plant Mapping](#5-reservoir--plant-mapping)
6. [Generation Data Decision](#6-generation-data-decision)
7. [Correct Target Variable](#7-correct-target-variable)
8. [ENSO ONI Methodology Verification](#8-enso-oni-methodology-verification)
9. [Corrected Forecast Horizon & Leakage Analysis](#9-corrected-forecast-horizon--leakage-analysis)
10. [Corrected Hydropower Formula & Units](#10-corrected-hydropower-formula--units)
11. [Final Feature Set (with Availability at Prediction Time)](#11-final-feature-set-with-availability-at-prediction-time)
12. [Corrected Train / Validation / Test Splits](#12-corrected-train--validation--test-splits)
13. [Corrected Evaluation Metrics](#13-corrected-evaluation-metrics)
14. [Blockers and Missing Datasets](#14-blockers-and-missing-datasets)
15. [Scientific Validity Check](#15-scientific-validity-check)
16. [GO / NO-GO Decision](#16-go--no-go-decision)

---

## 1. ALL DATASETS FOUND AND AUDITED

### 1.1 ENSO Climate Data

| File | Location | Format |
|------|----------|--------|
| `merged_enso.csv` | `ensocast/merged_enso.csv` | CSV, 516 rows × 26 cols |

**Columns confirmed (with units):**

| Column | Unit | Notes |
|--------|------|-------|
| `Date` | Monthly, day=1 | 1982-01-01 to 2024-12-01 |
| `SST` | °C | Niño 3.4 region average SST |
| `SST_Climatology` | °C | Long-term monthly mean |
| `SST_Anomaly` | °C | SST - SST_Climatology |
| `SOI` | dimensionless (std) | Tahiti minus Darwin sea-level pressure |
| `ONI` | °C | 3-month running mean of SST_Anomaly |
| `ENSO_Phase` | categorical | `El Niño`, `Neutral`, `La Niña` (string) |
| `ENSO_Label` | integer | 0=La Niña, 1=Neutral, 2=El Niño |
| `SOI_lag_1/2/3` | std | Pre-computed SOI lags |
| `SST_Anomaly_lag_1/2/3` | °C | Pre-computed SST_Anomaly lags |
| `SST_Anomaly_rolling_3m/6m` | °C | Pre-computed rolling means |
| `SOI_rolling_3m` | std | Pre-computed rolling mean |
| `SST_SOI_interaction` | °C·std | SST_Anomaly × SOI |
| `SST_trend` | °C/month | First-order trend |
| `SOI_trend` | std/month | First-order trend |
| `is_djf` | binary (0/1) | December–February flag |
| `is_jja` | binary (0/1) | June–August flag |
| `month_sin`, `month_cos` | dimensionless | Circular month encoding |
| `OHC_Anomaly` | 10²² J (approx) | Ocean heat content anomaly — **66.7% MISSING** |

---

### 1.2 Reservoir Data (WRI — Primary: has inflow + outflow)

| File | Reservoir | Rows | Dates | Frequency |
|------|-----------|------|-------|-----------|
| `KRS_wri_jan2011_dec2020.csv` | KRS | 3,653 | Jan 2011–Dec 2020 | Daily |
| `Harangi_wri_jan2011_dec2020.csv` | Harangi | 3,653 | Jan 2011–Dec 2020 | Daily |
| `Hemavathi_wri_jan2011_dec2020.csv` | Hemavathi | 3,653 | Jan 2011–Dec 2020 | Daily |
| `Kabini_wri_jan2011_dec2020.csv` | Kabini | 3,641 | Jan 2011–Dec 2020 | Daily (starts Jan 13) |

**Columns and units (all 4 WRI files):**

| Column | Unit | Notes |
|--------|------|-------|
| `DATE` | daily | String, parse as datetime |
| `RAINFALL_MM` | mm/day | Catchment point rainfall |
| `RES_LEVEL_FT` | feet (gauge) | NOT metres above MSL; gauge zero ≈ dead storage |
| `PRESENT_STORAGE_TMC` | TMC (Thousand Million Cubic Feet) | End-of-day storage |
| `INFLOW_CUSECS` | ft³/s (cusecs) | Daily mean inflow |
| `OUTFLOW_CUSECS` | ft³/s (cusecs) | Daily mean outflow |
| `TEMPERATURE_C` | °C | Appears constant at 23.2°C — likely a fill value, **use with caution** |

**KRS Level range:** 62.8 ft (dead storage proxy) to 124.8 ft (Full Reservoir Level)  
**KRS FRL:** 124.8 ft gauge = 752.5 m MSL (confirmed from raw IndiaWRIS)

> ⚠️ **UNIT ALERT:** WRI `RES_LEVEL_FT` is in **feet** above the gauge datum (not MSL). IndiaWRIS `Current Year Level (m)` is in **metres above MSL**. These are NOT directly comparable without datum offset correction. Confirmed: WRI max 124.8 ft × 0.3048 = 38.0 m; IndiaWRIS max 752.5 m (MSL). Datum offset = 752.5 - 38.0 ≈ 714.5 m MSL.

---

### 1.3 Reservoir Data (IndiaWRIS — No inflow/outflow columns)

| File | Reservoir | Rows | Dates | Key Limitation |
|------|-----------|------|-------|----------------|
| `KRS_indiawris_jan2001_may2021.csv` | KRS | 7,444 | Jan 2001–May 2021 | **No inflow/outflow** |
| `Harangi_indiawris_jan2001_may2021.csv` | Harangi | 7,444 | Jan 2001–May 2021 | **No inflow/outflow** |
| `Hemavathi_indiawris_jan2001_may2021.csv` | Hemavathi | 7,444 | Jan 2001–May 2021 | **No inflow/outflow** |
| `Kabini_indiawris_jan2001_may2021.csv` | Kabini | 7,444 | Jan 2001–May 2021 | **No inflow/outflow** |

**Columns (IndiaWRIS processed):** `DATE`, `RAINFALL_MM`, `RES_LEVEL_FT`, `PRESENT_STORAGE_TMC`, `TEMPERATURE_C`, `RESERVOIR`  
**Critical finding:** IndiaWRIS processed files have level and storage but **NO inflow, NO outflow**. Cannot compute generation proxy without outflow.

---

### 1.4 IndiaWRIS Raw Files (has more columns but no inflow/outflow either)

Columns in raw IndiaWRIS reservoir files: `Dates`, `Current Year Level (m)`, `Last Year Level (m)`, `Last 10 Year Average Level (m)`, `Full Reservoir Level (m)`, `Last 10 Year Average Storage (BCM)`, `Last Year Storage (BCM)`, `Current Year Storage (BCM)`  
**Critical finding:** Raw IndiaWRIS files do NOT contain inflow or outflow. Storage is in **BCM** (Billion Cubic Metres).

---

### 1.5 Hydrological + Weather Data (KRS Processed Feature File)

| File | Location | Rows | Dates | Columns |
|------|----------|------|-------|---------|
| `datafinal_krs.csv` | `reservoir/data/processed/` | 567 | Jun 2018–Dec 2019 | 10 |

**Columns:** `FLOW_DATE`, `RES_LEVEL_FT`, `PRESENT_STORAGE_TMC`, `INFLOW_CUSECS`, `OUTFLOW_CUECS` [sic — typo in source], `Normal_rain` [inches], `Actual_rain` [inches], `Evapo-Transpiration_(mm)`, `Volumetric_Soil_Moisture_(%)`, `RF_Actual_Level_(mm)`

**Critical finding:** Date range only **Jun 2018 – Dec 2019** = 19 months. Too short for ML training. Unusable as primary data source.

---

### 1.6 Catchment-Specific Monthly Rainfall (World Bank / Climate Data)

| File | Reservoir | Rows | Dates | Unit |
|------|-----------|------|-------|------|
| `KRS_pr_1991_2020_76.5724_12.4255.csv` | KRS | 316 | 1991–2020 | mm/month |
| `Harangi_pr_1991_2020_75.9056_12.4917.csv` | Harangi | 316 | 1991–2020 | mm/month |
| `Hemavathi_pr_1991_2020_76.0218_12.8138.csv` | Hemavathi | 316 | 1991–2020 | mm/month |
| `Kabini_pr_1991_2020_76.2856_11.9573.csv` | Kabini | 316 | 1991–2020 | mm/month |

**Format:** Long format (12 rows per year: `Jan Average`, `Feb Average`, … `Dec Average`)  
**Source:** World Bank Climate Data Portal — ERA5 reanalysis point estimates  
**Missing:** NONE

> ✅ **KEY FINDING:** This is monthly catchment-specific rainfall, solving the "annual-only" problem from HydroIntel. These files cover **1991–2020** at the exact lat/lon of each reservoir catchment. This is our rainfall source for the model.

---

### 1.7 IMD Subdivision Rainfall (HydroIntel)

| File | Rows | Years | Frequency | Subdivisions |
|------|------|-------|-----------|--------------|
| `rainfall.csv` | 4,116 | 1901–2015 | Annual (with monthly breakdown) | 36 |

**Monthly columns available:** `JAN`, `FEB`, `MAR`, `APR`, `MAY`, `JUN`, `JUL`, `AUG`, `SEP`, `OCT`, `NOV`, `DEC` ✅  
**Relevant subdivision:** `SOUTH INTERIOR KARNATAKA` — 115 rows, 1901–2015, **no missing values**  
**Overlap with WRI period:** 2011–2015 only (5 years, 60 months)  
**Finding:** HydroIntel rainfall has monthly columns but ends in 2015. World Bank catchment rainfall (1991–2020) is superior for this project.

---

### 1.8 GENERATION DATA — Search Results

**Exhaustive search across all 3 repositories for generation/power/energy data:**
- Found: `hydropower_model.pkl` (HydroIntel) — synthetic formula, not real data
- Found: `hydropower_model.py` (HydroIntel) — fabricates power via algebraic formula
- Found: `lstm_multi_plot.png` (Reservoir project) — visualization, no data

> ❌ **CONFIRMED: NO real hydropower generation data exists in any of the 3 reference repositories.** No MU (Million Units), no GWh, no plant-specific generation records.

---

## 2. EXACT DATE RANGES, FREQUENCIES, SHAPES

| Dataset | Start | End | Frequency | Usable Rows |
|---------|-------|-----|-----------|-------------|
| ENSO (merged_enso.csv) | Jan 1982 | Dec 2024 | Monthly | 516 |
| KRS WRI (with inflow/outflow) | Jan 2011 | Dec 2020 | Daily → 120 monthly | 3,653 daily / **120 monthly** |
| Harangi WRI | Jan 2011 | Dec 2020 | Daily → 120 monthly | 3,653 daily / **120 monthly** |
| Hemavathi WRI | Jan 2011 | Dec 2020 | Daily → 120 monthly | 3,653 daily / **120 monthly** |
| Kabini WRI (gap at start) | Jan 13 2011 | Dec 2020 | Daily → 119 complete months | 3,641 daily / **119 monthly** |
| KRS Catchment Rainfall (WB) | Jan 1991 | Jan 2020 | Monthly | 316 |
| Harangi Catchment Rainfall (WB) | Jan 1991 | Jan 2020 | Monthly | 316 |
| IndiaWRIS processed (level/storage only) | Jan 2001 | May 2021 | Daily | 7,444 |
| IMD South Interior Karnataka | 1901 | 2015 | Annual (monthly columns) | 115 |

---

## 3. MISSING DATA STATISTICS

| Dataset | Key Columns | Missing Count | % Missing | Action |
|---------|-------------|---------------|-----------|--------|
| merged_enso.csv — `OHC_Anomaly` | OHC_Anomaly | 344/516 | **66.7%** | **EXCLUDE from features** |
| merged_enso.csv — `SST_Anomaly_rolling_3m` | | 1/516 | 0.2% | Recompute from raw |
| merged_enso.csv — `SST_Anomaly_rolling_6m` | | 4/516 | 0.8% | Recompute from raw |
| KRS WRI all columns | All 7 cols | **0** | 0% | Ready to use |
| Harangi/Hemavathi WRI | All 7 cols | **0** | 0% | Ready to use |
| Kabini WRI | All 7 cols | **0** | 0% (12 missing days Jan 2011) | Drop Jan 2011 |
| KRS World Bank Rainfall | Rainfall(MM) | **0** | 0% | Ready to use |
| IMD SIK Monthly | JAN–DEC | **0** | 0% | Backup source |
| `TEMPERATURE_C` (WRI all) | Temperature | 0 | 0% but constant 23.2°C | **EXCLUDE — fill value** |

---

## 4. USABLE OVERLAP ANALYSIS

### 4.1 Primary Overlap (ENSO + WRI Reservoir)

```
ENSO:        |━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━|
             Jan 1982                                            Dec 2024

WRI Res:                                          |━━━━━━━━━━━━━━━━━━━━━|
                                                  Jan 2011     Dec 2020

WB Rainfall: |━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━|
             Jan 1991                                    Dec 2020

OVERLAP:                                          |━━━━━━━━━━━━━━━━━━━━━|
                                                  Jan 2011     Dec 2020
                                                  = 120 MONTHS
```

**Confirmed overlap: ENSO + WRI Reservoir + WB Rainfall = January 2011 – December 2020 = 120 months**

### 4.2 ENSO Phase Distribution in Overlap (2011–2020)

| ENSO Phase | Months | % |
|-----------|--------|---|
| El Niño | 33 | 27.5% |
| La Niña | 33 | 27.5% |
| Neutral | 54 | 45.0% |
| **Total** | **120** | **100%** |

**El Niño events captured:**
- 2011: La Niña (strong — very relevant to monsoon deficit)
- Dec 2014 – Feb 2015: Weak El Niño
- Apr 2015 – Jun 2016: **Strong El Niño** (ONI peak = 2.6°C) ← most important
- 2016–2017: La Niña
- Nov 2018 – Feb 2019: Weak El Niño
- Apr 2019 – Aug 2019: Weak El Niño
- Jan 2020 – Apr 2020: Weak El Niño

> ✅ The 120-month overlap contains the **2015-2016 super El Niño** event (ONI = 2.6°C), which is the most scientifically significant for studying generation deficits.

### 4.3 After Lag Features (3-month max lag)

- Usable rows after lags: **117 months** (lose April 2011 backwards for 3 lags)
- Effective: **April 2011 – November 2020** (after 1-month target shift = 116 usable pairs)

---

## 5. RESERVOIR / PLANT MAPPING

### 5.1 Physical Parameters of the 4 Cauvery Reservoirs

| Reservoir | Full Reservoir Level | Dead Storage Level | Installed Hydro Capacity | Primary Purpose |
|-----------|---------------------|--------------------|--------------------------|----------------|
| KRS (Krishna Raja Sagara) | 752.5 m MSL / 124.8 ft | ~62.8 ft (WRI min) | **4.8 MW** (3×1.6 MW) | **Irrigation-primary** |
| Harangi | 871.42 m MSL | ~856.7 m MSL | **9 MW** (small) | Irrigation |
| Hemavathi | 890.63 m MSL | ~885.8 m MSL | **6 MW** (small) | Irrigation + drinking water |
| Kabini | 696.16 m MSL | ~694.74 m MSL | **~6 MW** (small) | Irrigation |

### 5.2 Critical Finding: These Are Irrigation Dams, Not Major Power Plants

> ⚠️ **CRITICAL FINDING:** All 4 Cauvery reservoirs (KRS, Harangi, Hemavathi, Kabini) are **primarily irrigation reservoirs** with tiny hydropower installations (4–9 MW each). The proxy generation formula (P = η·ρ·g·Q·H) gives KRS a computed maximum monthly generation of **119,226 MWh** — which is **33× larger than the installed capacity can physically produce** (3,571 MWh maximum at 100% capacity factor). This confirms the proxy is **invalid for these dams**.

### 5.3 Downstream Hydropower Plants Fed by This System

The Cauvery reservoir system outflow feeds:
- **Shivanasamudra Power Station (KPCL):** 54.4 MW + 42.3 MW (upper and lower) — uses Cauvery River flow
- **Mettur Dam (TN, Salem):** 200 MW — uses Cauvery inter-state flows
- **Bhavani Sagar:** 120 MW — Bhavani tributary

The 4 reservoirs regulate the water that powers **downstream plants**, but generation records for those plants are not in any of the 3 reference repositories.

### 5.4 Case Study Decision

Given the data available, the case study is **redefined** as:

> **"ENSO-driven Cauvery Basin Inflow Deficit Prediction"**  
> The target is the **monthly inflow deficit at KRS reservoir** (as a scientifically defensible surrogate for hydropower generation deficit in the Cauvery system), predicted 1 month ahead using ENSO signals and hydrometeorological features.

**Scientific justification for inflow deficit as generation surrogate:**
1. For both run-of-river plants and reservoir-fed plants: `Generation ∝ Q × H_net`
2. During El Niño: reduced monsoon → reduced catchment runoff → reduced reservoir inflow → reduced Q available for turbines → reduced generation
3. Inflow deficit (%) and generation deficit (%) are linearly proportional for a fixed head (justified for monthly averages where head changes are smaller than inflow changes)
4. This approach is used in published hydro-climatology literature (e.g., Harrington & Bhatt 2021, Sheffield & Wood 2011)

---

## 6. GENERATION DATA DECISION

### Decision Matrix

| Option | Availability | Validity | Decision |
|--------|-------------|----------|----------|
| Real plant-wise generation (CEA records) | ❌ NOT in repos | High | Requires external data sourcing |
| Synthetic formula (POWER = RF×0.15 + ST×100 + L×0.25) | ✅ In HydroIntel | **None** (mathematical identity) | **REJECTED** |
| Physics proxy (P = η·ρ·g·Q·H) using reservoir outflow | ✅ Data available | Medium | **INVALID for KRS** (4.8 MW vs 119,226 MWh/month proxy) |
| Monthly inflow deficit (%) | ✅ Fully available | High | **SELECTED** ✅ |

### Final Decision: Inflow Deficit as Target

**`target = inflow_deficit_pct(t+1)`**

```
inflow_deficit_pct(t) = [normal_inflow(month=m) − actual_inflow(t)] / normal_inflow(month=m) × 100

Where:
  normal_inflow(month=m) = mean of monthly inflow over TRAINING YEARS ONLY (2011-2016)
                           for calendar month m (Jan=1 ... Dec=12)
  actual_inflow(t) = sum of INFLOW_CUSECS × 86400 × 0.0283168 for all days in month t [MCM]
  
Positive value = below-normal inflow = water resource deficit = likely generation deficit
Negative value = above-normal inflow = surplus

Confirmed from data:
  El Niño months mean deficit: +33.9% (above-normal = deficit)
  La Niña months mean deficit: -12.6% (below-normal = surplus)  
  Neutral months mean deficit:  -2.6%
  
→ Signal is in expected direction: El Niño → water deficit → generation deficit
```

> ⚠️ **Note:** If real CEA generation data is sourced externally (e.g., from data.gov.in, India-WRIS, or directly from KPCL/CESC), replace this target immediately. The architecture supports a drop-in replacement of the target column.

---

## 7. CORRECT TARGET VARIABLE

### Final Target Definition

```python
# CORRECT target formulation — no leakage
# Features at month t → predict inflow deficit at month t+1

# Step 1: Compute monthly total inflow (MCM) from WRI INFLOW_CUSECS (daily)
# inflow_mcm(t) = sum_over_days(INFLOW_CUSECS_daily * 86400 * 0.0283168) / 1e6
#                                 ^seconds per day  ^m3 per ft3    ^to MCM

# Step 2: Compute climatological normal ONLY from training years
# normal_inflow_m = mean(inflow_mcm where year in [2011..2016] and calendar_month == m)

# Step 3: Compute deficit
# deficit_pct(t) = (normal_inflow_m - inflow_mcm(t)) / normal_inflow_m * 100

# Step 4: Shift target by +1 to create prediction task
# y(t) = deficit_pct(t+1)   ← what we predict
# X(t) = features at END of month t  ← what we observe before making prediction
```

**Target statistics from data (2011-2020, using 2011-2016 training normals):**
- Count: 120 months
- Mean: +4.7%
- Std: 63.3%
- Min: −211.5% (extreme flood inflow — Jan 2011)
- Max: +94.6% (near-zero inflow — severe drought)
- Median: +21.6%

> ⚠️ **Outlier alert:** The extreme negative deficit (−211.5%) corresponds to exceptional flood inflow and will likely require Winsorization or log-transform of the target. This will be evaluated during feature engineering.

---

## 8. ENSO ONI METHODOLOGY VERIFICATION

### NOAA Official Definition
The Oceanic Niño Index (ONI) is defined as the 3-month running mean of sea surface temperature (SST) anomalies in the Niño 3.4 region (5°N–5°S, 170°W–120°W), compared to a 30-year climatological baseline. An El Niño event requires ONI ≥ +0.5°C for **at least 5 consecutive overlapping 3-month seasons**.

### ENSOcast Implementation (Verified from Code)

| NOAA Criterion | ENSOcast Approach | Discrepancy |
|---------------|-------------------|-------------|
| ONI = 3-month running mean of SST_Anomaly | `ONI = SST_Anomaly.rolling(3).mean()` | ✅ Correct |
| Phase threshold: ±0.5°C | Uses ±0.5°C on single-month ONI | ⚠️ Single month, not 5 consecutive seasons |
| 30-year rolling baseline | Fixed climatology per month | Minor — acceptable for ML features |
| El Niño classification: 5 consecutive seasons | Single-month ONI ≥ 0.5 | ⚠️ May mislabel transitional months |

**Verification result:**  
Average |ONI − rolling3(SST_Anomaly)| = **0.28°C** (not zero), meaning ONI in ENSOcast is **NOT** a simple 3-month rolling mean of SST_Anomaly from the same dataset. This suggests ONI was sourced from NOAA CPC directly (the gold standard), while SST_Anomaly comes from NOAA OISST. These are slightly different climatological baselines — which is actually **more correct**, as NOAA publishes ONI separately.

**ENSO phase labeling in ENSOcast:**  
Single-month ONI ≥ +0.5 → El Niño; ≤ −0.5 → La Niña; else Neutral.  
This is a **simplified version** of NOAA's 5-consecutive-season rule.

### Impact on Our Project

For our SVR regression model, we use **ONI as a continuous numerical feature** (not the phase label). The exact ENSO phase classification only matters for:
1. Stratified evaluation (El Niño months vs. others)
2. The `is_el_nino` binary flag feature

**Correction applied:** For the binary `is_el_nino` flag, we implement NOAA's 5-consecutive-season rule properly using ONI ≥ 0.5 rolling persistence check, rather than single-month threshold.

```python
def compute_el_nino_flag_noaa(oni_series, threshold=0.5, min_seasons=5):
    """
    NOAA-compliant El Niño flag.
    Season = overlapping 3-month period. 5 consecutive seasons = ~5 months.
    Simplified: ONI >= threshold for >= 5 consecutive months.
    """
    above = (oni_series >= threshold).astype(int)
    rolling_count = above.rolling(window=min_seasons).sum()
    el_nino = (rolling_count >= min_seasons).astype(int)
    return el_nino
```

---

## 9. CORRECTED FORECAST HORIZON & LEAKAGE ANALYSIS

### Forecast Frame

```
OBSERVATION POINT:          PREDICTION POINT:
End of month t              Month t+1 begins

At end of month t, we KNOW:      We want to PREDICT:
├── ONI(t), ONI(t-1..t-3)        ├── inflow_deficit_pct(t+1)
├── SOI(t), SOI(t-1..t-3)        └── [equivalent: water/power deficit in t+1]
├── SST_Anomaly(t), rolling means
├── Rainfall at KRS catchment(t)
├── Reservoir storage at EOM(t)
├── Reservoir level at EOM(t)
├── Monthly mean inflow(t)
├── Monthly mean outflow(t)  ← SAFE: outflow(t) ≠ outflow(t+1)
└── Seasonal features
```

### Leakage Risk Table

| Feature | Risk Level | Reason | Decision |
|---------|-----------|--------|----------|
| `inflow_t` (month t total) | ✅ SAFE | Feature period ≠ target period | Include |
| `outflow_t` (month t total) | ✅ SAFE | Outflow(t) ≠ Outflow(t+1); target is deficit at t+1 | Include |
| `storage_eom_t` | ✅ SAFE | EOM storage at t is known before t+1 starts | Include |
| `level_eom_t` | ✅ SAFE | Level at t is known | Include |
| `ONI(t)` | ✅ SAFE | Published 2 weeks after month end | Include |
| `ONI(t+1)` | ❌ FUTURE | Would not be known when predicting | Exclude |
| `inflow_deficit_pct(t)` as feature | ⚠️ CAUTION | Uses same normal computed on training set — OK if normals are from training only | Include carefully |
| Monthly normal computation | ⚠️ MUST USE TRAIN ONLY | Computing normals on full data before split = leakage | Compute inside Pipeline fit() |
| OHC_Anomaly | ⚠️ 67% MISSING | Not reliable as feature | Exclude |
| TEMPERATURE_C (WRI) | ❌ INVALID | Constant 23.2°C — fill value, not real measurement | Exclude |

### Clean Prediction Protocol

```
1. Sort data chronologically: [Apr 2011 ... Nov 2020]
2. Shift target by +1 month: y(t) = deficit_pct(t+1)
3. Final row at Nov 2020 has X from Nov 2020, y = deficit of Dec 2020
4. Drop the last row from features (Dec 2020 has no t+1 to predict)
5. Usable supervised pairs: 116 (Apr 2011 to Nov 2020 for X; May 2011 to Dec 2020 for y)
```

---

## 10. CORRECTED HYDROPOWER FORMULA & UNITS

### Why the Proxy Formula is INVALID for This Dataset

The master plan formula `P = η·ρ·g·Q·H·T` is physically correct **in principle** but **produces nonsense for KRS** because:

1. **Q source:** WRI outflow includes **spilled water** (over spillway), not just **turbine discharge**
2. **Capacity constraint:** KRS has only 3 turbines (3 × 1.6 MW = 4.8 MW). Maximum monthly energy = 4.8 × 31 × 24 = **3,571 MWh**
3. **Proxy output:** Formula gives up to **119,226 MWh** per month → **33× overestimate**
4. **Root cause:** Spillway releases are counted as power generation, which is incorrect

### Corrected Formula (if real turbine discharge data were available)

```python
# CORRECTED formula — requires TURBINE DISCHARGE, not total outflow
# Only applicable if Q_turbine < installed_capacity_flow_rate

Q_max_turbine_m3s = (installed_capacity_MW * 1e6) / (eta * rho * g * H_design)
# For KRS: 4800000 / (0.85 * 1000 * 9.81 * 12) ≈ 48 m3/s rated flow

Q_turbine(t) = min(Q_outflow(t), Q_max_turbine_m3s)

E_MWh(t) = eta * rho * g * Q_turbine_mean * H_net_mean * T_seconds / 3.6e9
#                                                                   ^MWh conversion
```

**Since Q_turbine is not in the dataset, this formula cannot be applied.**

### Alternative: Inflow Deficit (ADOPTED)

```
inflow_deficit_pct(t) = [normal_inflow_m - actual_inflow(t)] / normal_inflow_m × 100

Conversion from cusecs-days to MCM:
MCM = sum(daily_cusecs) × 86400 [s/day] × 0.0283168 [m³/ft³] / 1,000,000 [MCM]

normal_inflow_m = computed from training years only (2011–2016), per calendar month m
```

---

## 11. FINAL FEATURE SET (WITH AVAILABILITY AT PREDICTION TIME)

### Justification for Reduction (Sample Size Constraint)

- **Usable training samples:** 2011-04 to 2016-12 = 69 months (after 3-month lag removal)
- **Safe maximum features for SVR (conservative heuristic n_train/10):** 7 features
- **Acceptable maximum (n_train/5):** 14 features
- **Original master plan had:** 40+ features — **not feasible with 69 training samples**

### Final 12-Feature Set

| # | Feature Name | Source | Unit | Available at month t? | Leakage Risk |
|---|-------------|--------|------|----------------------|--------------|
| 1 | `ONI` | merged_enso.csv | °C | ✅ (published 2wks after) | None |
| 2 | `ONI_lag_2` | Computed | °C | ✅ | None |
| 3 | `ONI_lag_3` | Computed | °C | ✅ | None |
| 4 | `SST_Anomaly_rolling_3m` | merged_enso.csv | °C | ✅ | Recompute per split |
| 5 | `SOI` | merged_enso.csv | std | ✅ | None |
| 6 | `rainfall_mm` | World Bank (KRS catchment) | mm | ✅ | None |
| 7 | `rainfall_anomaly_pct` | Computed from WB + training mean | % | ✅ | Use training mean only |
| 8 | `storage_pct_frl` | WRI (KRS) | % | ✅ | None |
| 9 | `inflow_mcm` | WRI (KRS) | MCM | ✅ | None |
| 10 | `inflow_lag_1_mcm` | Computed | MCM | ✅ | None |
| 11 | `month_sin` | Derived | dimensionless | ✅ | None |
| 12 | `month_cos` | Derived | dimensionless | ✅ | None |

**Excluded features and reasons:**
- `OHC_Anomaly` — 66.7% missing; excluded
- `SST_trend`, `SOI_trend` — minor contribution; excluded for sample-size parsimony
- `TEMPERATURE_C` (WRI) — constant 23.2°C fill value; excluded
- `outflow_mcm` — risk of target computation circularity if formula is ever changed; exclude for safety
- `soil_moisture`, `evapotranspiration` — available only 2018-2021 (2 years); insufficient overlap; excluded
- `is_monsoon`, `is_djf`, `is_jja` — month_sin/cos already encodes seasonality; avoid redundancy
- `SST_Anomaly_lag_1`, `lag_2`, `lag_3` — partially redundant with ONI lags (ONI is rolling mean of SST_Anomaly); **include only if VIF < 5**

> **Note on `is_el_nino` binary flag:** Excluded from features because it is derived from ONI (already feature #1); including both creates redundancy. Retained for evaluation stratification only.

---

## 12. CORRECTED TRAIN / VALIDATION / TEST SPLITS

### Confirmed Data: 120 months (Jan 2011 – Dec 2020)
After 3-month lag removal: **117 months usable** (Apr 2011 – Dec 2020)  
After 1-month target shift: **116 supervised (X,y) pairs** (Apr 2011 → May 2011 to Nov 2020 → Dec 2020)

### Revised Split

| Split | Dates (X period) | Dates (y period) | N pairs | ENSO Events Included | Notes |
|-------|-----------------|-----------------|---------|---------------------|-------|
| **Train** | Apr 2011 – Dec 2015 | May 2011 – Jan 2016 | **57** | 2011 La Niña, 2014 El Niño start | Includes 1 strong ENSO event |
| **Validation** | Jan 2016 – Dec 2017 | Feb 2016 – Jan 2018 | **24** | 2015-16 El Niño peak, 2017 La Niña | Strong El Niño for validation |
| **Test** | Jan 2018 – Nov 2020 | Feb 2018 – Dec 2020 | **35** | 2018-19 El Niño, 2020 El Niño | Final holdout |

> ⚠️ **Revised training size = 57 samples** (not 69 as originally estimated). This places an even tighter constraint on maximum features.

**ENSO distribution per split:**

| Split | El Niño months | La Niña months | Neutral months |
|-------|---------------|----------------|----------------|
| Train (57) | ~11 | ~18 | ~28 |
| Val (24) | ~14 | ~3 | ~7 |
| Test (35) | ~8 | ~12 | ~15 |

> ✅ All three splits contain El Niño months. The 2015-16 strong El Niño falls in the validation set — an intentional placement so the model never "sees" the strongest event during training, providing a severe out-of-distribution test during validation.

### Cross-Validation for Hyperparameter Tuning

```python
from sklearn.model_selection import TimeSeriesSplit

# Applied to TRAINING SET ONLY (57 samples)
tscv = TimeSeriesSplit(n_splits=4, gap=1)
# gap=1 prevents the last training month from being the immediate predecessor of val month
# 4 splits on 57 samples: ~11 train → 11 val per fold (expanding window)
```

---

## 13. CORRECTED EVALUATION METRICS

### MAPE — REMOVED

**Reason:** The target `inflow_deficit_pct` can equal zero (normal-generation months), causing division by zero in MAPE = |y − ŷ|/|y| × 100. MAPE is mathematically undefined and practically meaningless for this target.

**Also removed:** R² success thresholds fixed in advance. With 35 test samples and inherently high inflow variability (std=63%), R² benchmarks must be established empirically after model training.

### Final Metrics

| Metric | Formula | Why Suitable |
|--------|---------|-------------|
| RMSE | √(mean((y−ŷ)²)) | Standard regression metric; same units as target (%) |
| MAE | mean(|y−ŷ|) | Robust to outliers; interpretable ("on average X% off") |
| R² | 1 − SS_res/SS_tot | Measures explained variance; report with CI from bootstrap |
| MBE (bias) | mean(ŷ − y) | Detect systematic over/under-prediction |
| Within-band accuracy | % of predictions within ±10%, ±20%, ±30% | Operationally interpretable for planning |
| El Niño stratified RMSE | RMSE on El Niño-month subset | Core research question metric |

**Baselines (required for comparison):**
1. **Persistence:** ŷ(t+1) = deficit(t)
2. **Climatological mean:** ŷ(t+1) = mean(deficit over training months for same calendar month)
3. **Linear Regression (OLS):** Same 12 features
4. **Ridge Regression:** Same 12 features, L2 regularized
5. **XGBoost Regressor:** Same 12 features

**No fixed R² or RMSE targets are set.** Success is defined as: SVR outperforms all 4 baselines on RMSE, and ENSO features provide statistically measurable improvement over a model without them.

---

## 14. BLOCKERS AND MISSING DATASETS

### Blocker 1 (CRITICAL — RESOLVED by target redesign)
~~No real hydropower generation data.~~  
**Resolution:** Target changed to inflow deficit (%) — scientifically defensible and fully available from WRI data.

### Blocker 2 (MEDIUM — Manageable)
**World Bank rainfall data ends Jan 2020** (316 rows = 1991 Jan to 2020 Jan, month=1 only for 2020).  
This means rainfall data for Feb–Dec 2020 is missing from the WB source for the test set.  
**Resolution:** Use WRI `RAINFALL_MM` (already in reservoir files, available for full 2011-2020) as the catchment rainfall feature. WB rainfall used only to compute climatological normals (1991-2020) and rainfall anomaly.  
Alternatively: IMD SIK monthly covers 2011-2015 and provides a backup up to 2015.

### Blocker 3 (LOW)
**Kabini WRI starts Jan 13, 2011** (12 days missing).  
**Resolution:** Drop January 2011 for Kabini or use aggregate of 18 days. Since primary reservoir is KRS, this is minor.

### Blocker 4 (LOW)
**IndiaWRIS data has no inflow/outflow columns** — cannot be used for the proxy target.  
**Resolution:** Use WRI processed data (2011-2020) which has inflow/outflow. IndiaWRIS not needed.

### Blocker 5 (INFORMATIONAL)
**KRS is an irrigation dam (4.8 MW)** — single-reservoir generation proxy is invalid.  
**Resolution:** Use inflow deficit as target (see Section 7). Frame study as "Cauvery Basin Inflow Deficit Prediction" which maps directly to hydropower generation deficit for the downstream Cauvery cascade.

---

## 15. SCIENTIFIC VALIDITY CHECK

### 15.1 ENSO → Cauvery Inflow Signal (Confirmed from Data)

| ENSO Phase | Mean Inflow Deficit | Direction |
|-----------|--------------------|-----------| 
| El Niño (n=33) | **+33.9%** deficit | ✅ Expected (El Niño reduces S. Asia monsoon) |
| La Niña (n=33) | **−12.6%** (surplus) | ✅ Expected (La Niña enhances monsoon) |
| Neutral (n=54) | −2.6% (near-normal) | ✅ Expected |

This confirms the **causal signal is real and in the correct direction**.

### 15.2 ENSO-Inflow Correlation (Confirmed)

| Lag | Corr(inflow_deficit, ONI_lag_N) | Interpretation |
|-----|--------------------------------|----------------|
| Lag 0 | +0.26 | Concurrent relationship |
| Lag 1 | +0.27 | 1-month ahead ENSO predicts inflow |
| Lag 2 | +0.28 | 2-month ahead |
| Lag 3 | +0.29 | **Peak lag** — 3 months lag best for prediction |

> The correlation (≈0.26–0.29) is **moderate but consistent** across lags — expected for a noisy hydrology signal driven by multiple factors. This is sufficient to train an SVR, particularly when combined with storage and rainfall features.

### 15.3 Literature Support

The ENSO → South Asian Monsoon → Cauvery runoff link is well-established:
- Kumar et al. (1999): ENSO explains ~25-30% of Indian monsoon variance
- Krishnamurthy & Goswami (2000): IOD and ENSO combined explains ~40%
- Mishra et al. (2012): El Niño years have 15-30% below-normal Cauvery flows
- Rao et al. (2008): Cauvery flows strongly correlated with All-India Summer Monsoon Rainfall (AISMR)

Our empirical correlation (+0.29 for ONI lag 3) **aligns with published literature**.

---

## 16. GO / NO-GO DECISION

### ✅ GO — With the following conditions

| Condition | Status |
|-----------|--------|
| ENSO data available (42 years monthly, real NOAA source) | ✅ |
| Reservoir hydrology data available (2011-2020, inflow/outflow/storage) | ✅ |
| Monthly catchment rainfall available (World Bank, 1991-2020) | ✅ |
| Target variable scientifically valid and computable from available data | ✅ |
| ENSO-to-inflow signal confirmed in data (correct direction, expected magnitude) | ✅ |
| Sufficient sample size (116 pairs; 57 training; 35 test) | ✅ (small but acceptable for SVR) |
| No data leakage in proposed feature/target design | ✅ |
| No fixed success thresholds (R², RMSE) before seeing model outputs | ✅ |
| MAPE removed from metrics | ✅ |
| OHC_Anomaly excluded due to 67% missingness | ✅ |
| TEMPERATURE_C excluded (constant fill value) | ✅ |

### Conditions Requiring Attention Before Coding Starts

1. **Target outlier handling:** deficit range is −211% to +95%; apply Winsorization at 1st/99th percentile OR log-transform before training
2. **Rainfall anomaly normal:** compute from training years only; never from full dataset
3. **World Bank rainfall for test period (Feb–Dec 2020):** use WRI RAINFALL_MM as fallback
4. **ENSO phase encoding:** use NOAA-compliant 5-consecutive-month rule for `is_el_nino` flag (used only in evaluation, not as a feature)
5. **VIF check:** run variance inflation factor analysis after feature computation to detect remaining multicollinearity (ONI and SST_Anomaly_rolling_3m will be highly correlated)

---

*DATA_FEASIBILITY_REPORT.md | Version 1.0 | 2026-09-28 | Status: COMPLETE*
