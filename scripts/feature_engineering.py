"""
CollideX — Step 4: Feature Engineering Pipeline
====================================================
Builds the ML-ready dataset (ml_dataset.csv) for collision probability
classification from:
  • ESA CDM dataset  (CollideX/data/cdm/train_data.csv)
  • SGP4 trajectory  (CollideX/data/processed/future_positions.csv)

Output columns
--------------
  miss_distance, relative_speed, time_to_tca,
  combined_radius (engineered), rel_pos_r, rel_pos_t, rel_pos_n,
  rel_vel_r, rel_vel_t, rel_vel_n,
  mahalanobis_distance, t_j2k_ecc, c_j2k_ecc,
  t_j2k_inc, c_j2k_inc,
  risk_index  (engineered: relative_speed / miss_distance)
  risk_class  (target: 0=Low, 1=Medium, 2=High)
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.join(SCRIPT_DIR, "..")
CDM_FILE     = os.path.join(ROOT_DIR, "data", "cdm",       "train_data.csv")
TRAJ_FILE    = os.path.join(ROOT_DIR, "data", "processed", "future_positions.csv")
OUTPUT_DIR   = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "ml_dataset.csv")


# ===========================================================================
# STEP 4.1 — Load & Inspect ESA CDM Dataset
# ===========================================================================
def step_4_1_load(cdm_path: str) -> pd.DataFrame:
    print("\n[Step 4.1] Loading ESA CDM dataset …")
    df = pd.read_csv(cdm_path)
    print(f"  Rows   : {len(df):,}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Sample columns: {list(df.columns[:10])} …")
    return df


# ===========================================================================
# STEP 4.2 — Select Core Prediction Features
# (mapped to actual column names present in the ESA CDM dataset)
# ===========================================================================
CORE_FEATURES = [
    "miss_distance",          # closest approach distance (km)
    "relative_speed",         # relative velocity magnitude (km/s)
    "time_to_tca",            # time to closest approach (days)
    "relative_position_r",    # radial component of relative position
    "relative_position_t",    # transverse component
    "relative_position_n",    # normal component
    "relative_velocity_r",    # radial relative velocity
    "relative_velocity_t",    # transverse relative velocity
    "relative_velocity_n",    # normal relative velocity
    "mahalanobis_distance",   # statistical combined uncertainty metric
    "t_j2k_ecc",              # target eccentricity
    "c_j2k_ecc",              # chaser eccentricity
    "t_j2k_inc",              # target inclination
    "c_j2k_inc",              # chaser inclination
    "t_sigma_r",              # position uncertainty (target, radial)
    "c_sigma_r",              # position uncertainty (chaser, radial)
]

def step_4_2_select(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 4.2] Selecting core prediction features …")
    available = [c for c in CORE_FEATURES if c in df.columns]
    missing   = [c for c in CORE_FEATURES if c not in df.columns]
    if missing:
        print(f"  ⚠ Columns not found (skipped): {missing}")
    df_sel = df[available].copy()
    print(f"  Selected {len(available)} features: {available}")
    return df_sel


# ===========================================================================
# STEP 4.3 — Handle Missing Values
# ===========================================================================
def step_4_3_clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 4.3] Handling missing values …")
    before = len(df)
    df = df.dropna()
    after  = len(df)
    print(f"  Rows before: {before:,}  |  Rows after: {after:,}  |  Dropped: {before - after:,}")
    return df


# ===========================================================================
# STEP 4.4 — Engineered Features
# ===========================================================================
def step_4_4_engineer(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 4.4] Creating engineered features …")

    # 4.4a  combined_radius — proxy for combined cross-sectional size,
    #        derived from radial position uncertainties of both objects
    df["combined_radius"] = df["t_sigma_r"] + df["c_sigma_r"]

    # 4.4b  risk_index — speed-to-distance ratio (higher = more dangerous)
    #        Guard against division by zero
    df["risk_index"] = np.where(
        df["miss_distance"] > 0,
        df["relative_speed"] / df["miss_distance"],
        np.nan
    )
    df = df.dropna(subset=["risk_index"])

    # 4.4c  3-D relative speed magnitude cross-check
    df["rel_speed_xyz"] = np.sqrt(
        df["relative_velocity_r"] ** 2 +
        df["relative_velocity_t"] ** 2 +
        df["relative_velocity_n"] ** 2
    )

    print(f"  + combined_radius, risk_index, rel_speed_xyz")
    print(f"  Rows after engineering: {len(df):,}")
    return df


# ===========================================================================
# STEP 4.5 — Generate Target Variable  risk_class
# ===========================================================================
def step_4_5_target(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 4.5] Generating target variable (risk_class) …")

    # miss_distance is in METRES in the ESA CDM dataset
    # Thresholds calibrated to the actual data distribution:
    #   < 1,000 m  -> High Risk   (top ~10th percentile)
    #   < 5,000 m  -> Medium Risk (10th-25th percentile)
    #   >= 5,000 m -> Low Risk
    def risk_label(distance: float) -> int:
        if distance < 1_000:
            return 2   # High Risk
        elif distance < 5_000:
            return 1   # Medium Risk
        else:
            return 0   # Low Risk

    df["risk_class"] = df["miss_distance"].apply(risk_label)

    dist   = df["risk_class"].value_counts().sort_index()
    labels = {0: "Low", 1: "Medium", 2: "High"}
    for cls, label in labels.items():
        count = dist.get(cls, 0)
        pct   = count / len(df) * 100
        print(f"  Class {cls} ({label:6s}): {count:>8,}  ({pct:.1f}%)")
    return df


# ===========================================================================
# STEP 4.6 — Normalize Feature Values (StandardScaler)
# ===========================================================================
def step_4_6_normalize(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 4.6] Normalizing features …")
    feature_cols = [c for c in df.columns if c != "risk_class"]
    scaler       = StandardScaler()
    X_scaled     = scaler.fit_transform(df[feature_cols])

    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    scaled_df["risk_class"] = df["risk_class"].values
    print(f"  Scaled {len(feature_cols)} feature columns.")
    return scaled_df


# ===========================================================================
# STEP 4.7 — Save Processed ML Dataset
# ===========================================================================
def step_4_7_save(df: pd.DataFrame, output_path: str) -> None:
    print("\n[Step 4.7] Saving ml_dataset.csv …")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    size_mb = os.path.getsize(output_path) / 1_048_576
    print(f"  Saved  -> {output_path}")
    print(f"  Rows   : {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Size   : {size_mb:.2f} MB")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 62)
    print("  CollideX — Step 4: Feature Engineering Pipeline")
    print("=" * 62)

    # 4.1 Load
    df = step_4_1_load(CDM_FILE)

    # 4.2 Select features
    df = step_4_2_select(df)

    # 4.3 Drop NaNs
    df = step_4_3_clean(df)

    # 4.4 Engineer features
    df = step_4_4_engineer(df)

    # 4.5 Target variable
    df = step_4_5_target(df)

    # 4.6 Normalize
    df_final = step_4_6_normalize(df)

    # 4.7 Save
    step_4_7_save(df_final, OUTPUT_FILE)

    # -----------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("  STEP 4 COMPLETE -- ml_dataset.csv ready for ML training.")
    print("=" * 62)

    # Quick sanity check
    print("\n[Sanity Check] First 3 rows of ml_dataset.csv:")
    sample = pd.read_csv(OUTPUT_FILE, nrows=3)
    print(sample.to_string(index=False))
    print()
