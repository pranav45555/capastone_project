"""
CollideX -- Step 7.2
Prepare per-satellite LSTM time-series dataset.

Strategy
--------
future_positions_with_distance.csv has 4 time steps per satellite:
  horizon_h = [1, 6, 12, 24]

For each satellite we build ONE training sequence:
  Input  : [h1, h6, h12]   (3 steps x 6 features)
  Target : [h24]            (6 features to predict)

Features per step:
  future_x_km, future_y_km, future_z_km,
  vel_x_km_s,  vel_y_km_s,  vel_z_km_s

Output: CollideX/data/processed/lstm_dataset.csv
  - X_h1_x ... X_h12_vz   (3 * 6 = 18 input columns)
  - y_x, y_y, y_z, y_vx, y_vy, y_vz  (6 target columns)
  - norad_id  (for traceability)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.join(SCRIPT_DIR, "..")
INPUT_FILE  = os.path.join(ROOT_DIR, "data", "processed",
                            "future_positions_with_distance.csv")
OUTPUT_FILE = os.path.join(ROOT_DIR, "data", "processed", "lstm_dataset.csv")
SCALER_FILE = os.path.join(ROOT_DIR, "models", "lstm_scaler.pkl")

HORIZONS    = [1, 6, 12, 24]
FEAT_COLS   = ["future_x_km", "future_y_km", "future_z_km",
               "vel_x_km_s",  "vel_y_km_s",  "vel_z_km_s"]
N_FEATURES  = len(FEAT_COLS)       # 6
SEQ_LEN     = 3                    # input steps (h1, h6, h12)


def main():
    print("=" * 62)
    print("  CollideX -- Step 7.2: LSTM Dataset Preparation")
    print("=" * 62)

    # ------------------------------------------------------------------
    print("\n[7.2.1] Loading trajectory data ...")
    df = pd.read_csv(INPUT_FILE)
    df = df[df["sgp4_error"] == 0].copy()
    print(f"  Valid rows : {len(df):,}")
    print(f"  Satellites : {df['norad_id'].nunique():,}")

    # ------------------------------------------------------------------
    print("\n[7.2.2] Building per-satellite sequences ...")
    records   = []
    skipped   = 0

    for norad_id, grp in df.groupby("norad_id"):
        grp = grp.sort_values("horizon_h")
        # Must have all 4 horizons
        if set(grp["horizon_h"].tolist()) != {1, 6, 12, 24}:
            skipped += 1
            continue

        # Extract ordered feature matrix [4 x 6]
        seq = grp.set_index("horizon_h")[FEAT_COLS].loc[HORIZONS].values  # (4, 6)

        # Build record: input = h1..h12 (3 rows), target = h24 (1 row)
        row = {"norad_id": norad_id}
        for step_idx, h in enumerate(HORIZONS[:SEQ_LEN]):     # h1, h6, h12
            for fi, feat in enumerate(FEAT_COLS):
                row[f"X_h{h}_{feat}"] = seq[step_idx, fi]

        for fi, feat in enumerate(FEAT_COLS):                  # h24 target
            row[f"y_{feat}"] = seq[3, fi]

        records.append(row)

    print(f"  Sequences built : {len(records):,}")
    print(f"  Sequences skipped (incomplete horizons): {skipped:,}")

    dataset = pd.DataFrame(records)

    # ------------------------------------------------------------------
    print("\n[7.2.3] Normalizing features (MinMaxScaler) ...")
    feat_input_cols  = [c for c in dataset.columns if c.startswith("X_")]
    feat_target_cols = [c for c in dataset.columns if c.startswith("y_")]

    all_feat_cols = feat_input_cols + feat_target_cols
    scaler = MinMaxScaler()
    dataset[all_feat_cols] = scaler.fit_transform(dataset[all_feat_cols])

    os.makedirs(os.path.dirname(SCALER_FILE), exist_ok=True)
    joblib.dump(scaler, SCALER_FILE)
    print(f"  Scaler saved -> {SCALER_FILE}")

    # ------------------------------------------------------------------
    print("\n[7.2.4] Saving lstm_dataset.csv ...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    dataset.to_csv(OUTPUT_FILE, index=False)
    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"  Saved  -> {OUTPUT_FILE}")
    print(f"  Rows   : {len(dataset):,}")
    print(f"  Cols   : {dataset.shape[1]}  "
          f"({len(feat_input_cols)} input + {len(feat_target_cols)} target + 1 id)")
    print(f"  Size   : {size_kb:.1f} KB")
    print(f"\n  Input columns  (first 3): {feat_input_cols[:3]}")
    print(f"  Target columns         : {feat_target_cols}")
    print()


if __name__ == "__main__":
    main()
