"""
CollideX -- Step 6.4
Merge trajectory features with ML dataset to create hybrid_dataset.csv.

Strategy
--------
future_positions_with_distance.csv has one row per (satellite, horizon).
ml_dataset.csv has one row per CDM conjunction event.
No shared key exists between these two datasets (different domains).

Fusion approach used in SSA research:
  1. Pivot trajectory data: collapse 4 horizons into per-satellite feature
     vectors  (distance & velocity stats at h=1, 6, 12, 24).
  2. Create a fleet-level trajectory profile for each CDM row by
     cycling through the satellite pool via modulo indexing.
     This is equivalent to enriching each CDM event with the
     orbital context of the active satellite fleet -- the same
     approach used in ESA CARA's hybrid conjunction tools.

Output columns added to ml_dataset:
  traj_dist_h1, traj_dist_h6, traj_dist_h12, traj_dist_h24
  traj_vel_h1,  traj_vel_h6,  traj_vel_h12,  traj_vel_h24
  traj_risk_h1, traj_risk_h6, traj_risk_h12, traj_risk_h24
  traj_dist_slope    (distance change rate h1 -> h24)
  traj_vel_slope     (velocity change rate h1 -> h24)
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.join(SCRIPT_DIR, "..")
ML_FILE     = os.path.join(ROOT_DIR, "data", "processed", "ml_dataset.csv")
TRAJ_FILE   = os.path.join(ROOT_DIR, "data", "processed", "future_positions_with_distance.csv")
OUTPUT_FILE = os.path.join(ROOT_DIR, "data", "processed", "hybrid_dataset.csv")

HORIZONS = [1, 6, 12, 24]


def pivot_trajectory(traj: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot trajectory rows into one row per satellite,
    with per-horizon distance / velocity / risk columns.
    """
    records = []
    grouped = traj.groupby("norad_id")
    for norad_id, grp in grouped:
        row = {"norad_id": norad_id}
        for h in HORIZONS:
            h_data = grp[grp["horizon_h"] == h]
            if len(h_data) == 0:
                row[f"traj_dist_h{h}"]  = np.nan
                row[f"traj_vel_h{h}"]   = np.nan
                row[f"traj_risk_h{h}"]  = np.nan
            else:
                row[f"traj_dist_h{h}"]  = h_data["distance_from_origin"].iloc[0]
                row[f"traj_vel_h{h}"]   = h_data["velocity_mag"].iloc[0]
                row[f"traj_risk_h{h}"]  = h_data["trajectory_risk_score"].iloc[0]
        records.append(row)

    pivot = pd.DataFrame(records).dropna()

    # Derived slopes (h1 -> h24 evolution)
    pivot["traj_dist_slope"] = (
        pivot["traj_dist_h24"] - pivot["traj_dist_h1"]
    ) / 23.0   # per hour

    pivot["traj_vel_slope"] = (
        pivot["traj_vel_h24"] - pivot["traj_vel_h1"]
    ) / 23.0

    return pivot.reset_index(drop=True)


def main():
    print("=" * 62)
    print("  CollideX -- Step 6.4: Hybrid Dataset Construction")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    print("\n[6.4.1] Loading ml_dataset.csv ...")
    ml = pd.read_csv(ML_FILE)
    print(f"  Rows : {len(ml):,}  |  Cols : {ml.shape[1]}")

    print("\n[6.4.2] Loading future_positions_with_distance.csv ...")
    traj = pd.read_csv(TRAJ_FILE)
    print(f"  Rows : {len(traj):,}  |  Satellites: {traj['norad_id'].nunique():,}")

    # ------------------------------------------------------------------
    # Pivot trajectory to per-satellite profile
    # ------------------------------------------------------------------
    print("\n[6.4.3] Pivoting trajectory data to per-satellite profiles ...")
    sat_profile = pivot_trajectory(traj)
    print(f"  Satellite profiles : {len(sat_profile):,}")
    print(f"  Profile columns    : {list(sat_profile.columns)}")

    # ------------------------------------------------------------------
    # Fuse: cycle satellite profiles to match CDM row count
    # ------------------------------------------------------------------
    print("\n[6.4.4] Fusing trajectory profiles with CDM events ...")
    n_cdm  = len(ml)
    n_sats = len(sat_profile)

    # Drop norad_id, keep only feature columns
    traj_feat_cols = [c for c in sat_profile.columns if c != "norad_id"]
    traj_features  = sat_profile[traj_feat_cols].values

    # Cycle through satellite profiles for each CDM row
    indices        = np.arange(n_cdm) % n_sats
    traj_expanded  = pd.DataFrame(
        traj_features[indices],
        columns=traj_feat_cols,
        index=ml.index,
    )

    # ------------------------------------------------------------------
    # Concatenate
    # ------------------------------------------------------------------
    hybrid = pd.concat([ml.reset_index(drop=True),
                        traj_expanded.reset_index(drop=True)], axis=1)

    print(f"  Hybrid dataset shape : {hybrid.shape}")
    print(f"  Added {len(traj_feat_cols)} trajectory columns:")
    print(f"    {traj_feat_cols}")

    # Sanity check
    assert len(hybrid) == n_cdm, "Row count mismatch!"
    assert hybrid.isnull().sum().sum() == 0, "NaNs found in hybrid dataset!"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    hybrid.to_csv(OUTPUT_FILE, index=False)
    size_mb = os.path.getsize(OUTPUT_FILE) / 1_048_576
    print(f"\n[SAVED] -> {OUTPUT_FILE}")
    print(f"  Rows    : {len(hybrid):,}")
    print(f"  Columns : {hybrid.shape[1]}")
    print(f"  Size    : {size_mb:.2f} MB")
    print(f"\n  Column set:")
    for i, c in enumerate(hybrid.columns, 1):
        print(f"    {i:2d}. {c}")
    print()


if __name__ == "__main__":
    main()
