"""
CollideX -- Step 6.1 to 6.3
Compute relative distance and velocity magnitude features
from SGP4 future position trajectories.

Input : CollideX/data/processed/future_positions.csv
Output: CollideX/data/processed/future_positions_with_distance.csv
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, "..")
INPUT_FILE = os.path.join(ROOT_DIR, "data", "processed", "future_positions.csv")
OUTPUT_FILE= os.path.join(ROOT_DIR, "data", "processed", "future_positions_with_distance.csv")


def main():
    print("=" * 62)
    print("  CollideX -- Step 6.1-6.3: Trajectory Feature Engineering")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Step 6.1 -- Load and compute distance from Earth's center
    # ------------------------------------------------------------------
    print("\n[Step 6.1] Loading future positions ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Rows   : {len(df):,}")
    print(f"  Cols   : {list(df.columns)}")

    # Keep only successful SGP4 propagations
    before = len(df)
    df = df[df["sgp4_error"] == 0].copy()
    print(f"  Valid propagations: {len(df):,}  (dropped {before - len(df):,} errors)")

    # Step 6.1 -- Magnitude of position vector (km from Earth centre)
    df["distance_from_origin"] = np.sqrt(
        df["future_x_km"] ** 2 +
        df["future_y_km"] ** 2 +
        df["future_z_km"] ** 2
    )
    print(f"\n[Step 6.1] distance_from_origin added.")
    print(f"  Mean : {df['distance_from_origin'].mean():.1f} km")
    print(f"  Min  : {df['distance_from_origin'].min():.1f} km")
    print(f"  Max  : {df['distance_from_origin'].max():.1f} km")

    # ------------------------------------------------------------------
    # Step 6.2 -- Velocity magnitude (km/s)
    # ------------------------------------------------------------------
    df["velocity_mag"] = np.sqrt(
        df["vel_x_km_s"] ** 2 +
        df["vel_y_km_s"] ** 2 +
        df["vel_z_km_s"] ** 2
    )
    print(f"\n[Step 6.2] velocity_mag added.")
    print(f"  Mean : {df['velocity_mag'].mean():.3f} km/s")
    print(f"  Min  : {df['velocity_mag'].min():.3f} km/s")
    print(f"  Max  : {df['velocity_mag'].max():.3f} km/s")

    # ------------------------------------------------------------------
    # Step 6.3 -- Trajectory risk score (velocity / distance)
    #             Higher = faster object at lower altitude = higher risk
    # ------------------------------------------------------------------
    df["trajectory_risk_score"] = np.where(
        df["distance_from_origin"] > 0,
        df["velocity_mag"] / df["distance_from_origin"],
        np.nan
    )
    df = df.dropna(subset=["trajectory_risk_score"])
    print(f"\n[Step 6.3] trajectory_risk_score added.")
    print(f"  Mean : {df['trajectory_risk_score'].mean():.6f}")
    print(f"  Max  : {df['trajectory_risk_score'].max():.6f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    df.to_csv(OUTPUT_FILE, index=False)
    size_mb = os.path.getsize(OUTPUT_FILE) / 1_048_576
    print(f"\n[SAVED] -> {OUTPUT_FILE}")
    print(f"  Rows : {len(df):,}")
    print(f"  Size : {size_mb:.2f} MB")
    print(f"  New columns: distance_from_origin, velocity_mag, trajectory_risk_score")

    # Per-horizon summary
    print("\n  Per-horizon summary:")
    summary = df.groupby("horizon_h")[
        ["distance_from_origin", "velocity_mag", "trajectory_risk_score"]
    ].mean()
    print(summary.to_string())
    print()


if __name__ == "__main__":
    main()
