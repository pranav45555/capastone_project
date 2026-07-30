"""
CollideX - Steps 3.4 to 3.7
Apply SGP4 propagation over multiple prediction horizons and export
future_positions.csv (one of the three final model outputs).

Prediction horizons:
  +1 h, +6 h, +12 h, +24 h  (research-level upgrade)

Output columns:
  satellite_name, norad_id, base_epoch, horizon_h,
  future_x_km, future_y_km, future_z_km,
  vel_x_km_s, vel_y_km_s, vel_z_km_s, sgp4_error
"""

import os
import sys
import datetime
import pandas as pd
from sgp4.api import Satrec, jday

# -- paths --------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.join(SCRIPT_DIR, "..")
TLE_FILE     = os.path.join(ROOT_DIR, "data", "tle", "full_catalog_3le.txt")
OUTPUT_DIR   = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "future_positions.csv")

# -- prediction horizons (hours ahead) ----------------------------------------
HORIZONS_H = [1, 6, 12, 24]

# -- reference epoch: 2026-04-17 00:00:00 UTC ---------------------------------
BASE_EPOCH = datetime.datetime(2026, 4, 17, 0, 0, 0)


# ----------------------------------------------------------------------------
def parse_tle_file(filepath):
    """Return list of (name, line1, line2) tuples from a 3LE file."""
    satellites = []
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip() for ln in f.readlines()]

    i = 0
    while i < len(lines) - 2:
        name_line  = lines[i]
        line1_cand = lines[i + 1]
        line2_cand = lines[i + 2]
        if line1_cand.startswith("1 ") and line2_cand.startswith("2 "):
            name = name_line.lstrip("0 ").strip()
            satellites.append((name, line1_cand.strip(), line2_cand.strip()))
            i += 3
        else:
            i += 1
    return satellites


def propagate(satellites, horizons_h):
    """
    Run SGP4 for every satellite x every horizon.
    Returns a flat DataFrame.
    """
    records = []
    total   = len(satellites)

    for idx, (name, line1, line2) in enumerate(satellites):
        if idx % 1000 == 0:
            pct = idx / total * 100
            sys.stdout.write("\r  Propagating ... %6d/%d (%.1f%%)" % (idx, total, pct))
            sys.stdout.flush()

        try:
            sat      = Satrec.twoline2rv(line1, line2)
            norad_id = int(line1[2:7].strip())
        except Exception:
            continue

        for h in horizons_h:
            epoch_t = BASE_EPOCH + datetime.timedelta(hours=h)
            jd, fr  = jday(epoch_t.year, epoch_t.month, epoch_t.day,
                           epoch_t.hour, epoch_t.minute, epoch_t.second)
            try:
                error, pos, vel = sat.sgp4(jd, fr)
            except Exception:
                error = 99
                pos   = (None, None, None)
                vel   = (None, None, None)

            ok = (error == 0)
            records.append({
                "satellite_name" : name,
                "norad_id"       : norad_id,
                "base_epoch"     : BASE_EPOCH.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "horizon_h"      : h,
                "future_x_km"    : round(pos[0], 4) if ok else None,
                "future_y_km"    : round(pos[1], 4) if ok else None,
                "future_z_km"    : round(pos[2], 4) if ok else None,
                "vel_x_km_s"     : round(vel[0], 6) if ok else None,
                "vel_y_km_s"     : round(vel[1], 6) if ok else None,
                "vel_z_km_s"     : round(vel[2], 6) if ok else None,
                "sgp4_error"     : error,
            })

    sys.stdout.write("\n")
    return pd.DataFrame(records)


# -- main ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  CollideX -- SGP4 Trajectory Propagation Engine")
    print("=" * 60)

    # Step 3.3 -- Parse TLE
    print("\n[Step 3.3] Parsing TLE file ...")
    satellites = parse_tle_file(TLE_FILE)
    print("  -> Total satellites loaded: %d" % len(satellites))

    # Step 3.4 / 3.6 -- Propagate
    print("\n[Step 3.4/3.6] Running SGP4 across horizons: %s hours ..." % HORIZONS_H)
    df = propagate(satellites, HORIZONS_H)

    # Step 3.5 -- Export to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print("\n[Step 3.5] Saved -> %s" % OUTPUT_FILE)
    print("  Rows  : %d" % len(df))
    print("  Cols  : %s" % list(df.columns))

    # Summary statistics
    ok_df = df[df["sgp4_error"] == 0]
    print("\n[Step 3.7] Summary:")
    print("  Successful propagations : %d" % len(ok_df))
    print("  Failed propagations     : %d" % (len(df) - len(ok_df)))
    print("\nSample future positions (first 5 rows, 24h horizon):")
    sample = ok_df[ok_df["horizon_h"] == 24].head(5)[
        ["satellite_name", "norad_id", "horizon_h",
         "future_x_km", "future_y_km", "future_z_km", "sgp4_error"]
    ]
    print(sample.to_string(index=False))
    print("\n[DONE] STEP 3 COMPLETE -- future_positions.csv ready for ML pipeline.\n")
