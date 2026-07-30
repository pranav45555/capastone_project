"""
CollideX — Step 3.3
Parse 3LE file and extract satellite TLE triplets.
"""

import os

TLE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "tle", "full_catalog_3le.txt")

def parse_tle_file(filepath: str) -> list[tuple[str, str, str]]:
    """
    Parse a 3LE (three-line element) file and return a list of
    (name, line1, line2) tuples.
    """
    satellites = []
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        lines = [l.rstrip() for l in f.readlines()]

    i = 0
    while i < len(lines) - 2:
        # Lines that begin with '0 ' or are pure name lines (not TLE lines)
        name_line  = lines[i]
        line1_cand = lines[i + 1]
        line2_cand = lines[i + 2]

        # TLE line 1 starts with '1 ', line 2 starts with '2 '
        if line1_cand.startswith("1 ") and line2_cand.startswith("2 "):
            name = name_line.lstrip("0 ").strip()
            satellites.append((name, line1_cand.strip(), line2_cand.strip()))
            i += 3
        else:
            i += 1

    return satellites


if __name__ == "__main__":
    sats = parse_tle_file(TLE_FILE)
    print(f"[OK] Total satellites loaded: {len(sats)}")
    print("Sample (first 3):")
    for s in sats[:3]:
        print(f"  Name : {s[0]}")
        print(f"  Line1: {s[1]}")
        print(f"  Line2: {s[2]}")
        print()
