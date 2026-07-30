"""
CollideX Dashboard — Data Loading Utilities
============================================
All file I/O, parsing, and data-prep for the dashboard.
Never re-trains or re-runs ML — only reads existing files.
"""

import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    COLLISION_REPORT_CSV, FUTURE_POSITIONS_CSV, METRICS_CSV,
    EVAL_METRICS_TXT, CLASSIFICATION_REPORT, LSTM_METRICS_TXT,
    RESULTS_DIR, SCRIPTS_DIR, DEFAULT_TLE_FILE, LEO_TLE_FILE,
)


# ===========================================================================
# CACHED DATA LOADERS
# ===========================================================================

@st.cache_data(show_spinner=False)
def load_collision_report() -> pd.DataFrame:
    """Load the main inference collision report CSV."""
    if not os.path.exists(COLLISION_REPORT_CSV):
        st.warning("inference_collision_report.csv not found. Run a prediction first.")
        return pd.DataFrame()
    df = pd.read_csv(COLLISION_REPORT_CSV)
    # Ensure risk_label column exists
    if "risk_label" not in df.columns and "risk_class" in df.columns:
        df["risk_label"] = df["risk_class"].map({0: "Low", 1: "Medium", 2: "High"})
    return df


@st.cache_data(show_spinner=False)
def load_future_positions() -> pd.DataFrame:
    """Load the future positions CSV (SGP4 + LSTM)."""
    if not os.path.exists(FUTURE_POSITIONS_CSV):
        return pd.DataFrame()
    return pd.read_csv(FUTURE_POSITIONS_CSV)


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    """Load metrics.csv as a DataFrame."""
    if not os.path.exists(METRICS_CSV):
        return pd.DataFrame()
    return pd.read_csv(METRICS_CSV)


@st.cache_data(show_spinner=False)
def load_eval_metrics_text() -> str:
    """Load evaluation_metrics.txt as raw text."""
    if not os.path.exists(EVAL_METRICS_TXT):
        return "Evaluation metrics file not found."
    with open(EVAL_METRICS_TXT, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def load_classification_report_text() -> str:
    """Load classification_report.txt as raw text."""
    if not os.path.exists(CLASSIFICATION_REPORT):
        return "Classification report not found."
    with open(CLASSIFICATION_REPORT, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def load_lstm_metrics_text() -> str:
    """Load lstm_metrics.txt as raw text."""
    if not os.path.exists(LSTM_METRICS_TXT):
        return "LSTM metrics file not found."
    with open(LSTM_METRICS_TXT, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def parse_eval_metrics_to_dict() -> dict:
    """
    Parse evaluation_metrics.txt into a key-value dictionary
    for programmatic display.
    """
    text = load_eval_metrics_text()
    result = {}
    patterns = [
        (r"Accuracy\s*:\s*([\d.]+)", "Accuracy"),
        (r"Balanced Accuracy\s*:\s*([\d.]+)", "Balanced Accuracy"),
        (r"Matthews Corr\. Coef\.\s*:\s*([\d.]+)", "MCC"),
        (r"Cohen Kappa\s*:\s*([\d.]+)", "Cohen Kappa"),
        (r"Log Loss\s*:\s*([\d.]+)", "Log Loss"),
        (r"ROC-AUC \(macro OVR\)\s*:\s*([\d.]+)", "ROC-AUC"),
        (r"Total Samples\s*:\s*([\d,]+)", "Total Samples"),
        (r"Feature Count\s*:\s*(\d+)", "Feature Count"),
        (r"Train Samples\s*:\s*([\d,]+)", "Train Samples"),
        (r"Test\s+Samples\s*:\s*([\d,]+)", "Test Samples"),
        (r"Number of Trees\s*:\s*(\d+)", "N Trees"),
        (r"Max Depth\s*:\s*(\d+)", "Max Depth"),
        (r"Prediction\s+\(total\)\s*:\s*([\d.]+) ms", "Prediction Time (ms)"),
        (r"Inference\s+\(total\)\s*:\s*([\d.]+) ms", "Inference Time (ms)"),
    ]
    for pattern, key in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "")
            try:
                result[key] = float(val)
            except ValueError:
                result[key] = val
    return result


# ===========================================================================
# TLE UTILITIES
# ===========================================================================

def count_tle_objects(filepath: str) -> int:
    """Count number of valid TLE objects in a file."""
    if not os.path.exists(filepath):
        return 0
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f.readlines()]
        count = 0
        for i in range(len(lines) - 2):
            if lines[i+1].startswith("1 ") and lines[i+2].startswith("2 "):
                count += 1
        return count
    except Exception:
        return 0


def save_uploaded_tle(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temp path and return it."""
    suffix = ".txt"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix,
                                      prefix="collidex_tle_")
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    tmp.close()
    return tmp.name


# ===========================================================================
# PREDICTION RUNNER
# ===========================================================================

def run_prediction(
    tle_path: str,
    top_n: Optional[int] = None,
    progress_callback=None,
) -> Tuple[bool, str, pd.DataFrame, pd.DataFrame]:
    """
    Invoke predict.py as a subprocess, stream output,
    then reload result CSVs.

    Returns:
        success   : bool
        log_text  : str  (stdout + stderr combined)
        report_df : pd.DataFrame  (collision report)
        pos_df    : pd.DataFrame  (future positions)
    """
    predict_script = os.path.join(SCRIPTS_DIR, "predict.py")
    if not os.path.exists(predict_script):
        return False, f"predict.py not found at {predict_script}", pd.DataFrame(), pd.DataFrame()

    cmd = [sys.executable, predict_script, "--tle", tle_path]
    if top_n:
        cmd += ["--top", str(top_n)]

    log_lines = []
    stages = [
        "Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5",
    ]
    current_stage = 0

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="ignore",
        )

        for line in proc.stdout:
            log_lines.append(line.rstrip())
            # Update progress based on stage detection
            for i, stage in enumerate(stages):
                if stage in line:
                    current_stage = i + 1
                    if progress_callback:
                        progress_callback(current_stage / len(stages), line.strip())
                    break

        proc.wait()
        success = proc.returncode == 0
    except Exception as e:
        return False, str(e), pd.DataFrame(), pd.DataFrame()

    log_text = "\n".join(log_lines)

    # Reload result CSVs (clear cache so fresh data appears)
    load_collision_report.clear()
    load_future_positions.clear()

    report_df = load_collision_report()
    pos_df    = load_future_positions()

    return success, log_text, report_df, pos_df


# ===========================================================================
# DATA HELPERS
# ===========================================================================

def get_risk_summary(df: pd.DataFrame) -> dict:
    """Return counts and percentages per risk class."""
    if df.empty or "risk_label" not in df.columns:
        return {"Low": 0, "Medium": 0, "High": 0}
    counts = df["risk_label"].value_counts().to_dict()
    return {k: counts.get(k, 0) for k in ["Low", "Medium", "High"]}


def get_top_risk_objects(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top N highest-risk objects."""
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in [
        "norad_id", "satellite_name", "collision_probability",
        "risk_label", "altitude_km", "velocity_mag_km_s"
    ] if c in df.columns]
    return df[cols].nlargest(n, "collision_probability").reset_index(drop=True)


def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric columns for table display."""
    out = df.copy()
    for col in out.select_dtypes(include=[float]).columns:
        if "probability" in col.lower():
            out[col] = out[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        elif "km" in col.lower():
            out[col] = out[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    return out


def metric_value(metrics_df: pd.DataFrame, name: str,
                 default: float = 0.0) -> float:
    """Extract a named metric value from the metrics DataFrame."""
    if metrics_df.empty:
        return default
    row = metrics_df[metrics_df["Metric"] == name]
    if row.empty:
        return default
    return float(row.iloc[0]["Value"])


@st.cache_data(show_spinner=False)
def load_image_bytes(path: str) -> Optional[bytes]:
    """Return raw bytes of an image file, or None if missing."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()
