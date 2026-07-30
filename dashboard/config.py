"""
CollideX Dashboard — Central Configuration
==========================================
All paths, theme tokens, and constants live here.
"""

import os

# ---------------------------------------------------------------------------
# Project Root Detection
# ---------------------------------------------------------------------------
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(DASHBOARD_DIR, ".."))

DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR   = os.path.join(DATA_DIR, "results")
TLE_DIR       = os.path.join(DATA_DIR, "tle")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
SCRIPTS_DIR   = os.path.join(PROJECT_ROOT, "scripts")

# ---------------------------------------------------------------------------
# Results Files
# ---------------------------------------------------------------------------
COLLISION_REPORT_CSV   = os.path.join(RESULTS_DIR, "inference_collision_report.csv")
FUTURE_POSITIONS_CSV   = os.path.join(RESULTS_DIR, "inference_future_positions.csv")
METRICS_CSV            = os.path.join(RESULTS_DIR, "metrics.csv")
EVAL_METRICS_TXT       = os.path.join(RESULTS_DIR, "evaluation_metrics.txt")
CLASSIFICATION_REPORT  = os.path.join(RESULTS_DIR, "classification_report.txt")
LSTM_METRICS_TXT       = os.path.join(RESULTS_DIR, "lstm_metrics.txt")
EVAL_REPORT_PDF        = os.path.join(RESULTS_DIR, "CollideX_Model_Evaluation_Report.pdf")

# Visual results
CONFUSION_MATRIX_PNG        = os.path.join(RESULTS_DIR, "confusion_matrix.png")
ROC_CURVE_PNG               = os.path.join(RESULTS_DIR, "roc_curve.png")
FEATURE_IMPORTANCE_PNG      = os.path.join(RESULTS_DIR, "feature_importance.png")
PRECISION_RECALL_PNG        = os.path.join(RESULTS_DIR, "precision_recall_curve.png")
COLLISION_HIST_PNG          = os.path.join(RESULTS_DIR, "collision_prob_histogram.png")
RISK_DIST_PNG               = os.path.join(RESULTS_DIR, "risk_class_distribution.png")
TRAJECTORY_ERROR_PNG        = os.path.join(RESULTS_DIR, "trajectory_error_distribution.png")

# TLE Files
DEFAULT_TLE_FILE = os.path.join(TLE_DIR, "full_catalog_3le.txt")
LEO_TLE_FILE     = os.path.join(TLE_DIR, "leo_3le.txt")

# ---------------------------------------------------------------------------
# Color Palette — Space/Aerospace Dark Theme
# ---------------------------------------------------------------------------
COLORS = {
    "bg_primary"      : "#050A14",   # Deep space black
    "bg_secondary"    : "#0A1628",   # Dark navy
    "bg_card"         : "#0D1F3C",   # Card background
    "bg_card_hover"   : "#122347",   # Card hover
    "accent_cyan"     : "#00D4FF",   # Primary accent
    "accent_cyan_dim" : "#00A3CC",   # Dimmer cyan
    "accent_blue"     : "#1E6FFF",   # Space blue
    "accent_orange"   : "#FF6B35",   # Alert/warning
    "accent_green"    : "#00FF9F",   # Success / Low risk
    "accent_red"      : "#FF3860",   # High risk / error
    "accent_yellow"   : "#FFD700",   # Medium risk
    "text_primary"    : "#E8F4FD",   # Main text
    "text_secondary"  : "#8BA3C7",   # Secondary text
    "text_muted"      : "#4A6A8A",   # Muted text
    "border"          : "#1A3A5C",   # Card borders
    "gradient_start"  : "#050A14",
    "gradient_end"    : "#0A2040",
}

RISK_COLORS = {
    "Low"    : "#00FF9F",
    "Medium" : "#FFD700",
    "High"   : "#FF3860",
}

RISK_BG_COLORS = {
    "Low"    : "rgba(0, 255, 159, 0.15)",
    "Medium" : "rgba(255, 215, 0, 0.15)",
    "High"   : "rgba(255, 56, 96, 0.15)",
}

# ---------------------------------------------------------------------------
# Plotly Base Template
# ---------------------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_dark"

PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor  = "rgba(0,0,0,0)",
    plot_bgcolor   = "rgba(13, 31, 60, 0.5)",
    font           = dict(family="Inter, sans-serif", color="#E8F4FD", size=12),
    title_font     = dict(family="Inter, sans-serif", color="#E8F4FD", size=16),
    margin         = dict(l=40, r=20, t=50, b=40),
    xaxis          = dict(gridcolor="#1A3A5C", linecolor="#1A3A5C", tickcolor="#4A6A8A"),
    yaxis          = dict(gridcolor="#1A3A5C", linecolor="#1A3A5C", tickcolor="#4A6A8A"),
    legend         = dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1A3A5C"),
)

# ---------------------------------------------------------------------------
# App Constants
# ---------------------------------------------------------------------------
APP_TITLE       = "CollideX"
APP_SUBTITLE    = "AI-Based Space Debris Collision Prediction System"
APP_VERSION     = "v1.0 Production"
APP_ICON        = "🛸"

# Model stats (static, from evaluation)
MODEL_ACCURACY      = 99.99
MODEL_ROC_AUC       = 100.0
TOTAL_SATELLITES    = 15_299
TOTAL_DEBRIS        = 15_299
INFERENCE_TIME_MS   = 251.17
TRAINING_SAMPLES    = 130_098
TESTING_SAMPLES     = 32_525
