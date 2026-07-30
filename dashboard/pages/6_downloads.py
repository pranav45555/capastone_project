"""
PAGE 6 — Download Reports
One-click download for all generated output files.
"""

import os
import streamlit as st

from components.ui import (
    section_header, page_title, divider, info_box
)
from config import (
    COLLISION_REPORT_CSV, FUTURE_POSITIONS_CSV, METRICS_CSV,
    EVAL_METRICS_TXT, CLASSIFICATION_REPORT, LSTM_METRICS_TXT,
    EVAL_REPORT_PDF, COLORS
)


def _file_size(path: str) -> str:
    if not os.path.exists(path):
        return "N/A"
    size = os.path.getsize(path)
    if size < 1024:
        return f"{size} B"
    elif size < 1024 ** 2:
        return f"{size/1024:.1f} KB"
    else:
        return f"{size/1024**2:.2f} MB"


def _download_card(col, title: str, subtitle: str, filepath: str,
                   filename: str, mime: str, icon: str = "⬇"):
    file_size = _file_size(filepath)
    exists = os.path.exists(filepath)

    with col:
        st.markdown(f"""
        <div style="background:rgba(13,31,60,0.8);border:1px solid {'#1A3A5C' if exists else '#3A1A1A'};
                    border-radius:14px;padding:20px;margin-bottom:8px;
                    {'opacity:0.5;' if not exists else ''}">
            <div style="font-size:22px;margin-bottom:8px;">{icon}</div>
            <div style="font-size:14px;font-weight:700;color:#E8F4FD;margin-bottom:4px;">{title}</div>
            <div style="font-size:12px;color:#4A6A8A;margin-bottom:4px;">{subtitle}</div>
            <div style="font-size:11px;color:#1E6FFF;font-family:'JetBrains Mono',monospace;">
                {filename} · {file_size}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if exists:
            with open(filepath, "rb") as f:
                data = f.read()
            st.download_button(
                f"Download {filename}",
                data=data,
                file_name=filename,
                mime=mime,
                width='stretch',
                key=f"dl_{filename}"
            )
        else:
            st.button(
                "File Not Found",
                disabled=True,
                width='stretch',
                key=f"dl_missing_{filename}"
            )


def render():
    st.markdown(page_title(
        "Download Reports",
        "EXPORT · ALL GENERATED FILES"
    ), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Evaluation Reports
    # -----------------------------------------------------------------------
    st.markdown(section_header("Evaluation Reports"), unsafe_allow_html=True)

    r1c1, r1c2, r1c3 = st.columns(3, gap="medium")

    _download_card(
        r1c1,
        "Evaluation Report PDF",
        "Complete model evaluation with all metrics, charts, and analysis",
        EVAL_REPORT_PDF,
        "CollideX_Model_Evaluation_Report.pdf",
        "application/pdf",
        icon="📄"
    )
    _download_card(
        r1c2,
        "Metrics CSV",
        "All performance metrics in tabular CSV format",
        METRICS_CSV,
        "metrics.csv",
        "text/csv",
        icon="📊"
    )
    _download_card(
        r1c3,
        "Evaluation Metrics TXT",
        "Full evaluation summary report in plain text",
        EVAL_METRICS_TXT,
        "evaluation_metrics.txt",
        "text/plain",
        icon="📝"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Prediction Outputs
    # -----------------------------------------------------------------------
    st.markdown(section_header("Prediction Outputs"), unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2, gap="medium")

    _download_card(
        r2c1,
        "Collision Report CSV",
        "Per-satellite collision probability, risk class, and orbital data",
        COLLISION_REPORT_CSV,
        "inference_collision_report.csv",
        "text/csv",
        icon="🛰"
    )
    _download_card(
        r2c2,
        "Future Positions CSV",
        "SGP4 + LSTM predicted XYZ coordinates at h=1, 6, 12, 24 hours",
        FUTURE_POSITIONS_CSV,
        "inference_future_positions.csv",
        "text/csv",
        icon="📡"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Classification Reports
    # -----------------------------------------------------------------------
    st.markdown(section_header("Classification Reports"), unsafe_allow_html=True)

    r3c1, r3c2 = st.columns(2, gap="medium")

    _download_card(
        r3c1,
        "Classification Report TXT",
        "Detailed per-class precision, recall, F1 from sklearn",
        CLASSIFICATION_REPORT,
        "classification_report.txt",
        "text/plain",
        icon="🎯"
    )
    _download_card(
        r3c2,
        "LSTM Metrics TXT",
        "LSTM trajectory model regression metrics (MSE, RMSE, MAE, R²)",
        LSTM_METRICS_TXT,
        "lstm_metrics.txt",
        "text/plain",
        icon="🧠"
    )

    st.markdown(divider(), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # File Status Table
    # -----------------------------------------------------------------------
    st.markdown(section_header("File Status Overview"), unsafe_allow_html=True)

    all_files = [
        ("CollideX_Model_Evaluation_Report.pdf", EVAL_REPORT_PDF),
        ("metrics.csv",                          METRICS_CSV),
        ("evaluation_metrics.txt",               EVAL_METRICS_TXT),
        ("inference_collision_report.csv",        COLLISION_REPORT_CSV),
        ("inference_future_positions.csv",        FUTURE_POSITIONS_CSV),
        ("classification_report.txt",             CLASSIFICATION_REPORT),
        ("lstm_metrics.txt",                      LSTM_METRICS_TXT),
    ]

    for fname, fpath in all_files:
        exists = os.path.exists(fpath)
        size   = _file_size(fpath)
        color  = "#00FF9F" if exists else "#FF3860"
        status = "READY" if exists else "MISSING"
        dot    = "●"

        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:10px 0;border-bottom:1px solid #1A3A5C;">
            <span style="font-size:13px;color:#E8F4FD;font-family:'JetBrains Mono',monospace;">
                {fname}
            </span>
            <span style="font-size:11px;color:#4A6A8A;margin:0 24px;">{size}</span>
            <span style="font-size:12px;font-weight:600;color:{color};">{dot} {status}</span>
        </div>
        """, unsafe_allow_html=True)
