"""
PAGE 3 — Evaluation
Loads results files and displays all model metrics.
"""

import os
import streamlit as st
import pandas as pd

from components.ui import (
    section_header, metric_card, page_title,
    divider, info_box
)
from components.charts import metrics_bar_chart
from utils.data_loader import (
    load_metrics, load_eval_metrics_text, parse_eval_metrics_to_dict,
    load_classification_report_text, load_lstm_metrics_text,
    metric_value
)
from config import COLORS, METRICS_CSV, EVAL_METRICS_TXT, CLASSIFICATION_REPORT


def _card_block(header: str, rows: list) -> str:
    """
    Build a complete info card as a single HTML string.
    rows = list of (label, value) tuples.
    """
    rows_html = "".join([
        f'<div style="display:flex;justify-content:space-between;'
        f'padding:8px 0;border-bottom:1px solid #1A3A5C;">'
        f'<span style="font-size:13px;color:#8BA3C7;">{lbl}</span>'
        f'<span style="font-size:13px;font-weight:600;color:#E8F4FD;'
        f'font-family:\'JetBrains Mono\',monospace;">{val}</span>'
        f'</div>'
        for lbl, val in rows
    ])
    return f"""
    <div style="background:rgba(13,31,60,0.7);border:1px solid #1A3A5C;
                border-radius:12px;padding:20px;">
        <div style="font-size:11px;color:#4A6A8A;letter-spacing:1.5px;
                    text-transform:uppercase;margin-bottom:16px;font-weight:600;">
            {header}
        </div>
        {rows_html}
    </div>
    """


def render():
    st.markdown(page_title(
        "Model Evaluation",
        "PERFORMANCE METRICS · CLASSIFICATION REPORT · INFERENCE ANALYTICS"
    ), unsafe_allow_html=True)

    metrics_df = load_metrics()
    parsed     = parse_eval_metrics_to_dict()

    # -----------------------------------------------------------------------
    # Top KPI Row
    # -----------------------------------------------------------------------
    st.markdown(section_header("Performance Summary", "Hybrid Random Forest Classifier"),
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_data = [
        (c1, f"{metric_value(metrics_df,'Accuracy')*100:.4f}%",
             "Accuracy", COLORS["accent_cyan"]),
        (c2, f"{metric_value(metrics_df,'Balanced Accuracy')*100:.4f}%",
             "Balanced Acc.", COLORS["accent_green"]),
        (c3, f"{metric_value(metrics_df,'Precision (macro)')*100:.4f}%",
             "Precision", COLORS["accent_blue"]),
        (c4, f"{metric_value(metrics_df,'Recall (macro)')*100:.4f}%",
             "Recall", COLORS["accent_cyan"]),
        (c5, f"{metric_value(metrics_df,'F1-Score (macro)')*100:.4f}%",
             "F1 Score", COLORS["accent_green"]),
        (c6, f"{metric_value(metrics_df,'ROC-AUC (macro OVR)')*100:.2f}%",
             "ROC-AUC", "#A855F7"),
    ]
    for col, val, lbl, color in kpi_data:
        with col:
            st.markdown(metric_card(val, lbl, color=color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Bar Chart + Per-Class Table
    # -----------------------------------------------------------------------
    chart_col, table_col = st.columns([1.5, 1], gap="large")

    with chart_col:
        if not metrics_df.empty:
            st.plotly_chart(metrics_bar_chart(metrics_df), width='stretch')

    with table_col:
        st.markdown(section_header("Per-Class Metrics"), unsafe_allow_html=True)

        # Build the ENTIRE per-class block as one HTML string
        risk_colors = {"Low": "#00FF9F", "Medium": "#FFD700", "High": "#FF3860"}
        per_class_rows = [
            ("Low",    "Precision"),
            ("Low",    "Recall"),
            ("Low",    "F1-Score"),
            ("Medium", "Precision"),
            ("Medium", "Recall"),
            ("Medium", "F1-Score"),
            ("High",   "Precision"),
            ("High",   "Recall"),
            ("High",   "F1-Score"),
        ]
        rows_html = ""
        for cls, met in per_class_rows:
            v     = metric_value(metrics_df, f"{met} ({cls})", 0.0)
            color = risk_colors[cls]
            rows_html += (
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;padding:7px 0;border-bottom:1px solid #1A3A5C;">'
                f'<span style="font-size:12px;color:{color};font-weight:600;'
                f'min-width:56px;">{cls}</span>'
                f'<span style="font-size:12px;color:#8BA3C7;flex:1;">{met}</span>'
                f'<span style="font-size:13px;font-weight:700;color:#E8F4FD;'
                f'font-family:\'JetBrains Mono\',monospace;">{v*100:.4f}%</span>'
                f'</div>'
            )
        st.markdown(f'<div style="padding:4px 0;">{rows_html}</div>',
                    unsafe_allow_html=True)

    st.markdown(divider(), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Dataset + Model Info — built as complete single HTML blocks
    # -----------------------------------------------------------------------
    st.markdown(section_header("Dataset & Model Configuration"), unsafe_allow_html=True)
    info_l, info_r = st.columns(2, gap="large")

    with info_l:
        dataset_rows = [
            ("Dataset",         "hybrid_dataset.csv"),
            ("Total Samples",   f"{int(parsed.get('Total Samples', 162623)):,}"),
            ("Features",        f"{int(parsed.get('Feature Count', 33))}"),
            ("Train Samples",   f"{int(parsed.get('Train Samples', 130098)):,}  (80%)"),
            ("Test Samples",    f"{int(parsed.get('Test Samples', 32525)):,}   (20%)"),
            ("Target Classes",  "0=Low · 1=Medium · 2=High"),
        ]
        st.markdown(_card_block("Dataset Information", dataset_rows),
                    unsafe_allow_html=True)

    with info_r:
        model_rows = [
            ("Model Name",     "Hybrid Random Forest"),
            ("Model File",     "collision_model_hybrid.pkl"),
            ("N Trees",        f"{int(parsed.get('N Trees', 200))}"),
            ("Max Depth",      f"{int(parsed.get('Max Depth', 12))}"),
            ("Class Weight",   "balanced"),
            ("sklearn",        "1.7.1"),
        ]
        st.markdown(_card_block("Model Configuration", model_rows),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Inference Timing
    # -----------------------------------------------------------------------
    st.markdown(section_header("Inference Timing"), unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    timing = [
        (t1, f"{metric_value(metrics_df,'Prediction Time Total (ms)'):.2f} ms",
             "Prediction Time", "Total batch"),
        (t2, f"{metric_value(metrics_df,'Prediction Time / Sample (ms)'):.5f} ms",
             "Per Sample", "Prediction"),
        (t3, f"{metric_value(metrics_df,'Inference Time Total (ms)'):.2f} ms",
             "Inference Time", "Total batch"),
        (t4, f"{metric_value(metrics_df,'Inference Time / Sample (ms)'):.5f} ms",
             "Per Sample", "Inference"),
    ]
    for col, val, lbl, sub in timing:
        with col:
            st.markdown(metric_card(val, lbl, sublabel=sub,
                                    color=COLORS["accent_cyan"]), unsafe_allow_html=True)

    st.markdown(divider(), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # LSTM Metrics
    # -----------------------------------------------------------------------
    st.markdown(section_header("LSTM Trajectory Model", "Sequential Encoder → Regressor"),
                unsafe_allow_html=True)

    lstm_text = load_lstm_metrics_text()
    with st.expander("LSTM Evaluation Report", expanded=True):
        st.code(lstm_text, language="")

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Full Metrics Table
    # -----------------------------------------------------------------------
    st.markdown(section_header("Complete Metrics Table"), unsafe_allow_html=True)
    if not metrics_df.empty:
        fmt = metrics_df.copy()
        fmt["Value"] = fmt["Value"].apply(
            lambda x: f"{float(x)*100:.6f}%" if float(x) <= 1.0 else f"{float(x):.4f}"
        )
        st.dataframe(fmt, width='stretch', hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Raw Reports
    # -----------------------------------------------------------------------
    with st.expander("Full Evaluation Metrics Report (raw text)"):
        st.code(load_eval_metrics_text(), language="")

    with st.expander("Full Classification Report (raw text)"):
        st.code(load_classification_report_text(), language="")

    # -----------------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------------
    st.markdown(divider(), unsafe_allow_html=True)
    st.markdown(section_header("Export"), unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)
    with dl1:
        if not metrics_df.empty:
            st.download_button(
                "Download Metrics CSV",
                data=metrics_df.to_csv(index=False).encode(),
                file_name="collidex_metrics.csv",
                mime="text/csv",
            )
    with dl2:
        eval_text = load_eval_metrics_text()
        st.download_button(
            "Download Evaluation Report TXT",
            data=eval_text.encode(),
            file_name="evaluation_metrics.txt",
            mime="text/plain",
        )
