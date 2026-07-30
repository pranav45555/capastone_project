"""
PAGE 2 — Prediction
Upload TLE → Run Prediction → Display Results
"""

import os
import time
import tempfile

import pandas as pd
import streamlit as st

from components.ui import (
    section_header, kpi_card, info_box, divider, page_title, risk_badge
)
from components.charts import (
    risk_donut, top_risk_bar, altitude_scatter,
    gauge_chart, position_3d_scatter
)
from utils.data_loader import (
    load_collision_report, load_future_positions,
    get_risk_summary, get_top_risk_objects, save_uploaded_tle,
    run_prediction, count_tle_objects, format_dataframe_for_display
)
from config import (
    COLORS, DEFAULT_TLE_FILE, LEO_TLE_FILE,
    RISK_COLORS
)


def render():
    st.markdown(page_title(
        "Prediction Engine",
        "RUN COLLISION PROBABILITY ANALYSIS"
    ), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Input Panel
    # -----------------------------------------------------------------------
    st.markdown(section_header("Input Configuration", "Select TLE source and parameters"),
                unsafe_allow_html=True)

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.markdown("""
        <div style="background:rgba(13,31,60,0.7);border:1px solid #1A3A5C;
                    border-radius:14px;padding:24px;">
            <div style="font-size:13px;font-weight:600;color:#00D4FF;
                        text-transform:uppercase;letter-spacing:1.5px;margin-bottom:20px;">
                TLE Data Source
            </div>
        """, unsafe_allow_html=True)

        tle_source = st.radio(
            "Select TLE Source",
            ["Full Catalog (Space-Track)", "LEO Catalog", "Upload Custom TLE"],
            index=0,
            help="Choose which TLE file to use for prediction",
            label_visibility="collapsed"
        )

        uploaded_file = None
        tle_path = DEFAULT_TLE_FILE

        if tle_source == "Full Catalog (Space-Track)":
            tle_path = DEFAULT_TLE_FILE
            n = count_tle_objects(tle_path)
            st.markdown(info_box(f"Full Space-Track catalog · {n:,} objects"), unsafe_allow_html=True)

        elif tle_source == "LEO Catalog":
            tle_path = LEO_TLE_FILE
            n = count_tle_objects(tle_path)
            st.markdown(info_box(f"LEO-filtered catalog · {n:,} objects"), unsafe_allow_html=True)

        else:
            uploaded_file = st.file_uploader(
                "Upload TLE File (.txt)",
                type=["txt"],
                help="Upload a 3LE TLE file from Space-Track or similar source"
            )
            if uploaded_file:
                tle_path = save_uploaded_tle(uploaded_file)
                n = count_tle_objects(tle_path)
                st.markdown(info_box(f"Uploaded · {n:,} objects detected"), unsafe_allow_html=True)
            else:
                st.markdown(info_box("Upload a TLE file to continue"), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div style="background:rgba(13,31,60,0.7);border:1px solid #1A3A5C;
                    border-radius:14px;padding:20px;">
            <div style="font-size:13px;font-weight:600;color:#00D4FF;
                        text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px;">
                Analysis Parameters
            </div>
        </div>
        """, unsafe_allow_html=True)

        limit_mode = st.checkbox(
            "Limit analysis to N satellites (faster)",
            value=True,
            help="Recommended for quick demo runs"
        )
        top_n = None
        if limit_mode:
            top_n = st.slider(
                "Number of satellites to analyze",
                min_value=10, max_value=1000, value=100, step=10
            )
            st.markdown(info_box(f"Will analyze top {top_n} satellites from the TLE file"),
                        unsafe_allow_html=True)
        else:
            st.markdown(info_box(
                "Full catalog analysis may take several minutes. "
                "Ensure sufficient RAM and GPU resources.",
                kind="warning"
            ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Run Button
    # -----------------------------------------------------------------------
    can_run = (tle_source != "Upload Custom TLE") or (uploaded_file is not None)

    col_run, col_load, _ = st.columns([1.2, 1, 3])
    with col_run:
        run_clicked = st.button(
            "Run Prediction",
            disabled=not can_run,
            type="primary",
            key="run_prediction_btn"
        )
    with col_load:
        load_clicked = st.button(
            "Load Last Results",
            key="load_results_btn"
        )

    # -----------------------------------------------------------------------
    # Prediction Execution
    # -----------------------------------------------------------------------
    if run_clicked and can_run:
        st.markdown(divider(), unsafe_allow_html=True)
        st.markdown(section_header("Running Prediction", "Processing pipeline stages..."),
                    unsafe_allow_html=True)

        stage_bar  = st.progress(0.0, text="Initializing CollideX pipeline...")
        status_box = st.empty()
        log_box    = st.expander("Pipeline Log", expanded=False)

        def update_progress(val, msg):
            stage_bar.progress(val, text=f"Stage {int(val*5)}/5 — {msg}")
            status_box.markdown(info_box(msg), unsafe_allow_html=True)

        t0 = time.time()
        with st.spinner("Running CollideX Inference Engine..."):
            success, log_text, report_df, pos_df = run_prediction(
                tle_path=tle_path,
                top_n=top_n,
                progress_callback=update_progress,
            )

        elapsed = time.time() - t0
        stage_bar.progress(1.0, text="Prediction complete.")

        with log_box:
            st.code(log_text, language="")

        if success:
            status_box.markdown(
                f'<div class="info-box" style="background:rgba(0,255,159,0.06);'
                f'border-color:rgba(0,255,159,0.3);">'
                f'<span style="color:#00FF9F;margin-right:8px;">✓</span>'
                f'<span style="color:#8BA3C7;">Prediction completed in {elapsed:.1f}s</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            _display_results(report_df, pos_df)
        else:
            status_box.error("Prediction failed. Check the pipeline log above for details.")

    elif load_clicked:
        report_df = load_collision_report()
        pos_df    = load_future_positions()
        if report_df.empty:
            st.warning("No existing results found. Run a prediction first.")
        else:
            _display_results(report_df, pos_df)


def _display_results(report_df: pd.DataFrame, pos_df: pd.DataFrame):
    """Render all prediction result sections."""
    if report_df.empty:
        st.warning("Result data is empty.")
        return

    st.markdown(divider(), unsafe_allow_html=True)
    st.markdown(section_header("Prediction Results", "Hybrid RF + LSTM Collision Analysis"),
                unsafe_allow_html=True)

    # ---- KPI Row ----
    risk = get_risk_summary(report_df)
    total = len(report_df)
    avg_prob = report_df["collision_probability"].mean()
    max_prob = report_df["collision_probability"].max()

    k1, k2, k3, k4, k5 = st.columns(5)
    cards = [
        (k1, f"{total:,}",          "Objects Analyzed",    "🛰",  COLORS["accent_cyan"]),
        (k2, f"{risk['High']:,}",   "High Risk",           "🔴",  COLORS["accent_red"]),
        (k3, f"{risk['Medium']:,}", "Medium Risk",         "🟡",  COLORS["accent_yellow"]),
        (k4, f"{risk['Low']:,}",    "Low Risk",            "🟢",  COLORS["accent_green"]),
        (k5, f"{avg_prob:.4f}",     "Avg. Collision Prob", "📊",  COLORS["accent_blue"]),
    ]
    for col, val, lbl, icon, color in cards:
        with col:
            st.markdown(kpi_card(val, lbl, icon, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Gauges ----
    st.markdown(section_header("Risk Indicators"), unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(gauge_chart(max_prob, "Peak Collision Probability"), width='stretch')
    with g2:
        st.plotly_chart(gauge_chart(avg_prob, "Average Collision Probability"), width='stretch')
    with g3:
        high_frac = risk["High"] / max(total, 1)
        st.plotly_chart(gauge_chart(high_frac, "Fraction High-Risk"), width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Charts ----
    chart_l, chart_r = st.columns(2, gap="medium")
    with chart_l:
        st.plotly_chart(risk_donut(risk), width='stretch')
    with chart_r:
        st.plotly_chart(top_risk_bar(report_df, n=12), width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)
    if "altitude_km" in report_df.columns:
        st.plotly_chart(altitude_scatter(report_df), width='stretch')

    # ---- Top Risk Table ----
    st.markdown(section_header("Top 20 Highest-Risk Objects"), unsafe_allow_html=True)
    top20 = get_top_risk_objects(report_df, n=20)

    # Add colored risk column
    if not top20.empty and "risk_label" in top20.columns:
        def color_risk(val):
            c = RISK_COLORS.get(str(val), "#8BA3C7")
            return f"color: {c}; font-weight: 600;"
        styled = top20.style.map(color_risk, subset=["risk_label"])
        st.dataframe(styled, width='stretch', hide_index=True)
    else:
        st.dataframe(top20, width='stretch', hide_index=True)

    # ---- 3D Positions ----
    if not pos_df.empty and all(c in pos_df.columns for c in ["future_x_km","future_y_km","future_z_km"]):
        st.markdown(section_header("3D Future Position Map"), unsafe_allow_html=True)

        # Merge risk_label into pos_df if possible
        if "risk_label" not in pos_df.columns and not report_df.empty and "norad_id" in pos_df.columns:
            risk_map = report_df.set_index("norad_id")["risk_label"].to_dict()
            pos_df["risk_label"] = pos_df["norad_id"].map(risk_map).fillna("Low")
            if "satellite_name" not in pos_df.columns:
                name_map = report_df.set_index("norad_id")["satellite_name"].to_dict()
                pos_df["satellite_name"] = pos_df["norad_id"].map(name_map)

        st.plotly_chart(position_3d_scatter(pos_df), width='stretch')

    # ---- Download ----
    st.markdown(section_header("Download Results"), unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download Collision Report CSV",
            data=report_df.to_csv(index=False).encode(),
            file_name="collision_report.csv",
            mime="text/csv",
        )
    with dl2:
        if not pos_df.empty:
            st.download_button(
                "Download Future Positions CSV",
                data=pos_df.to_csv(index=False).encode(),
                file_name="future_positions.csv",
                mime="text/csv",
            )
