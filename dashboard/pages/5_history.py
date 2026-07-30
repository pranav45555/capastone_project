"""
PAGE 5 — Prediction History
Searchable, sortable, filterable table of inference_collision_report.csv
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from components.ui import (
    section_header, page_title, divider, info_box, risk_badge
)
from utils.data_loader import load_collision_report, get_risk_summary
from config import COLORS, RISK_COLORS, COLLISION_REPORT_CSV


def render():
    st.markdown(page_title(
        "Prediction History",
        "SEARCH · FILTER · SORT · EXPORT"
    ), unsafe_allow_html=True)

    df = load_collision_report()

    if df.empty:
        st.markdown(info_box(
            "No prediction history found. Run a prediction on the Prediction page first.",
            kind="warning"
        ), unsafe_allow_html=True)
        return

    # -----------------------------------------------------------------------
    # Summary KPIs
    # -----------------------------------------------------------------------
    st.markdown(section_header("Dataset Summary"), unsafe_allow_html=True)
    risk = get_risk_summary(df)
    total = len(df)
    avg_p = df["collision_probability"].mean()
    max_p = df["collision_probability"].max()

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div style="background:rgba(13,31,60,0.8);border:1px solid #1A3A5C;border-radius:12px;
                    padding:16px;text-align:center;">
            <div style="font-size:28px;font-weight:700;color:#00D4FF;
                        font-family:'JetBrains Mono',monospace;">{total:,}</div>
            <div style="font-size:11px;color:#4A6A8A;text-transform:uppercase;
                        letter-spacing:1px;margin-top:4px;">Total Records</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div style="background:rgba(255,56,96,0.08);border:1px solid rgba(255,56,96,0.25);
                    border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:28px;font-weight:700;color:#FF3860;
                        font-family:'JetBrains Mono',monospace;">{risk['High']:,}</div>
            <div style="font-size:11px;color:#FF3860;text-transform:uppercase;
                        letter-spacing:1px;margin-top:4px;">High Risk</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div style="background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.25);
                    border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:28px;font-weight:700;color:#FFD700;
                        font-family:'JetBrains Mono',monospace;">{risk['Medium']:,}</div>
            <div style="font-size:11px;color:#FFD700;text-transform:uppercase;
                        letter-spacing:1px;margin-top:4px;">Medium Risk</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div style="background:rgba(0,255,159,0.08);border:1px solid rgba(0,255,159,0.25);
                    border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:28px;font-weight:700;color:#00FF9F;
                        font-family:'JetBrains Mono',monospace;">{risk['Low']:,}</div>
            <div style="font-size:11px;color:#00FF9F;text-transform:uppercase;
                        letter-spacing:1px;margin-top:4px;">Low Risk</div>
        </div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div style="background:rgba(13,31,60,0.8);border:1px solid #1A3A5C;border-radius:12px;
                    padding:16px;text-align:center;">
            <div style="font-size:28px;font-weight:700;color:#1E6FFF;
                        font-family:'JetBrains Mono',monospace;">{avg_p:.4f}</div>
            <div style="font-size:11px;color:#4A6A8A;text-transform:uppercase;
                        letter-spacing:1px;margin-top:4px;">Avg. Probability</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(divider(), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Filters
    # -----------------------------------------------------------------------
    st.markdown(section_header("Search & Filters"), unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns([2, 1.2, 1.2, 1.2])

    with f1:
        search_term = st.text_input(
            "Search by satellite name or NORAD ID",
            placeholder="e.g. STARLINK, ISS, 25544",
            key="history_search"
        )

    with f2:
        risk_filter = st.multiselect(
            "Risk Class",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
            key="history_risk_filter"
        )

    with f3:
        prob_range = st.slider(
            "Collision Probability",
            min_value=0.0, max_value=1.0,
            value=(0.0, 1.0), step=0.01,
            key="history_prob_filter"
        )

    with f4:
        sort_col = st.selectbox(
            "Sort by",
            ["collision_probability", "norad_id", "altitude_km", "velocity_mag_km_s"],
            index=0, key="history_sort"
        )
        sort_asc = st.checkbox("Ascending", value=False, key="history_sort_asc")

    # -----------------------------------------------------------------------
    # Apply Filters
    # -----------------------------------------------------------------------
    filtered = df.copy()

    if search_term:
        term = search_term.strip().lower()
        mask_name  = filtered["satellite_name"].fillna("").str.lower().str.contains(term, na=False)
        mask_norad = filtered["norad_id"].astype(str).str.contains(term, na=False)
        filtered = filtered[mask_name | mask_norad]

    if risk_filter:
        filtered = filtered[filtered["risk_label"].isin(risk_filter)]

    filtered = filtered[
        (filtered["collision_probability"] >= prob_range[0]) &
        (filtered["collision_probability"] <= prob_range[1])
    ]

    if sort_col in filtered.columns:
        filtered = filtered.sort_values(sort_col, ascending=sort_asc)

    # -----------------------------------------------------------------------
    # Results Count
    # -----------------------------------------------------------------------
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:16px 0 12px;">
        <span style="font-size:13px;color:#4A6A8A;">Showing</span>
        <span style="font-size:16px;font-weight:700;color:#00D4FF;
                     font-family:'JetBrains Mono',monospace;">{len(filtered):,}</span>
        <span style="font-size:13px;color:#4A6A8A;">of {total:,} records</span>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Table Display
    # -----------------------------------------------------------------------
    display_cols = [c for c in [
        "norad_id", "satellite_name", "collision_probability",
        "risk_label", "altitude_km", "velocity_mag_km_s",
        "lstm_pred_alt_km", "lstm_risk_score", "dist_evolution"
    ] if c in filtered.columns]

    display_df = filtered[display_cols].copy()

    # Style function
    def color_risk_label(val):
        colors = {"High": "#FF3860", "Medium": "#FFD700", "Low": "#00FF9F"}
        c = colors.get(str(val), "#8BA3C7")
        return f"color: {c}; font-weight: 600;"

    def color_probability(val):
        try:
            v = float(val)
            if v >= 0.65:
                return "color: #FF3860; font-weight: 700;"
            elif v >= 0.35:
                return "color: #FFD700; font-weight: 600;"
            return "color: #00FF9F;"
        except Exception:
            return ""

    styled = display_df.style
    if "risk_label" in display_df.columns:
        styled = styled.map(color_risk_label, subset=["risk_label"])
    if "collision_probability" in display_df.columns:
        styled = styled.map(color_probability, subset=["collision_probability"])
        styled = styled.format({"collision_probability": "{:.4f}"})
    if "altitude_km" in display_df.columns:
        styled = styled.format({"altitude_km": "{:.2f}"})

    st.dataframe(styled, width='stretch', hide_index=True,
                 height=480)

    st.markdown(divider(), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------------
    st.markdown(section_header("Export"), unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.download_button(
            "Download Filtered CSV",
            data=filtered.to_csv(index=False).encode(),
            file_name="collidex_filtered_report.csv",
            mime="text/csv",
        )
    with dl2:
        st.download_button(
            "Download Full Report CSV",
            data=df.to_csv(index=False).encode(),
            file_name="inference_collision_report.csv",
            mime="text/csv",
        )
    with dl3:
        # Top 50 high risk
        top50 = df.nlargest(50, "collision_probability")
        st.download_button(
            "Download Top 50 High-Risk",
            data=top50.to_csv(index=False).encode(),
            file_name="top50_high_risk.csv",
            mime="text/csv",
        )
