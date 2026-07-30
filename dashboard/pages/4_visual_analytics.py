"""
PAGE 4 — Visual Analytics
Loads and displays all PNG result images + interactive Plotly charts.
"""

import os
import streamlit as st
import pandas as pd
from PIL import Image

from components.ui import (
    section_header, page_title, divider, info_box
)
from components.charts import (
    risk_donut, collision_prob_histogram,
    top_risk_bar, altitude_scatter,
    velocity_altitude_heatmap, risk_time_series,
    position_3d_scatter
)
from utils.data_loader import (
    load_collision_report, load_future_positions,
    load_image_bytes, get_risk_summary
)
from config import (
    CONFUSION_MATRIX_PNG, ROC_CURVE_PNG, FEATURE_IMPORTANCE_PNG,
    PRECISION_RECALL_PNG, COLLISION_HIST_PNG, RISK_DIST_PNG,
    TRAJECTORY_ERROR_PNG, COLORS
)


def _image_panel(image_path: str, title: str, caption: str = ""):
    """Render image card: title header + st.image (native Streamlit, no split HTML)."""
    img_bytes = load_image_bytes(image_path)

    # Header — fully self-contained HTML block
    st.markdown(
        f'<div style="background:rgba(13,31,60,0.8);border:1px solid #1A3A5C;'
        f'border-radius:14px;padding:12px 16px 0;margin-bottom:0;">'
        f'<div style="font-size:12px;font-weight:600;color:#4A6A8A;'
        f'text-transform:uppercase;letter-spacing:1.5px;padding-bottom:10px;'
        f'border-bottom:1px solid #1A3A5C;">{title}</div></div>',
        unsafe_allow_html=True,
    )
    # Image body — wrapped in a styled container
    with st.container():
        if img_bytes:
            st.image(img_bytes, width='stretch')
            if caption:
                st.markdown(
                    f'<p style="font-size:11px;color:#4A6A8A;text-align:center;'
                    f'margin-top:4px;padding-bottom:8px;">{caption}</p>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<div style="height:140px;display:flex;align-items:center;'
                f'justify-content:center;color:#4A6A8A;font-size:12px;">'
                f'Image not found: {os.path.basename(image_path)}</div>',
                unsafe_allow_html=True,
            )


def render():
    st.markdown(page_title(
        "Visual Analytics",
        "MODEL DIAGNOSTICS · RISK VISUALIZATION · ORBITAL ANALYTICS"
    ), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # TAB LAYOUT
    # -----------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "Model Diagnostics",
        "Risk Analytics",
        "Orbital Visualization"
    ])

    # =====================================================================
    # TAB 1 — Model Diagnostics (static PNGs)
    # =====================================================================
    with tab1:
        st.markdown(section_header(
            "Model Diagnostic Plots",
            "Auto-loaded from results/ directory"
        ), unsafe_allow_html=True)

        # Row 1: Confusion Matrix + ROC Curve
        row1_l, row1_r = st.columns(2, gap="medium")
        with row1_l:
            _image_panel(
                CONFUSION_MATRIX_PNG,
                "Confusion Matrix",
                "Normalized per-class prediction accuracy"
            )
        with row1_r:
            _image_panel(
                ROC_CURVE_PNG,
                "ROC Curve",
                "Receiver Operating Characteristic — macro OVR"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 2: Precision-Recall + Feature Importance
        row2_l, row2_r = st.columns(2, gap="medium")
        with row2_l:
            _image_panel(
                PRECISION_RECALL_PNG,
                "Precision-Recall Curve",
                "Trade-off between precision and recall per class"
            )
        with row2_r:
            _image_panel(
                FEATURE_IMPORTANCE_PNG,
                "Feature Importance",
                "Top features ranked by Random Forest importance score"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 3: Histogram + Risk Dist + Trajectory Error
        row3_a, row3_b, row3_c = st.columns(3, gap="medium")
        with row3_a:
            _image_panel(
                COLLISION_HIST_PNG,
                "Collision Prob. Histogram",
                "Distribution of predicted probabilities"
            )
        with row3_b:
            _image_panel(
                RISK_DIST_PNG,
                "Risk Class Distribution",
                "Low / Medium / High class breakdown"
            )
        with row3_c:
            _image_panel(
                TRAJECTORY_ERROR_PNG,
                "Trajectory Error Distribution",
                "LSTM prediction error per output variable"
            )

    # =====================================================================
    # TAB 2 — Risk Analytics (interactive Plotly)
    # =====================================================================
    with tab2:
        df = load_collision_report()

        if df.empty:
            st.markdown(info_box(
                "No collision report found. Run a prediction to generate data.",
                kind="warning"
            ), unsafe_allow_html=True)
            return

        risk = get_risk_summary(df)

        st.markdown(section_header(
            "Interactive Risk Analytics",
            "Click · Zoom · Pan · Hover for details"
        ), unsafe_allow_html=True)

        # Mini summary
        tot = len(df)
        c1, c2, c3 = st.columns(3)
        for col, lbl, cnt, color in [
            (c1, "High Risk",   risk["High"],   "#FF3860"),
            (c2, "Medium Risk", risk["Medium"], "#FFD700"),
            (c3, "Low Risk",    risk["Low"],    "#00FF9F"),
        ]:
            with col:
                pct = cnt / max(tot, 1) * 100
                st.markdown(f"""
                <div style="background:rgba(13,31,60,0.7);border:1px solid #1A3A5C;
                            border-radius:10px;padding:12px 16px;text-align:center;">
                    <div style="font-size:26px;font-weight:700;color:{color};
                                font-family:'JetBrains Mono',monospace;">{cnt:,}</div>
                    <div style="font-size:11px;color:{color};text-transform:uppercase;
                                letter-spacing:1px;">{lbl}</div>
                    <div style="font-size:12px;color:#4A6A8A;">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Donut + Histogram
        d1, d2 = st.columns(2, gap="medium")
        with d1:
            st.plotly_chart(risk_donut(risk), width='stretch')
        with d2:
            st.plotly_chart(collision_prob_histogram(df), width='stretch')

        # Top Risk Bar
        st.plotly_chart(top_risk_bar(df, n=20), width='stretch')

        # Scatter + Heatmap
        s1, s2 = st.columns(2, gap="medium")
        with s1:
            if "altitude_km" in df.columns:
                st.plotly_chart(altitude_scatter(df), width='stretch')
        with s2:
            if all(c in df.columns for c in ["altitude_km", "velocity_mag_km_s"]):
                st.plotly_chart(velocity_altitude_heatmap(df), width='stretch')

        # NORAD timeline
        if "norad_id" in df.columns:
            st.plotly_chart(risk_time_series(df), width='stretch')

    # =====================================================================
    # TAB 3 — Orbital Visualization
    # =====================================================================
    with tab3:
        pos_df = load_future_positions()
        report_df = load_collision_report()

        st.markdown(section_header(
            "3D Orbital Visualization",
            "Future position coordinates — SGP4 + LSTM"
        ), unsafe_allow_html=True)

        if pos_df.empty:
            st.markdown(info_box(
                "No future positions data found. Run a prediction first.",
                kind="warning"
            ), unsafe_allow_html=True)
            return

        # Merge risk labels
        if "risk_label" not in pos_df.columns and not report_df.empty and "norad_id" in pos_df.columns:
            risk_map = report_df.set_index("norad_id")["risk_label"].to_dict()
            pos_df["risk_label"] = pos_df["norad_id"].map(risk_map).fillna("Low")
            if "satellite_name" not in pos_df.columns:
                name_map = report_df.set_index("norad_id")["satellite_name"].to_dict()
                pos_df["satellite_name"] = pos_df["norad_id"].map(name_map)

        sample_size = st.slider(
            "Number of objects to plot (3D)",
            min_value=100, max_value=min(5000, len(pos_df)),
            value=min(1000, len(pos_df)), step=100,
            help="More objects = slower render"
        )
        st.plotly_chart(
            position_3d_scatter(pos_df, n_sample=sample_size),
            width='stretch'
        )

        # Altitude stats
        if "altitude_km" in pos_df.columns:
            st.markdown(section_header("Altitude Statistics"), unsafe_allow_html=True)
            alt = pos_df["altitude_km"].dropna()
            asc1, asc2, asc3, asc4 = st.columns(4)
            for col, lbl, val in [
                (asc1, "Min Altitude",  f"{alt.min():.1f} km"),
                (asc2, "Max Altitude",  f"{alt.max():.1f} km"),
                (asc3, "Mean Altitude", f"{alt.mean():.1f} km"),
                (asc4, "Std Altitude",  f"{alt.std():.1f} km"),
            ]:
                with col:
                    st.markdown(f"""
                    <div style="background:rgba(13,31,60,0.7);border:1px solid #1A3A5C;
                                border-radius:10px;padding:14px;text-align:center;">
                        <div style="font-size:22px;font-weight:700;color:#00D4FF;
                                    font-family:'JetBrains Mono',monospace;">{val}</div>
                        <div style="font-size:11px;color:#4A6A8A;text-transform:uppercase;
                                    letter-spacing:1px;margin-top:4px;">{lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)
