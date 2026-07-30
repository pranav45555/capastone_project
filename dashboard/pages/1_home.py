"""
PAGE 1 — Home
CollideX Professional Landing Page
"""

import streamlit as st
import pandas as pd

from components.ui import (
    metric_card, section_header, hero_section,
    pipeline_step, status_indicator, info_box, divider
)
from utils.data_loader import (
    load_collision_report, get_risk_summary,
    count_tle_objects, load_metrics, metric_value
)
from config import (
    COLORS, DEFAULT_TLE_FILE, LEO_TLE_FILE,
    MODEL_ACCURACY, MODEL_ROC_AUC, TOTAL_SATELLITES,
    INFERENCE_TIME_MS, TRAINING_SAMPLES, TESTING_SAMPLES
)


def render():
    # -----------------------------------------------------------------------
    # Hero Section
    # -----------------------------------------------------------------------
    st.markdown(hero_section(
        title="CollideX",
        subtitle=(
            "AI-powered space debris collision prediction using hybrid "
            "SGP4 propagation, LSTM trajectory refinement, and ensemble "
            "risk fusion — built for operational aerospace teams."
        ),
        tagline="AI-BASED SPACE DEBRIS COLLISION PREDICTION SYSTEM"
    ), unsafe_allow_html=True)

    # CTA Buttons — use session_state for importlib-based routing
    col_a, col_b, col_c, col_spacer = st.columns([1, 1, 1, 3])
    with col_a:
        if st.button("Run Prediction", key="home_cta_predict", type="primary"):
            st.session_state["current_page"] = "Prediction"
            st.rerun()
    with col_b:
        if st.button("View Evaluation", key="home_cta_eval"):
            st.session_state["current_page"] = "Evaluation"
            st.rerun()
    with col_c:
        if st.button("Analytics", key="home_cta_analytics"):
            st.session_state["current_page"] = "Visual Analytics"
            st.rerun()

    st.markdown(divider(), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Live Statistics Cards
    # -----------------------------------------------------------------------
    st.markdown(section_header(
        "System Overview",
        "Real-time statistics from the latest prediction run"
    ), unsafe_allow_html=True)

    df = load_collision_report()
    risk = get_risk_summary(df)
    total_obj  = len(df) if not df.empty else TOTAL_SATELLITES
    high_risk  = risk.get("High", 0)
    med_risk   = risk.get("Medium", 0)
    tle_count  = count_tle_objects(DEFAULT_TLE_FILE) or TOTAL_SATELLITES

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card(
            value=f"{tle_count:,}",
            label="Tracked Objects",
            sublabel="Space-Track TLE Catalog",
            color=COLORS["accent_cyan"],
            icon="🛰"
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card(
            value=f"{total_obj:,}",
            label="Objects Scored",
            sublabel="Latest Prediction Run",
            color=COLORS["accent_blue"],
            icon="📡"
        ), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card(
            value=f"{MODEL_ACCURACY:.2f}%",
            label="Model Accuracy",
            sublabel="Hybrid RF + LSTM Fusion",
            color=COLORS["accent_green"],
            icon="🎯"
        ), unsafe_allow_html=True)
    with col4:
        status_color = COLORS["accent_red"] if high_risk > 50 else COLORS["accent_yellow"]
        st.markdown(metric_card(
            value=f"{high_risk:,}",
            label="High-Risk Objects",
            sublabel=f"{med_risk:,} Medium · {risk.get('Low',0):,} Low",
            color=status_color,
            icon="⚠"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Risk Overview Mini Row
    # -----------------------------------------------------------------------
    if not df.empty:
        r1, r2, r3 = st.columns(3)
        total = max(len(df), 1)
        with r1:
            pct = risk.get("High", 0) / total * 100
            st.markdown(f"""
            <div style="background:rgba(255,56,96,0.08);border:1px solid rgba(255,56,96,0.25);
                        border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:32px;font-weight:800;color:#FF3860;
                            font-family:'JetBrains Mono',monospace;">
                    {risk.get("High",0):,}
                </div>
                <div style="font-size:11px;color:#FF3860;text-transform:uppercase;
                            letter-spacing:1.5px;margin-top:4px;">High Risk</div>
                <div style="font-size:12px;color:#4A6A8A;margin-top:4px;">{pct:.1f}% of catalog</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            pct = risk.get("Medium", 0) / total * 100
            st.markdown(f"""
            <div style="background:rgba(255,215,0,0.08);border:1px solid rgba(255,215,0,0.25);
                        border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:32px;font-weight:800;color:#FFD700;
                            font-family:'JetBrains Mono',monospace;">
                    {risk.get("Medium",0):,}
                </div>
                <div style="font-size:11px;color:#FFD700;text-transform:uppercase;
                            letter-spacing:1.5px;margin-top:4px;">Medium Risk</div>
                <div style="font-size:12px;color:#4A6A8A;margin-top:4px;">{pct:.1f}% of catalog</div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            pct = risk.get("Low", 0) / total * 100
            st.markdown(f"""
            <div style="background:rgba(0,255,159,0.08);border:1px solid rgba(0,255,159,0.25);
                        border-radius:12px;padding:16px;text-align:center;">
                <div style="font-size:32px;font-weight:800;color:#00FF9F;
                            font-family:'JetBrains Mono',monospace;">
                    {risk.get("Low",0):,}
                </div>
                <div style="font-size:11px;color:#00FF9F;text-transform:uppercase;
                            letter-spacing:1.5px;margin-top:4px;">Low Risk</div>
                <div style="font-size:12px;color:#4A6A8A;margin-top:4px;">{pct:.1f}% of catalog</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Pipeline + Model Info
    # -----------------------------------------------------------------------
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(section_header("Prediction Pipeline", "Hybrid Fusion Architecture"),
                    unsafe_allow_html=True)
        steps = [
            ("Space-Track TLE Catalog",     "3LE format, full catalog"),
            ("SGP4 Orbit Propagation",       "h = 1, 6, 12, 24 hours"),
            ("Trajectory Feature Eng.",      "distance, velocity, risk score"),
            ("LSTM Trajectory Refinement",   "Encoder → Regressor"),
            ("Hybrid Risk Fusion",           "50% LSTM + 30% SGP4 + 20% altitude"),
            ("Outputs Generated",            "collision_probability, risk_class, positions"),
        ]
        for i, (label, detail) in enumerate(steps, 1):
            st.markdown(pipeline_step(i, label, detail), unsafe_allow_html=True)

    with right:
        st.markdown(section_header("Model Performance", "Latest Evaluation Results"),
                    unsafe_allow_html=True)
        metrics_df = load_metrics()

        rows = [
            ("Model",           "Hybrid RF + LSTM",             False),
            ("Accuracy",        f"{MODEL_ACCURACY:.4f}%",        True),
            ("ROC-AUC",         f"{MODEL_ROC_AUC:.4f}%",        True),
            ("Training Samples",f"{TRAINING_SAMPLES:,}",        False),
            ("Testing Samples", f"{TESTING_SAMPLES:,}",         False),
            ("Inference Time",  f"{INFERENCE_TIME_MS:.2f} ms",  False),
            ("N Trees",         "200",                           False),
            ("Max Depth",       "12",                            False),
        ]
        # Build entire table as ONE html block — Streamlit closes all tags between calls
        table_html = '<div style="padding:8px 0;">'
        for label, value, hl in rows:
            color = COLORS["accent_cyan"] if hl else COLORS["text_primary"]
            table_html += (
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:9px 0;border-bottom:1px solid #1A3A5C;">'
                f'<span style="font-size:13px;color:#8BA3C7;">{label}</span>'
                f'<span style="font-size:13px;font-weight:600;color:{color};'
                f'font-family:\'JetBrains Mono\',monospace;">{value}</span>'
                f'</div>'
            )
        table_html += '</div>'
        st.markdown(table_html, unsafe_allow_html=True)

    st.markdown(divider(), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # System Status
    # -----------------------------------------------------------------------
    st.markdown(section_header("System Status"), unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)

    import os
    from config import (COLLISION_REPORT_CSV, METRICS_CSV, RESULTS_DIR,
                        DEFAULT_TLE_FILE as TLE_F)

    checks = [
        (sc1, "Prediction Engine",  os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "scripts", "predict.py"))),
        (sc2, "TLE Catalog",        os.path.exists(TLE_F)),
        (sc3, "Collision Report",   os.path.exists(COLLISION_REPORT_CSV)),
        (sc4, "Evaluation Metrics", os.path.exists(METRICS_CSV)),
    ]
    for col, label, ok in checks:
        with col:
            status = "online" if ok else "offline"
            dot_color = "#00FF9F" if ok else "#FF3860"
            state_txt = "READY" if ok else "MISSING"
            st.markdown(f"""
            <div style="background:rgba(13,31,60,0.8);border:1px solid #1A3A5C;
                        border-radius:10px;padding:14px 16px;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="width:8px;height:8px;border-radius:50%;
                                 background:{dot_color};display:inline-block;
                                 box-shadow:0 0 6px {dot_color};"></span>
                    <span style="font-size:12px;font-weight:600;color:#8BA3C7;">
                        {label}
                    </span>
                </div>
                <div style="font-size:18px;font-weight:700;color:{dot_color};
                            font-family:'JetBrains Mono',monospace;margin-top:6px;">
                    {state_txt}
                </div>
            </div>
            """, unsafe_allow_html=True)
