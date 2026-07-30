"""
PAGE 7 — About
Problem statement, architecture, dataset, algorithms, workflow, future scope.
"""

import streamlit as st

from components.ui import (
    section_header, page_title, divider, pipeline_step, info_box
)
from config import COLORS


def render():
    st.markdown(page_title(
        "About CollideX",
        "MISSION · ARCHITECTURE · TEAM"
    ), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Mission Statement
    # -----------------------------------------------------------------------
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(0,212,255,0.06),rgba(30,111,255,0.04));
                border:1px solid rgba(0,212,255,0.2);border-radius:16px;
                padding:32px;margin-bottom:32px;">
        <div style="font-size:11px;color:#00D4FF;text-transform:uppercase;
                    letter-spacing:3px;margin-bottom:12px;font-weight:600;">
            MISSION STATEMENT
        </div>
        <div style="font-size:22px;font-weight:700;color:#E8F4FD;
                    line-height:1.4;margin-bottom:16px;">
            Protecting orbital infrastructure through AI-powered collision prediction.
        </div>
        <div style="font-size:14px;color:#8BA3C7;line-height:1.7;">
            CollideX is a capstone project that demonstrates how machine learning can
            address one of the most pressing challenges in modern space operations —
            the growing threat of orbital debris collisions. By fusing physics-based
            SGP4 propagation with deep learning trajectory refinement, CollideX
            achieves near-perfect classification of satellite collision risk.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Problem Statement
    # -----------------------------------------------------------------------
    st.markdown(section_header("Problem Statement"), unsafe_allow_html=True)

    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.markdown("""
        <div style="font-size:14px;color:#8BA3C7;line-height:1.8;padding:4px 0;">
            <p>The Low Earth Orbit (LEO) environment currently hosts over <strong style="color:#00D4FF;">27,000+
            tracked objects</strong>, with millions of sub-cm debris fragments that remain untracked.
            The collision of even a small fragment with an operational satellite can cause catastrophic
            mission failure and generate a cascade of new debris (the Kessler Syndrome).</p>
            <p>Traditional conjunction analysis (e.g., ESA CARA, NASA CARA) relies on
            high-fidelity ephemeris propagation and Monte Carlo simulations — computationally
            expensive operations that may not scale to real-time monitoring of the
            growing catalog.</p>
            <p>CollideX addresses this by applying a hybrid AI approach that can analyze
            the <strong style="color:#00D4FF;">entire tracked catalog in seconds</strong>
            with near-perfect accuracy.</p>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        facts = [
            ("Tracked LEO objects",           "27,000+"),
            ("Annual conjunction warnings",   "~1,000+"),
            ("Kessler cascade threshold",     "LEO, 500–2000 km"),
            ("Most dangerous recent event",   "Cosmos 954 × Iridium 33"),
            ("ISS collision avoidance maneuvers","~3 per year"),
            ("Average CDM processing time",   "24–48 hours (traditional)"),
            ("CollideX inference time",       "< 5 minutes (full catalog)"),
        ]
        for label, value in facts:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        padding:9px 0;border-bottom:1px solid #1A3A5C;">
                <span style="font-size:13px;color:#8BA3C7;">{label}</span>
                <span style="font-size:13px;font-weight:600;color:#00D4FF;
                             font-family:'JetBrains Mono',monospace;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(divider(), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # System Architecture
    # -----------------------------------------------------------------------
    st.markdown(section_header("System Architecture",
                               "Hybrid SGP4 + LSTM + Random Forest Fusion"),
                unsafe_allow_html=True)

    arc1, arc2 = st.columns([1, 1.2], gap="large")

    with arc1:
        for i, (step, detail) in enumerate([
            ("TLE Ingestion",             "Space-Track 3LE catalog, 15,000+ objects"),
            ("SGP4 Propagation",          "h = 1, 6, 12, 24 hour horizons"),
            ("Trajectory Feature Eng.",   "Distance, altitude, velocity, risk score"),
            ("LSTM Encoder → Regressor",  "Input (3,6) → Output (6,) at h=24"),
            ("Hybrid Risk Fusion",        "50% LSTM + 30% SGP4 + 20% altitude"),
            ("Risk Classification",       "RF Classifier → Low / Medium / High"),
            ("Outputs",                   "collision_probability, risk_class, positions"),
        ], 1):
            st.markdown(pipeline_step(i, step, detail), unsafe_allow_html=True)

    with arc2:
        st.markdown("""
        <div style="background:rgba(13,31,60,0.7);border:1px solid #1A3A5C;
                    border-radius:12px;padding:20px;">
            <div style="font-size:11px;color:#4A6A8A;letter-spacing:1.5px;
                        text-transform:uppercase;margin-bottom:16px;font-weight:600;">
                Hybrid Risk Formula
            </div>
            <div style="background:rgba(10,22,40,0.8);border:1px solid #1A3A5C;
                        border-radius:8px;padding:16px;font-family:'JetBrains Mono',monospace;
                        font-size:13px;color:#00D4FF;line-height:2;">
                P_hybrid = <br>
                &nbsp;&nbsp;0.50 × lstm_trajectory_risk<br>
                &nbsp;&nbsp;+ 0.30 × sgp4_distance_evolution<br>
                &nbsp;&nbsp;+ 0.20 × altitude_orbital_risk
            </div>
            <div style="font-size:12px;color:#4A6A8A;margin-top:12px;">
                Inspired by ESA CARA hybrid conjunction method. Risk thresholds:
                <br>• P ≥ 0.65 → <span style="color:#FF3860;font-weight:600;">HIGH RISK</span>
                <br>• 0.35 ≤ P < 0.65 → <span style="color:#FFD700;font-weight:600;">MEDIUM RISK</span>
                <br>• P < 0.35 → <span style="color:#00FF9F;font-weight:600;">LOW RISK</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(divider(), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------------
    st.markdown(section_header("Datasets"), unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3, gap="medium")
    datasets = [
        (d1, "Space-Track TLE Catalog",
         "Two-Line Element sets for 15,000+ tracked objects. "
         "Used for SGP4 propagation and coordinate generation.",
         "3LE format · 15,299 objects"),
        (d2, "Hybrid RF Dataset",
         "162,623 samples with 33 features generated from SGP4 propagation "
         "and trajectory feature engineering.",
         "162,623 samples · 33 features"),
        (d3, "LSTM Sequence Dataset",
         "15,298 sequences of shape (3,6) built from SGP4 outputs at "
         "h=1, 6, 12 hours, predicting h=24 state vector.",
         "15,298 sequences · (3,6) input"),
    ]
    for col, title, desc, meta in datasets:
        with col:
            st.markdown(f"""
            <div style="background:rgba(13,31,60,0.7);border:1px solid #1A3A5C;
                        border-radius:12px;padding:20px;height:100%;">
                <div style="font-size:14px;font-weight:700;color:#E8F4FD;margin-bottom:8px;">
                    {title}
                </div>
                <div style="font-size:13px;color:#8BA3C7;line-height:1.6;margin-bottom:12px;">
                    {desc}
                </div>
                <div style="font-size:11px;color:#1E6FFF;font-family:'JetBrains Mono',monospace;">
                    {meta}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(divider(), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Algorithms
    # -----------------------------------------------------------------------
    st.markdown(section_header("Algorithms"), unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3, gap="medium")
    algos = [
        (a1, "SGP4 / SDP4", "Physics-Based Propagation",
         "Simplified General Perturbations model. Propagates TLE orbital elements to "
         "Cartesian ECI coordinates at arbitrary future epochs. Industry standard "
         "used by NORAD, NASA, and ESA.",
         ["Deterministic", "High speed", "Physics-grounded"]),
        (a2, "LSTM Neural Network", "Deep Learning Regressor",
         "Sequential LSTM encoder-regressor trained on SGP4 trajectory sequences. "
         "Architecture: LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.1) → "
         "Dense(16, relu) → Dense(6). Predicts h=24 orbital state.",
         ["R² = 0.849", "MAE = 0.069 (norm.)", "Batch inference"]),
        (a3, "Hybrid Random Forest", "Ensemble Classifier",
         "200-tree Random Forest with balanced class weights trained on "
         "hybrid feature set. Achieves 99.99% accuracy on 3-class risk classification.",
         ["200 trees", "99.99% accuracy", "ROC-AUC = 1.000"]),
    ]
    for col, name, stype, desc, bullets in algos:
        with col:
            bl = "".join([f'<li style="color:#8BA3C7;font-size:12px;">{b}</li>' for b in bullets])
            st.markdown(f"""
            <div style="background:rgba(13,31,60,0.7);border:1px solid #1A3A5C;
                        border-radius:12px;padding:20px;">
                <div style="font-size:11px;color:#00D4FF;text-transform:uppercase;
                            letter-spacing:1.5px;margin-bottom:6px;font-weight:600;">
                    {stype}
                </div>
                <div style="font-size:15px;font-weight:700;color:#E8F4FD;margin-bottom:10px;">
                    {name}
                </div>
                <div style="font-size:13px;color:#8BA3C7;line-height:1.6;margin-bottom:12px;">
                    {desc}
                </div>
                <ul style="padding-left:16px;margin:0;">{bl}</ul>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(divider(), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Future Scope
    # -----------------------------------------------------------------------
    st.markdown(section_header("Future Scope"), unsafe_allow_html=True)

    future_items = [
        ("Real-Time TLE Streaming",
         "Direct integration with Space-Track API for automated catalog refresh every 24 hours."),
        ("Transformer-Based Trajectory Model",
         "Replace LSTM with a Transformer encoder for multi-satellite attention modeling."),
        ("Conjunction Data Message (CDM) Generation",
         "Auto-generate CCSDS CDM format alerts for high-risk pairs."),
        ("Monte Carlo Uncertainty Propagation",
         "Add covariance matrix propagation for probabilistic miss distance estimation."),
        ("Maneuver Planning Assistant",
         "Suggest collision avoidance maneuver delta-V for medium/high risk objects."),
        ("Global Debris Field Simulation",
         "Long-term Kessler cascade simulation using the Collisional Cascading Model."),
    ]

    cols = st.columns(2, gap="medium")
    for i, (title, desc) in enumerate(future_items):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="display:flex;gap:12px;padding:14px 0;border-bottom:1px solid #1A3A5C;">
                <div style="background:linear-gradient(135deg,#1E6FFF,#00D4FF);
                            color:#050A14;font-weight:700;font-size:11px;
                            width:24px;height:24px;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;
                            flex-shrink:0;margin-top:2px;">{i+1}</div>
                <div>
                    <div style="font-size:13px;font-weight:600;color:#E8F4FD;margin-bottom:4px;">
                        {title}
                    </div>
                    <div style="font-size:12px;color:#4A6A8A;line-height:1.5;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(divider(), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Contributors
    # -----------------------------------------------------------------------
    st.markdown(section_header("Contributors"), unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(13,31,60,0.7);border:1px solid #1A3A5C;
                border-radius:14px;padding:28px;">
        <div style="display:flex;flex-wrap:wrap;gap:24px;justify-content:center;">
            <div style="text-align:center;min-width:160px;">
                <div style="width:56px;height:56px;border-radius:50%;
                            background:linear-gradient(135deg,#1E6FFF,#00D4FF);
                            display:flex;align-items:center;justify-content:center;
                            font-size:22px;font-weight:800;color:#050A14;
                            margin:0 auto 12px;">P</div>
                <div style="font-size:14px;font-weight:700;color:#E8F4FD;">Putra</div>
                <div style="font-size:12px;color:#4A6A8A;margin-top:4px;">
                    ML Engineer · Full Stack · Project Lead
                </div>
                <div style="margin-top:8px;">
                    <span style="background:rgba(0,212,255,0.1);color:#00D4FF;
                                 font-size:10px;padding:3px 8px;border-radius:20px;
                                 border:1px solid rgba(0,212,255,0.3);">
                        CollideX Capstone 2026
                    </span>
                </div>
            </div>
        </div>
        <div style="text-align:center;margin-top:24px;padding-top:20px;
                    border-top:1px solid #1A3A5C;">
            <div style="font-size:12px;color:#4A6A8A;">
                Built with Python · Streamlit · TensorFlow · scikit-learn · SGP4 · Plotly
            </div>
            <div style="font-size:12px;color:#4A6A8A;margin-top:4px;">
                Inspired by ESA CARA · NASA Debris Office · Space-Track.org
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
