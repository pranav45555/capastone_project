"""
CollideX Dashboard — Main Entry Point
========================================
Professional multi-page Streamlit application.
Navigation via session_state + importlib (stable across all Streamlit versions).

Run:
    cd cosmicguard/dashboard
    streamlit run app.py
"""

import sys
import os
import importlib.util

# ---------------------------------------------------------------------------
# Path bootstrap — make dashboard root importable
# ---------------------------------------------------------------------------
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)

import streamlit as st
from assets.styles import GLOBAL_CSS
from components.ui import sidebar_logo
from config import APP_TITLE, APP_SUBTITLE, APP_ICON


# ---------------------------------------------------------------------------
# Helper — load a page module by file path
# ---------------------------------------------------------------------------
def _load_page(rel_path: str):
    """Dynamically load a page module and call render()."""
    abs_path = os.path.join(DASHBOARD_DIR, rel_path)
    spec     = importlib.util.spec_from_file_location("_page_module", abs_path)
    mod      = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Page Config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_TITLE} — Space Debris Collision Prediction",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": f"### {APP_TITLE}\n{APP_SUBTITLE}",
    }
)

# ---------------------------------------------------------------------------
# Inject Global CSS + Sidebar Nav Styling
# ---------------------------------------------------------------------------
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.markdown("""
<style>
/* Style sidebar nav buttons to look like clean link-style navigation */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    color: #8BA3C7 !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    letter-spacing: 0.4px !important;
    padding: 10px 16px !important;
    width: 100% !important;
    text-align: left !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] .stButton > button:hover {
    color: #00D4FF !important;
    background: rgba(0,212,255,0.07) !important;
    border: none !important;
    box-shadow: none !important;
}
/* Active page button */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] .stButton > button[kind="primary"] {
    color: #00D4FF !important;
    font-weight: 700 !important;
    background: rgba(0,212,255,0.10) !important;
    border-left: 3px solid #00D4FF !important;
    border-right: none !important;
    border-top: none !important;
    border-bottom: none !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] .stButton > button[kind="primary"]:hover {
    background: rgba(0,212,255,0.15) !important;
}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] .stButton > button:focus {
    box-shadow: none !important;
    outline: none !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navigation Config
# ---------------------------------------------------------------------------
NAV_PAGES = [
    ("Home",               "pages/1_home.py",             "🏠"),
    ("Prediction",         "pages/2_prediction.py",       "⚡"),
    ("Evaluation",         "pages/3_evaluation.py",       "📊"),
    ("Visual Analytics",   "pages/4_visual_analytics.py", "📈"),
    ("Prediction History", "pages/5_history.py",          "🕐"),
    ("Downloads",          "pages/6_downloads.py",        "📥"),
    ("About",              "pages/7_about.py",            "ℹ️"),
]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # Logo area
    st.markdown(sidebar_logo(), unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:12px 20px 8px;">
        <div style="font-size:9px;color:#1A3A5C;text-transform:uppercase;
                    letter-spacing:2px;font-weight:700;">Navigation</div>
    </div>
    """, unsafe_allow_html=True)

    # Default to Home
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"

    # Nav buttons — styled like clean link navigation
    for page_name, _, icon in NAV_PAGES:
        is_active = st.session_state["current_page"] == page_name
        btn = st.button(
            f"{icon}  {page_name}",
            key=f"nav_btn_{page_name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        )
        if btn:
            st.session_state["current_page"] = page_name
            st.rerun()

    # System Status block
    st.markdown("<br>" * 2, unsafe_allow_html=True)
    st.markdown("""
    <div style="border-top:1px solid #1A3A5C;padding:16px 20px 0;">
        <div style="font-size:9px;color:#1A3A5C;text-transform:uppercase;
                    letter-spacing:2px;font-weight:700;margin-bottom:10px;">
            System Status
        </div>
    """, unsafe_allow_html=True)

    from config import COLLISION_REPORT_CSV, METRICS_CSV, DEFAULT_TLE_FILE
    _checks = [
        ("Predict Engine", os.path.exists(os.path.join(DASHBOARD_DIR, "..", "scripts", "predict.py"))),
        ("TLE Catalog",    os.path.exists(DEFAULT_TLE_FILE)),
        ("Results",        os.path.exists(COLLISION_REPORT_CSV)),
        ("Metrics",        os.path.exists(METRICS_CSV)),
    ]
    for lbl, ok in _checks:
        dot_c = "#00FF9F" if ok else "#FF3860"
        state = "OK" if ok else "MISSING"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:3px 0;">
            <span style="width:6px;height:6px;border-radius:50%;
                         background:{dot_c};display:inline-block;"></span>
            <span style="font-size:11px;color:#4A6A8A;flex:1;">{lbl}</span>
            <span style="font-size:10px;color:{dot_c};
                         font-family:'JetBrains Mono',monospace;">{state}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Version footer
    st.markdown("""
    <div style="text-align:center;padding:16px 0 4px;">
        <div style="font-size:10px;color:#1A3A5C;font-family:'JetBrains Mono',monospace;">
            CollideX v1.0 · 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page Routing — dynamically load the selected page file
# ---------------------------------------------------------------------------
current = st.session_state.get("current_page", "Home")

page_file = next(
    (f for name, f, _ in NAV_PAGES if name == current),
    "pages/1_home.py"
)

page_mod = _load_page(page_file)
page_mod.render()
