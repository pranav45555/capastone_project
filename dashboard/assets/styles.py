"""
CollideX Dashboard — Global CSS Injection
==========================================
Professional aerospace dark-theme stylesheet.
All styles injected via st.markdown() in app.py.
"""

GLOBAL_CSS = """
<style>
/* ============================
   FONTS
   ============================ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ============================
   ROOT VARIABLES
   ============================ */
:root {
    --bg-primary:      #050A14;
    --bg-secondary:    #0A1628;
    --bg-card:         #0D1F3C;
    --accent-cyan:     #00D4FF;
    --accent-blue:     #1E6FFF;
    --accent-green:    #00FF9F;
    --accent-red:      #FF3860;
    --accent-yellow:   #FFD700;
    --text-primary:    #E8F4FD;
    --text-secondary:  #8BA3C7;
    --text-muted:      #4A6A8A;
    --border:          #1A3A5C;
    --radius:          12px;
    --radius-lg:       18px;
}

/* ============================
   GLOBAL RESET & BASE
   ============================ */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: linear-gradient(135deg, #050A14 0%, #0A1628 50%, #050E20 100%) !important;
    background-attachment: fixed !important;
}

/* ============================
   HIDE DEFAULT STREAMLIT UI
   ============================ */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.stDeployButton { display: none; }
.viewerBadge_container__1QSob { display: none !important; }

/* ============================
   SIDEBAR
   ============================ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050A14 0%, #070D1A 100%) !important;
    border-right: 1px solid #1A3A5C !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

[data-testid="stSidebarNav"] {
    padding-top: 0 !important;
}

.sidebar-logo-area {
    padding: 24px 20px 16px;
    border-bottom: 1px solid #1A3A5C;
    margin-bottom: 8px;
}

.sidebar-logo-title {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 2px;
    color: #00D4FF;
    text-transform: uppercase;
}

.sidebar-logo-sub {
    font-size: 10px;
    color: #4A6A8A;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 2px;
}

.sidebar-version {
    font-size: 10px;
    color: #1E6FFF;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* ============================
   SIDEBAR NAV LINKS
   ============================ */
[data-testid="stSidebarNav"] ul {
    padding: 0 !important;
    margin: 0 !important;
}

[data-testid="stSidebarNav"] li {
    margin: 2px 0 !important;
}

[data-testid="stSidebarNav"] a {
    padding: 10px 20px !important;
    border-radius: 8px !important;
    margin: 0 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #8BA3C7 !important;
    transition: all 0.2s ease !important;
    text-decoration: none !important;
}

[data-testid="stSidebarNav"] a:hover {
    background: rgba(0, 212, 255, 0.08) !important;
    color: #00D4FF !important;
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(0, 212, 255, 0.12) !important;
    color: #00D4FF !important;
    border-left: 3px solid #00D4FF !important;
}

/* ============================
   METRIC CARDS
   ============================ */
.metric-card {
    background: rgba(13, 31, 60, 0.8);
    border: 1px solid #1A3A5C;
    border-radius: var(--radius-lg);
    padding: 24px 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00D4FF, transparent);
}

.metric-card:hover {
    border-color: #00D4FF;
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0, 212, 255, 0.12);
}

.metric-value {
    font-size: 36px;
    font-weight: 800;
    color: #00D4FF;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}

.metric-label {
    font-size: 11px;
    color: #4A6A8A;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 8px;
    font-weight: 600;
}

.metric-sublabel {
    font-size: 12px;
    color: #8BA3C7;
    margin-top: 4px;
}

/* ============================
   KPI CARDS
   ============================ */
.kpi-card {
    background: rgba(13, 31, 60, 0.9);
    border: 1px solid #1A3A5C;
    border-radius: var(--radius);
    padding: 20px;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}

.kpi-card:hover {
    border-color: rgba(0, 212, 255, 0.4);
    box-shadow: 0 4px 24px rgba(0, 212, 255, 0.08);
}

.kpi-icon {
    font-size: 28px;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #E8F4FD;
}

.kpi-label {
    font-size: 12px;
    color: #4A6A8A;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
    font-weight: 600;
}

/* ============================
   SECTION HEADERS
   ============================ */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 20px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid #1A3A5C;
}

.section-header-line {
    width: 4px;
    height: 24px;
    background: linear-gradient(180deg, #00D4FF, #1E6FFF);
    border-radius: 2px;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #E8F4FD;
    letter-spacing: 0.5px;
}

/* ============================
   STATUS BADGE
   ============================ */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.badge-high   { background: rgba(255,56,96,0.2);  color: #FF3860; border: 1px solid rgba(255,56,96,0.4); }
.badge-medium { background: rgba(255,215,0,0.2);  color: #FFD700; border: 1px solid rgba(255,215,0,0.4); }
.badge-low    { background: rgba(0,255,159,0.2);  color: #00FF9F; border: 1px solid rgba(0,255,159,0.4); }
.badge-active { background: rgba(0,212,255,0.15); color: #00D4FF; border: 1px solid rgba(0,212,255,0.3); }

/* ============================
   BUTTONS
   ============================ */
.stButton > button {
    background: linear-gradient(135deg, #1E6FFF, #0050CC) !important;
    color: #FFFFFF !important;
    border: 1px solid #1E6FFF !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.3px !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #00D4FF, #1E6FFF) !important;
    border-color: #00D4FF !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(30, 111, 255, 0.35) !important;
}

/* Primary CTA */
.stButton[data-testid*="primary"] > button {
    background: linear-gradient(135deg, #00D4FF, #1E6FFF) !important;
}

/* ============================
   SIDEBAR NAV BUTTONS
   ============================ */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: #8BA3C7 !important;
    text-align: left !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0, 212, 255, 0.08) !important;
    color: #00D4FF !important;
    transform: none !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] .stButton > button:focus:not(:active) {
    background: rgba(0, 212, 255, 0.12) !important;
    color: #00D4FF !important;
    border-left: 3px solid #00D4FF !important;
    border-radius: 0 8px 8px 0 !important;
    box-shadow: none !important;
}

/* ============================
   INPUTS
   ============================ */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(10, 22, 40, 0.8) !important;
    border: 1px solid #1A3A5C !important;
    border-radius: 8px !important;
    color: #E8F4FD !important;
    font-family: 'Inter', sans-serif !important;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div:focus-within {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.15) !important;
}

/* ============================
   FILE UPLOADER
   ============================ */
[data-testid="stFileUploader"] {
    background: rgba(10, 22, 40, 0.6) !important;
    border: 1.5px dashed #1A3A5C !important;
    border-radius: var(--radius) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #00D4FF !important;
    background: rgba(0, 212, 255, 0.04) !important;
}

/* ============================
   DATAFRAME / TABLE
   ============================ */
[data-testid="stDataFrame"] {
    border: 1px solid #1A3A5C !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* ============================
   PROGRESS BAR
   ============================ */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #1E6FFF, #00D4FF) !important;
    border-radius: 4px !important;
}

/* ============================
   ALERTS
   ============================ */
.stAlert {
    border-radius: var(--radius) !important;
    border: 1px solid #1A3A5C !important;
    background: rgba(13, 31, 60, 0.8) !important;
}

/* ============================
   EXPANDER
   ============================ */
[data-testid="stExpander"] {
    border: 1px solid #1A3A5C !important;
    border-radius: var(--radius) !important;
    background: rgba(13, 31, 60, 0.5) !important;
}

[data-testid="stExpander"] summary {
    color: #8BA3C7 !important;
    font-weight: 500 !important;
}

/* ============================
   TABS
   ============================ */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #1A3A5C !important;
    gap: 4px !important;
}

[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    border: none !important;
    color: #4A6A8A !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    border-radius: 6px 6px 0 0 !important;
    transition: all 0.2s ease !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: rgba(0, 212, 255, 0.1) !important;
    color: #00D4FF !important;
    border-bottom: 2px solid #00D4FF !important;
}

/* ============================
   SCROLLBAR
   ============================ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #050A14; }
::-webkit-scrollbar-thumb { background: #1A3A5C; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00D4FF; }

/* ============================
   HERO SECTION
   ============================ */
.hero-section {
    text-align: center;
    padding: 80px 40px 60px;
    position: relative;
    overflow: hidden;
}

.hero-tagline {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 4px;
    color: #00D4FF;
    text-transform: uppercase;
    margin-bottom: 16px;
}

.hero-title {
    font-size: clamp(40px, 6vw, 72px);
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1.05;
    background: linear-gradient(135deg, #E8F4FD 0%, #00D4FF 50%, #1E6FFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 12px;
}

.hero-subtitle {
    font-size: 18px;
    color: #8BA3C7;
    max-width: 600px;
    margin: 0 auto 32px;
    line-height: 1.6;
    font-weight: 400;
}

/* ============================
   CHART CARDS
   ============================ */
.chart-card {
    background: rgba(13, 31, 60, 0.8);
    border: 1px solid #1A3A5C;
    border-radius: var(--radius-lg);
    padding: 20px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.chart-card:hover {
    border-color: rgba(0, 212, 255, 0.3);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
}

.chart-title {
    font-size: 13px;
    font-weight: 600;
    color: #8BA3C7;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 16px;
}

/* ============================
   INFO BOX
   ============================ */
.info-box {
    background: rgba(0, 212, 255, 0.06);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin: 12px 0;
}

.warning-box {
    background: rgba(255, 107, 53, 0.06);
    border: 1px solid rgba(255, 107, 53, 0.2);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin: 12px 0;
}

/* ============================
   PIPELINE FLOW
   ============================ */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: rgba(13, 31, 60, 0.6);
    border: 1px solid #1A3A5C;
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #8BA3C7;
    transition: all 0.2s ease;
}

.pipeline-step:hover {
    border-color: #00D4FF;
    color: #E8F4FD;
}

.pipeline-step .step-num {
    background: linear-gradient(135deg, #1E6FFF, #00D4FF);
    color: #050A14;
    font-weight: 700;
    font-size: 11px;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

/* ============================
   SPINNER OVERRIDE
   ============================ */
.stSpinner > div {
    border-top-color: #00D4FF !important;
}

/* ============================
   SELECTBOX LABEL
   ============================ */
.stSelectbox label, .stTextInput label, .stFileUploader label,
.stSlider label, .stNumberInput label, .stCheckbox label {
    color: #8BA3C7 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ============================
   DIVIDER
   ============================ */
hr {
    border-color: #1A3A5C !important;
    margin: 24px 0 !important;
}

/* ============================
   MARKDOWN HEADERS
   ============================ */
h1, h2, h3, h4, h5, h6 {
    color: #E8F4FD !important;
    font-family: 'Inter', sans-serif !important;
}

/* ============================
   CODE BLOCKS
   ============================ */
code, pre {
    background: rgba(10, 22, 40, 0.8) !important;
    border: 1px solid #1A3A5C !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #00D4FF !important;
}

/* ============================
   STATUS INDICATOR
   ============================ */
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}

.status-dot.online  { background: #00FF9F; box-shadow: 0 0 6px #00FF9F; animation: pulse 2s infinite; }
.status-dot.warning { background: #FFD700; box-shadow: 0 0 6px #FFD700; }
.status-dot.offline { background: #FF3860; box-shadow: 0 0 6px #FF3860; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ============================
   TABLE STYLING
   ============================ */
.dataframe { font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }
</style>
"""
