


st.set_page_config(
    page_title="PAPS Care+ Intelligence",
    page_title="PAPS CARE+",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="collapsed",
    initial_sidebar_state="expanded",
)



    :root {
        --bg: #f2f6fb;
        --surface: rgba(255, 255, 255, 0.88);
        --surface: rgba(255, 255, 255, 0.9);
        --surface-strong: #ffffff;
        --stroke: rgba(16, 34, 53, 0.10);
        --text: #102235;
    }

    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"] > div {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.6fr 0.8fr;
        grid-template-columns: 1.55fr 0.85fr;
        gap: 20px;
        align-items: end;
    }
        margin-top: 8px;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.92);
    .student-shell {
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(246,250,255,0.95) 100%);
        border: 1px solid var(--stroke);
        border-radius: 30px;
        padding: 24px;
        box-shadow: var(--shadow);
    }

    .student-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(37,116,234,0.08);
        border: 1px solid rgba(37,116,234,0.12);
        color: #2759b2;
        font-size: 12px;
        font-weight: 700;
    }

    .student-title {
        margin: 14px 0 8px;
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--text);
    }

    .student-copy {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.75;
        margin-bottom: 18px;
    }

    .plan-card {
        background: rgba(255,255,255,0.96);
        border: 1px solid var(--stroke);
        border-radius: 24px;
        padding: 16px 18px;
        box-shadow: 0 16px 36px rgba(15, 39, 64, 0.07);
        padding: 18px;
        box-shadow: 0 14px 30px rgba(15, 39, 64, 0.08);
        height: 100%;
    }

    .plan-week {
        font-size: 12px;
        font-weight: 800;
        color: var(--blue);
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .plan-title {
        margin: 8px 0 12px;
        font-size: 18px;
