import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="PAPS CARE+",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

:root {
    --bg: #f2f6fb;
    --surface: rgba(255, 255, 255, 0.9);
    --surface-strong: #ffffff;
    --stroke: rgba(16, 34, 53, 0.10);
    --text: #102235;
    --muted: #607486;
    --navy: #0f2740;
    --blue: #2574ea;
    --teal: #0ea5a4;
    --amber: #d99a25;
    --red: #d44b57;
    --orange: #ef8b2c;
    --green: #1c9d74;
    --shadow: 0 24px 60px rgba(15, 39, 64, 0.10);
}

    html, body, [class*="css"] {
        font-family: 'Pretendard', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(37,116,234,0.12), transparent 28%),
            radial-gradient(circle at 100% 0%, rgba(14,165,164,0.10), transparent 24%),
            linear-gradient(180deg, #f8fbfe 0%, #edf3f8 100%);
        color: var(--text);
    }

    #MainMenu, header, footer {
        display: none;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10253d 0%, #173756 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f7fbff !important;
    }

    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"] > div {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
    }

    div[data-testid="stTabs"] button {
        border-radius: 999px;
        padding: 10px 18px;
        font-weight: 700;
        color: var(--muted);
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #16324f 0%, #2574ea 100%);
        color: white;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding: 8px 2px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-badge {
        width: 44px;
        height: 44px;
        border-radius: 14px;
        background: linear-gradient(135deg, #14304c 0%, #2677f0 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: 800;
        box-shadow: 0 16px 35px rgba(20, 48, 76, 0.26);
    }

    .brand-copy h1 {
        margin: 0;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    .brand-copy p {
        margin: 4px 0 0;
        color: var(--muted);
        font-size: 13px;
    }

    .status-chip {
        border-radius: 999px;
        padding: 10px 14px;
        background: rgba(37, 116, 234, 0.08);
        border: 1px solid rgba(37, 116, 234, 0.12);
        color: #2759b2;
        font-size: 12px;
        font-weight: 700;
    }

    .brand-mark {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-top: 10px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.14);
        font-size: 12px;
        font-weight: 700;
        color: rgba(255,255,255,0.92);
    }

    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 34px;
        padding: 40px;
        background:
            radial-gradient(circle at 80% 20%, rgba(255,255,255,0.16), transparent 18%),
            linear-gradient(130deg, #0f2740 0%, #153e67 50%, #2680b7 100%);
        color: white;
        box-shadow: 0 30px 80px rgba(15, 39, 64, 0.22);
        margin-bottom: 22px;
    }

    .hero::before {
        content: "";
        position: absolute;
        right: -100px;
        bottom: -100px;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(217,154,37,0.32) 0%, rgba(217,154,37,0.02) 70%);
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.55fr 0.85fr;
        gap: 20px;
        align-items: end;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.16);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .hero h2 {
        margin: 18px 0 12px;
        font-size: 44px;
        line-height: 1.08;
        letter-spacing: -0.04em;
    }

    .hero p {
        margin: 0;
        max-width: 760px;
        color: rgba(255,255,255,0.86);
        font-size: 16px;
        line-height: 1.75;
    }

    .hero-subtitle {
        margin-top: 10px;
        font-size: 18px;
        font-weight: 600;
        color: rgba(255,255,255,0.9);
    }

    .hero-notice {
        margin-top: 18px;
        padding: 16px 18px;
        border-radius: 18px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.14);
        color: rgba(255,255,255,0.94);
        font-size: 14px;
        line-height: 1.7;
    }

    .hero-highlight {
        margin-top: 18px;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
    }

    .hero-stat {
        padding: 16px 18px;
        border-radius: 20px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(8px);
    }

    .hero-stat-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.70);
        font-weight: 700;
    }

    .hero-stat-value {
        margin-top: 6px;
        font-size: 20px;
        font-weight: 800;
        color: white;
    }

    .hero-aside {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 26px;
        padding: 22px;
        backdrop-filter: blur(10px);
    }

    .hero-aside h3 {
        margin: 0 0 12px;
        font-size: 16px;
    }

    .hero-aside ul {
        margin: 0;
        padding-left: 18px;
        color: rgba(255,255,255,0.82);
        line-height: 1.8;
        font-size: 13px;
    }

    .shell {
        background: var(--surface);
        border: 1px solid rgba(255,255,255,0.55);
        border-radius: 30px;
        padding: 22px;
        box-shadow: var(--shadow);
    }

    .shell-dark {
        background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(245,249,253,0.96) 100%);
    }

    .panel-title {
        margin: 0;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    .panel-copy {
        margin: 8px 0 0;
        color: var(--muted);
        font-size: 14px;
        line-height: 1.65;
    }

    .mini-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        font-weight: 800;
        margin-bottom: 8px;
    }

    .summary-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid var(--stroke);
        border-radius: 24px;
        padding: 22px;
        min-height: 152px;
        box-shadow: 0 16px 36px rgba(15, 39, 64, 0.07);
    }

    .summary-value {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: var(--text);
        margin-top: 6px;
    }

    .summary-help {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.65;
        margin-top: 10px;
    }

    .note-card {
        background: #f7fafc;
        border: 1px solid var(--stroke);
        border-radius: 22px;
        padding: 18px;
    }

    .note-card strong {
        color: var(--text);
    }

    .report-card {
        background: rgba(255,255,255,0.94);
        border: 1px solid var(--stroke);
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 18px 40px rgba(15, 39, 64, 0.08);
        height: 100%;
    }

    .report-tag {
        display: inline-flex;
        border-radius: 999px;
        padding: 8px 12px;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 14px;
    }

    .tag-red { background: rgba(212,75,87,0.14); color: #b22d3c; }
    .tag-orange { background: rgba(239,139,44,0.16); color: #b96215; }
    .tag-green { background: rgba(28,157,116,0.14); color: #0f7658; }
    .tag-blue { background: rgba(37,116,234,0.14); color: #1f56ba; }

    .report-card h4 {
        margin: 0 0 8px;
        font-size: 18px;
        font-weight: 800;
    }

    .report-stat {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.7;
        margin-bottom: 14px;
    }

    .report-section {
        margin-top: 14px;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text);
    }

    .report-copy {
        color: #314556;
        font-size: 14px;
        line-height: 1.8;
        margin-top: 8px;
    }

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
        font-weight: 800;
        color: var(--text);
    }

.plan-item {
    margin-top: 8px;
    color: #314556;
    font-size: 14px;
    line-height: 1.7;
}

.phone-stage {
    display: grid;
    grid-template-columns: 1fr 80px 1fr;
    gap: 18px;
    align-items: center;
    margin-top: 22px;
}

.phone-arrow {
    text-align: center;
    font-size: 54px;
    font-weight: 800;
    color: rgba(217,154,37,0.65);
}

.phone-shell {
    max-width: 380px;
    margin: 0 auto;
    background: linear-gradient(180deg, #111111 0%, #1d1d1d 100%);
    border-radius: 42px;
    padding: 14px;
    box-shadow: 0 30px 70px rgba(15, 39, 64, 0.20);
}

.phone-screen {
    min-height: 760px;
    border-radius: 30px;
    overflow: hidden;
    background:
        radial-gradient(circle at top left, rgba(255,211,127,0.35), transparent 32%),
        linear-gradient(180deg, #fff9ef 0%, #fffdf8 100%);
    padding: 18px 18px 22px;
    position: relative;
}

.phone-dynamic {
    width: 118px;
    height: 24px;
    border-radius: 999px;
    background: #111111;
    margin: 0 auto 12px;
}

.phone-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13px;
    color: #23394f;
    font-weight: 700;
    margin-bottom: 10px;
}

.phone-title {
    text-align: center;
    font-size: 24px;
    font-weight: 800;
    color: #1a2d41;
    margin: 8px 0 18px;
}

.segmented {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    padding: 6px;
    border-radius: 16px;
    background: rgba(35,57,79,0.08);
    margin-bottom: 18px;
}

.segmented span {
    text-align: center;
    padding: 10px 12px;
    border-radius: 12px;
    color: #5f7284;
    font-size: 13px;
    font-weight: 700;
}

.segmented .active {
    background: white;
    color: #1c3147;
    box-shadow: 0 10px 20px rgba(15,39,64,0.08);
}

.diagnosis-layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    align-items: center;
}

.gauge-wrap {
    position: relative;
    width: 220px;
    height: 220px;
    margin: 0 auto;
}

.gauge-ring {
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: conic-gradient(#f0b24c 0deg, #e98f42 150deg, #d9dde3 150deg, #d9dde3 300deg, rgba(0,0,0,0) 300deg);
    position: relative;
    box-shadow: inset 0 12px 28px rgba(255,255,255,0.5);
}

.gauge-ring::before {
    content: "";
    position: absolute;
    inset: 22px;
    border-radius: 50%;
    background: linear-gradient(180deg, #fffefc 0%, #ffffff 100%);
}

.gauge-needle {
    position: absolute;
    width: 6px;
    height: 92px;
    background: #31343a;
    left: 50%;
    bottom: 50%;
    transform-origin: bottom center;
    transform: translateX(-50%) rotate(32deg);
    border-radius: 999px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}

.gauge-needle::after {
    content: "";
    position: absolute;
    bottom: -12px;
    left: 50%;
    transform: translateX(-50%);
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #2e343b;
}

.gauge-text {
    position: absolute;
    left: 50%;
    bottom: 24px;
    transform: translateX(-50%);
    text-align: center;
    color: #172a3d;
}

.gauge-text strong {
    display: block;
    font-size: 34px;
    font-weight: 800;
}

.gauge-text span {
    font-size: 14px;
    color: #66788a;
    font-weight: 700;
}

.avatar-card {
    text-align: center;
}

.avatar-circle {
    width: 180px;
    height: 180px;
    margin: 0 auto 12px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 25%, #fff5df 0%, #f6e2bf 34%, #f2cf8b 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 86px;
}

.avatar-meta {
    text-align: left;
    display: grid;
    gap: 10px;
    margin-top: 8px;
}

.avatar-meta div {
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(17,34,53,0.08);
    border-radius: 14px;
    padding: 12px 14px;
}

.avatar-meta b {
    display: block;
    font-size: 12px;
    color: #7b8b9b;
    margin-bottom: 4px;
}

.avatar-meta strong {
    color: #192e43;
    font-size: 17px;
}

.bottom-nav {
    position: absolute;
    left: 18px;
    right: 18px;
    bottom: 16px;
    background: rgba(255,255,255,0.94);
    border: 1px solid rgba(17,34,53,0.08);
    border-radius: 22px;
    padding: 12px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #687b8e;
    font-size: 22px;
}

.fab {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, #f2ba62 0%, #ea8a3d 100%);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 30px;
    box-shadow: 0 16px 28px rgba(233,138,61,0.28);
}

.prescription-card {
    margin-top: 18px;
    background: linear-gradient(180deg, #eca14e 0%, #e48134 100%);
    border-radius: 28px;
    padding: 22px;
    color: white;
    box-shadow: 0 22px 36px rgba(226,132,52,0.28);
}

.prescription-card small {
    display: inline-flex;
    align-items: center;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.18);
    font-size: 12px;
    font-weight: 800;
}

.prescription-card h4 {
    margin: 14px 0 10px;
    font-size: 34px;
    line-height: 1.25;
    letter-spacing: -0.03em;
}

.prescription-card p {
    margin: 0;
    font-size: 16px;
    color: rgba(255,255,255,0.92);
    line-height: 1.7;
}

.runner-stage {
    margin-top: 22px;
    min-height: 220px;
    border-radius: 22px;
    background:
        linear-gradient(180deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%),
        linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(0,0,0,0.06) 100%);
    position: relative;
    overflow: hidden;
}

.runner-stage::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px);
    background-size: 26px 26px;
    opacity: 0.35;
}

.pulse-line {
    position: absolute;
    left: 18px;
    right: 18px;
    top: 50%;
    height: 3px;
    background: rgba(255,240,210,0.45);
}

.pulse-line::before {
    content: "";
    position: absolute;
    left: 6%;
    right: 6%;
    top: -42px;
    height: 90px;
    background: linear-gradient(90deg,
        transparent 0%,
        transparent 12%,
        #ffe6b0 12%,
        #ffe6b0 16%,
        transparent 16%,
        transparent 30%,
        #ffe6b0 30%,
        #ffe6b0 33%,
        transparent 33%,
        transparent 46%,
        #ffe6b0 46%,
        #ffe6b0 49%,
        transparent 49%,
        transparent 62%,
        #ffe6b0 62%,
        #ffe6b0 65%,
        transparent 65%,
        transparent 100%);
    clip-path: polygon(0 55%, 10% 55%, 14% 35%, 18% 75%, 24% 55%, 34% 55%, 38% 43%, 42% 63%, 49% 55%, 58% 55%, 62% 28%, 68% 78%, 74% 55%, 100% 55%, 100% 60%, 0 60%);
}

.runner-emoji {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -40%);
    font-size: 88px;
}

.fitt-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 20px;
}

.fitt-mini {
    background: rgba(255,255,255,0.14);
    border-radius: 16px;
    padding: 12px 10px;
    text-align: center;
}

.fitt-mini b {
    display: block;
    font-size: 12px;
    color: rgba(255,255,255,0.82);
    margin-bottom: 6px;
}

.fitt-mini span {
    font-size: 14px;
    font-weight: 800;
    color: white;
}

.ops-scene {
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(16,34,53,0.08);
    border-radius: 34px;
    padding: 18px;
    box-shadow: 0 28px 56px rgba(15,39,64,0.10);
    margin-top: 18px;
}

.ops-shell {
    display: grid;
    grid-template-columns: 210px 1fr;
    min-height: 700px;
    border-radius: 28px;
    overflow: hidden;
    background: linear-gradient(180deg, #fffdf9 0%, #ffffff 100%);
    border: 1px solid rgba(16,34,53,0.08);
}

.ops-nav {
    background: linear-gradient(180deg, #fbfbfb 0%, #f5f7fa 100%);
    border-right: 1px solid rgba(16,34,53,0.08);
    padding: 26px 18px;
}

.ops-nav-title {
    color: #90a0af;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 18px 0 10px;
}

.ops-nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 12px;
    border-radius: 14px;
    color: #53677b;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 4px;
}

.ops-nav-item.active {
    background: linear-gradient(135deg, rgba(91,114,255,0.12) 0%, rgba(233,143,66,0.10) 100%);
    color: #21354a;
    box-shadow: inset 0 0 0 1px rgba(91,114,255,0.08);
}

.ops-main {
    padding: 18px 20px 20px;
    background:
        radial-gradient(circle at top left, rgba(255,210,120,0.18), transparent 28%),
        linear-gradient(180deg, #fffdf9 0%, #ffffff 100%);
}

.ops-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    padding: 8px 6px 16px;
    border-bottom: 1px solid rgba(16,34,53,0.07);
}

.ops-search {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 340px;
    padding: 10px 14px;
    border-radius: 14px;
    background: rgba(16,34,53,0.04);
    color: #6c7f92;
    font-size: 13px;
    font-weight: 700;
}

.ops-icons {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #5e7388;
}

.ops-icon {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: rgba(16,34,53,0.05);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
}

.ops-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #f5c86e 0%, #eb8d48 100%);
}

.ops-content-title {
    font-size: 34px;
    font-weight: 800;
    color: #172b3f;
    margin: 22px 0 16px;
    letter-spacing: -0.03em;
}

.ops-panel {
    border: 1px solid rgba(16,34,53,0.08);
    border-radius: 24px;
    background: white;
    padding: 16px;
    box-shadow: 0 18px 40px rgba(15,39,64,0.06);
}

.ops-tabs {
    display: flex;
    gap: 20px;
    padding: 6px 4px 14px;
    border-bottom: 1px solid rgba(16,34,53,0.06);
}

.ops-tabs span {
    color: #7a8b9b;
    font-size: 14px;
    font-weight: 800;
}

.ops-tabs .active {
    color: #202f42;
}

.ops-filterbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding: 14px 0;
}

.ops-leftfilters {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.ops-chip {
    padding: 10px 14px;
    border-radius: 12px;
    background: #f7f8fb;
    border: 1px solid rgba(16,34,53,0.08);
    color: #526678;
    font-size: 13px;
    font-weight: 700;
}

.ops-actions {
    display: flex;
    gap: 10px;
}

.ops-map {
    position: relative;
    min-height: 510px;
    overflow: hidden;
    border-radius: 22px;
    border: 1px solid rgba(16,34,53,0.08);
    background:
        linear-gradient(0deg, rgba(237,240,244,0.85) 1px, transparent 1px),
        linear-gradient(90deg, rgba(237,240,244,0.85) 1px, transparent 1px),
        radial-gradient(circle at 18% 28%, rgba(200,210,220,0.35), transparent 18%),
        linear-gradient(180deg, #fcfcfb 0%, #f7f8fa 100%);
    background-size: 42px 42px, 42px 42px, auto, auto;
}

.ops-map::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 16% 32%, rgba(160,170,180,0.18) 0 3px, transparent 4px),
        radial-gradient(circle at 74% 14%, rgba(160,170,180,0.14) 0 2px, transparent 3px),
        radial-gradient(circle at 84% 74%, rgba(160,170,180,0.14) 0 2px, transparent 3px),
        linear-gradient(120deg, transparent 0 24%, rgba(195,203,211,0.35) 24% 26%, transparent 26% 100%),
        linear-gradient(20deg, transparent 0 56%, rgba(195,203,211,0.35) 56% 58%, transparent 58% 100%);
    opacity: 0.8;
}

.ops-region {
    position: absolute;
    inset: 34px 44px 26px 54px;
    background: #262f73;
    opacity: 0.95;
    clip-path: polygon(17% 24%, 31% 8%, 55% 10%, 73% 2%, 92% 22%, 88% 43%, 96% 56%, 82% 86%, 58% 93%, 41% 86%, 17% 92%, 2% 66%, 8% 48%, 3% 33%);
    box-shadow: inset 0 0 0 2px rgba(255,255,255,0.14), 0 18px 30px rgba(26,38,94,0.18);
}

.ops-river {
    position: absolute;
    left: 11%;
    right: 8%;
    top: 44%;
    height: 28px;
    background: rgba(223,227,233,0.9);
    border-radius: 999px;
    transform: rotate(8deg);
    box-shadow: 0 0 0 6px rgba(255,255,255,0.18);
}

.heat {
    position: absolute;
    border-radius: 50%;
    filter: blur(18px);
    background: radial-gradient(circle, rgba(246,157,84,0.95) 0%, rgba(238,129,51,0.75) 38%, rgba(241,147,71,0.30) 72%, transparent 100%);
}

.heat.one { width: 190px; height: 150px; left: 46%; top: 18%; }
.heat.two { width: 220px; height: 170px; left: 38%; top: 38%; }
.heat.three { width: 180px; height: 135px; left: 58%; top: 56%; }

.ops-callout {
    position: absolute;
    right: 110px;
    top: 148px;
    background: white;
    border-radius: 20px;
    padding: 18px 18px 14px;
    box-shadow: 0 18px 36px rgba(15,39,64,0.16);
    border: 1px solid rgba(16,34,53,0.08);
    min-width: 220px;
}

.ops-callout::after {
    content: "";
    position: absolute;
    left: 24px;
    bottom: -14px;
    border-width: 14px 12px 0 12px;
    border-style: solid;
    border-color: white transparent transparent transparent;
}

.ops-callout b {
    display: block;
    color: #20354b;
    font-size: 17px;
    margin-bottom: 12px;
}

.ops-scale {
    height: 14px;
    border-radius: 999px;
    background: linear-gradient(90deg, #f0bd76 0%, #f09b5b 40%, #3c438a 100%);
}

.ops-footnote {
    margin-top: 14px;
    color: #73869a;
    font-size: 13px;
    line-height: 1.7;
}

@media (max-width: 1100px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }
    .hero-highlight {
        grid-template-columns: 1fr;
    }
    .phone-stage {
        grid-template-columns: 1fr;
    }
    .phone-arrow {
        transform: rotate(90deg);
        font-size: 42px;
    }
    .ops-shell {
        grid-template-columns: 1fr;
    }
    .ops-nav {
        display: none;
    }
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_data
def load_raw_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    candidates = [
        os.path.join(data_dir, "PAPS_Final_Master (5).xlsx"),
        os.path.join(data_dir, "PAPS_Combined_Data.xlsx"),
    ]
    file_path = next((path for path in candidates if os.path.exists(path)), None)

    if file_path is None:
        searched = ", ".join(candidates)
        return None, {}, f"데이터 파일을 찾을 수 없습니다. 확인 경로: {searched}"

    try:
        df = pd.read_excel(file_path)
    except Exception as exc:
        return None, {}, f"엑셀 파일을 읽는 중 오류가 발생했습니다: {exc}"

    df.columns = df.columns.map(lambda col: str(col).strip())

    def find_col(keywords):
        for column in df.columns:
            name = str(column)
            if any(keyword in name for keyword in keywords):
                return column
        return None

    target_map = {
        "BMI": find_col(["BMI", "비만", "체질량"]),
        "심폐지구력": find_col(["왕복", "오래달리기", "심폐", "셔틀런"]),
        "근력/근지구력": find_col(["악력", "팔굽혀", "말아올리기"]),
        "유연성": find_col(["앉아윗몸", "유연성"]),
        "순발력": find_col(["제자리멀리", "순발력"]),
    }
    valid_targets = {name: column for name, column in target_map.items() if column}

    if len(valid_targets) < 2:
        return None, {}, "군집 분석을 위해서는 최소 2개의 측정 지표가 필요합니다."

    for column in valid_targets.values():
        cleaned = df[column].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        df[column] = pd.to_numeric(cleaned, errors="coerce")

    def find_first(keywords, default_series):
        found = find_col(keywords)
        return df[found].astype(str).str.strip() if found else default_series

    school_fallback = df.iloc[:, 0].astype(str).str.strip()
    df["순수학교명"] = find_first(["추출학교명", "학교명"], school_fallback)

    year_col = find_col(["연도"])
    df["연도"] = (
        pd.to_numeric(df[year_col], errors="coerce").fillna(0).astype(int)
        if year_col
        else 0
    )
    df["시군"] = find_first(["시군"], pd.Series(["미상"] * len(df), index=df.index))
    df["성별"] = find_first(["성별", "남여"], pd.Series(["전체"] * len(df), index=df.index))
    df["학년"] = find_first(["학년"], pd.Series(["전체"] * len(df), index=df.index))
    df = df.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    return df, {"valid": valid_targets, "file_path": file_path}, None


def apply_filters(df, years, regions, grades, genders, schools):
    filtered_df = df.copy()
    if years:
        filtered_df = filtered_df[filtered_df["연도"].isin(years)]
    if regions:
        filtered_df = filtered_df[filtered_df["시군"].isin(regions)]
    if grades:
        filtered_df = filtered_df[filtered_df["학년"].isin(grades)]
    if genders:
        filtered_df = filtered_df[filtered_df["성별"].isin(genders)]
    if schools:
        filtered_df = filtered_df[filtered_df["순수학교명"].isin(schools)]
    return filtered_df


def build_cluster_labels(cluster_summary, x_label):
    ordered = cluster_summary.sort_values("score", ascending=(x_label == "BMI")).index.tolist()
    label_sets = {
        2: ["관리 필요군", "건강 양호군"],
        3: ["고위험군", "일반군", "우수군"],
        4: ["고위험군", "중점관리군", "일반군", "우수군"],
    }
    names = label_sets[len(ordered)]
    return {cluster_id: names[index] for index, cluster_id in enumerate(ordered)}


def get_group_style(label):
    if "고위험" in label or "관리 필요" in label:
        return "tag-red"
    if "중점관리" in label:
        return "tag-orange"
    if "일반" in label:
        return "tag-green"
    return "tag-blue"


def get_prescription_content(label):
    if "고위험" in label or "관리 필요" in label:
        return (
            "기초 체력 회복 중심",
            "저강도 유산소와 기초 근력 루틴으로 활동량을 안정적으로 회복하고 생활 속 움직임을 늘리는 방향이 적합합니다.",
            "집중 지원 프로그램",
            "건강체력교실, 영양 상담, 가정 연계형 생활습관 피드백을 함께 운영하는 구성이 효과적입니다.",
        )
    if "중점관리" in label:
        return (
            "참여도 강화형 성장",
            "뉴스포츠와 순환운동을 활용해 흥미를 유지하면서 심폐지구력과 근지구력을 단계적으로 끌어올립니다.",
            "방과 후 성장 프로그램",
            "팀 스포츠 기반 참여형 프로그램과 주간 목표 피드백을 결합해 운동 지속성을 높입니다.",
        )
    if "일반" in label:
        return (
            "균형 유지형 관리",
            "근력, 유연성, 지구력의 밸런스를 유지할 수 있도록 주간 루틴과 회복 스트레칭을 함께 운영합니다.",
            "자율 습관 프로그램",
            "1인 1운동, 기록 관리, 선택형 종목 체험을 통해 생활체육 습관을 안정적으로 정착시킵니다.",
        )
    return (
        "심화 성장형 관리",
        "인터벌 트레이닝과 종목 특화 루틴을 통해 상위 체력군의 강점을 유지하고 한 단계 더 발전시키는 전략입니다.",
        "리더십 연계 프로그램",
        "학생 스포츠 리더, 멘토링, 지역 연계 심화 프로그램으로 동기와 역할을 확장할 수 있습니다.",
    )


def format_selection(values):
    if not values:
        return "전체"
    values = [str(value) for value in values]
    return ", ".join(values[:2]) + (f" 외 {len(values) - 2}개" if len(values) > 2 else "")


def build_student_plan(label):
    plans = {
        "고위험군": [
            {"week": "1주차", "title": "적응 시작", "freq": "주 3회", "intensity": "매우 가볍게", "time": "20분", "type": "빠르게 걷기 + 전신 스트레칭"},
            {"week": "2주차", "title": "기초 체력 회복", "freq": "주 3회", "intensity": "가볍게", "time": "25분", "type": "걷기 + 스쿼트 2세트 + 코어 5분"},
            {"week": "3주차", "title": "지구력 확보", "freq": "주 4회", "intensity": "가볍게", "time": "30분", "type": "인터벌 걷기/가벼운 조깅 + 하체 근력"},
            {"week": "4주차", "title": "습관 정착", "freq": "주 4회", "intensity": "보통 이하", "time": "35분", "type": "걷기·조깅 혼합 + 복합 맨몸운동"},
        ],
        "중점관리군": [
            {"week": "1주차", "title": "활동량 늘리기", "freq": "주 3회", "intensity": "가볍게", "time": "25분", "type": "걷기/조깅 혼합 + 전신 스트레칭"},
            {"week": "2주차", "title": "심폐 적응", "freq": "주 4회", "intensity": "보통 이하", "time": "30분", "type": "셔틀런 리듬훈련 + 하체 근력"},
            {"week": "3주차", "title": "근지구력 강화", "freq": "주 4회", "intensity": "보통", "time": "35분", "type": "순환운동 + 코어 + 점핑드릴"},
            {"week": "4주차", "title": "지속성 확보", "freq": "주 4회", "intensity": "보통", "time": "40분", "type": "조깅 + 서킷트레이닝 + 유연성 루틴"},
        ],
        "일반군": [
            {"week": "1주차", "title": "균형 유지", "freq": "주 4회", "intensity": "보통", "time": "30분", "type": "조깅 + 근력 2종 + 스트레칭"},
            {"week": "2주차", "title": "전신 밸런스", "freq": "주 4회", "intensity": "보통", "time": "35분", "type": "인터벌 달리기 + 맨몸 서킷"},
            {"week": "3주차", "title": "기록 향상", "freq": "주 4회", "intensity": "보통 이상", "time": "40분", "type": "셔틀런 훈련 + 코어 강화"},
            {"week": "4주차", "title": "자기주도 운동", "freq": "주 5회", "intensity": "보통 이상", "time": "40분", "type": "조깅 + 민첩성 드릴 + 회복 스트레칭"},
        ],
        "우수군": [
            {"week": "1주차", "title": "상위권 유지", "freq": "주 4회", "intensity": "보통 이상", "time": "40분", "type": "인터벌 러닝 + 코어 + 하체 강화"},
            {"week": "2주차", "title": "심화 훈련", "freq": "주 5회", "intensity": "높게", "time": "45분", "type": "셔틀런 고강도 세션 + 서킷"},
            {"week": "3주차", "title": "경쟁력 강화", "freq": "주 5회", "intensity": "높게", "time": "45분", "type": "민첩성·순발력 드릴 + 근지구력"},
            {"week": "4주차", "title": "리더십 단계", "freq": "주 5회", "intensity": "높게", "time": "50분", "type": "인터벌 + 기술 훈련 + 자기기록 관리"},
        ],
    }
    return plans.get(label, plans["일반군"])


def classify_student_profile(height_cm, weight_kg, shuttle_runs):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    allometric_index = shuttle_runs / (weight_kg ** 0.33)

    centroids = pd.DataFrame(
        [
            {"label": "고위험군", "bmi": 29.0, "allometric": 4.6},
            {"label": "중점관리군", "bmi": 25.0, "allometric": 5.8},
            {"label": "일반군", "bmi": 22.0, "allometric": 7.0},
            {"label": "우수군", "bmi": 19.5, "allometric": 8.4},
        ]
    )
    distances = (
        (centroids["bmi"] - bmi) ** 2 + ((centroids["allometric"] - allometric_index) * 1.2) ** 2
    ) ** 0.5
    matched = centroids.loc[distances.idxmin()]

    summaries = {
        "고위험군": "체중 관리와 심폐지구력 회복을 먼저 잡아야 하는 단계입니다.",
        "중점관리군": "활동량은 조금 더 늘리고, 규칙적인 루틴으로 체력 반등이 가능한 단계입니다.",
        "일반군": "전반적인 균형은 양호하며 기록 향상을 위한 루틴이 잘 맞습니다.",
        "우수군": "상위 체력군으로 분류되며, 심화 훈련과 리더십 활동까지 확장할 수 있습니다.",
    }

    return {
        "bmi": bmi,
        "allometric_index": allometric_index,
        "cluster_label": matched["label"],
        "summary": summaries[matched["label"]],
        "plan": build_student_plan(matched["label"]),
    }


def compute_student_grade(cluster_label):
    grade_map = {
        "우수군": 1,
        "일반군": 2,
        "중점관리군": 3,
        "관리 필요군": 4,
        "고위험군": 5,
    }
    return grade_map.get(cluster_label, 3)


def render_admin_view(df, meta):
    st.markdown(
        """
        <div class="hero">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">PAPS CARE+ ANALYTICS</div>
                    <div class="brand-mark">PAPS CARE+ 맞춤형 체력 관리 시스템</div>
                    <h2>PAPS CARE+</h2>
                    <div class="hero-subtitle">강원특별자치도 학교 데이터 AI 분석 시스템</div>
                    <div class="hero-notice">* 본 시스템은 <b>학교알리미</b> 공시 데이터를 기반으로 학생들의 건강체력평가(PAPS)를 AI로 분석한 결과를 제공합니다.</div>
                    <p style="margin-top:18px;">
                        학교별 체력 현황을 단순 나열이 아니라 분석 가능한 정보로 전환합니다.
                        위험군 비중, 집단별 분포, 맞춤형 처방 방향을 한 화면에서 보고서처럼 확인할 수 있습니다.
                    </p>
                    <div class="hero-highlight">
                        <div class="hero-stat">
                            <div class="hero-stat-label">Core Value</div>
                            <div class="hero-stat-value">AI 군집 분석</div>
                        </div>
                        <div class="hero-stat">
                            <div class="hero-stat-label">Output</div>
                            <div class="hero-stat-value">맞춤형 처방 리포트</div>
                        </div>
                        <div class="hero-stat">
                            <div class="hero-stat-label">View</div>
                            <div class="hero-stat-value">기관형 시각 대시보드</div>
                        </div>
                    </div>
                </div>
                <div class="hero-aside">
                    <h3>PAPS CARE+ 제공 내용</h3>
                    <ul>
                        <li>학교 체력 데이터의 분포와 위험 신호를 직관적으로 확인</li>
                        <li>두 개 지표 조합 기준의 AI 군집 분석 결과 제공</li>
                        <li>집단별 운동 처방과 교육 프로그램 추천 제시</li>
                        <li>보고용 화면에 맞춘 요약 카드와 시각 리포트 구성</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## 데이터 필터")
        st.caption("선택한 조건을 기준으로 차트와 리포트가 다시 계산됩니다.")

        s_year = st.multiselect("연도", sorted(df["연도"].dropna().unique()))
        s_region = st.multiselect("시·군", sorted(df["시군"].dropna().unique()))
        s_grade = st.multiselect("학년", sorted(df["학년"].dropna().unique()))
        s_gender = st.multiselect("성별", sorted(df["성별"].dropna().unique()))

        school_base_df = apply_filters(df, s_year, s_region, s_grade, s_gender, [])
        school_options = sorted(school_base_df["순수학교명"].dropna().unique())
        s_school = st.multiselect("학교", school_options)

        st.markdown("---")
        st.markdown("## 분석 설정")
        metric_options = list(meta["valid"].keys())
        x_ax = st.selectbox("수평축", metric_options, index=0)
        y_ax = st.selectbox("수직축", metric_options, index=1 if len(metric_options) > 1 else 0)
        n_cl = st.slider("군집 수", 2, 4, 3)

    st.markdown('<div class="shell shell-dark">', unsafe_allow_html=True)
    st.markdown('<h3 class="panel-title">현재 분석 기준</h3>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="panel-copy">연도 {format_selection(s_year)} · 지역 {format_selection(s_region)} · 학년 {format_selection(s_grade)} · 성별 {format_selection(s_gender)} · 학교 {format_selection(s_school)}</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="ops-scene">
            <div class="ops-shell">
                <div class="ops-nav">
                    <div class="ops-nav-title">내비게이션</div>
                    <div class="ops-nav-item">◫ 체력 현황</div>
                    <div class="ops-nav-item">◫ 지역 리포트</div>
                    <div class="ops-nav-item active">◫ 취약 영역 진단</div>
                    <div class="ops-nav-item">◫ 세부 지역</div>
                    <div class="ops-nav-item">◫ 상위 학교</div>
                    <div class="ops-nav-title">분석</div>
                    <div class="ops-nav-item">◫ 운동 처방맵</div>
                    <div class="ops-nav-item">◫ 상담 히스토리</div>
                    <div class="ops-nav-title">설정</div>
                    <div class="ops-nav-item">◫ 계정 정보</div>
                    <div class="ops-nav-item">◫ 리포트 내보내기</div>
                </div>
                <div class="ops-main">
                    <div class="ops-toolbar">
                        <div class="ops-search">☰ &nbsp; 교육행정 &nbsp;›&nbsp; 취약 체력권 진단</div>
                        <div class="ops-icons">
                            <div class="ops-icon">⚙</div>
                            <div class="ops-icon">🔔</div>
                            <div class="ops-avatar"></div>
                        </div>
                    </div>
                    <div class="ops-content-title">체육행정</div>
                    <div class="ops-panel">
                        <div class="ops-tabs">
                            <span class="active">취약 진단</span>
                            <span>체력 권장</span>
                        </div>
                        <div class="ops-filterbar">
                            <div class="ops-leftfilters">
                                <div class="ops-chip">취약 체력권</div>
                                <div class="ops-chip">{format_selection(s_region)} 생활권</div>
                                <div class="ops-chip">{format_selection(s_year)}년</div>
                                <div class="ops-chip">{x_ax} / {y_ax}</div>
                            </div>
                            <div class="ops-actions">
                                <div class="ops-chip">리포트 출력</div>
                                <div class="ops-chip">격자 보기</div>
                            </div>
                        </div>
                        <div class="ops-map">
                            <div class="ops-region"></div>
                            <div class="ops-river"></div>
                            <div class="heat one"></div>
                            <div class="heat two"></div>
                            <div class="heat three"></div>
                            <div class="ops-callout">
                                <b>취약 체력 종목</b>
                                <div class="ops-scale"></div>
                            </div>
                        </div>
                        <div class="ops-footnote">
                            선택된 필터 기준으로 취약 체력군이 집중된 생활권을 강조해 보여주는 관리자 시뮬레이션 화면입니다.
                            심사 발표에서는 “어느 지역에 어떤 체력 이슈가 몰려 있는지”를 직관적으로 설명하는 용도로 활용할 수 있습니다.
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filtered_df = apply_filters(df, s_year, s_region, s_grade, s_gender, s_school)
    if filtered_df.empty:
        st.warning("선택한 조건에 맞는 데이터가 없습니다. 필터를 조정해 주세요.")
        return

    group_cols = ["순수학교명", "연도", "시군", "학년", "성별"]
    agg_map = {column: "mean" for column in meta["valid"].values()}
    df_agg = filtered_df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()

    raw_x = meta["valid"][x_ax]
    raw_y = meta["valid"][y_ax]
    cluster_source = df_agg.dropna(subset=[raw_x, raw_y]).copy()

    if len(cluster_source) < n_cl:
        st.warning(f"현재 조건에서는 군집 {n_cl}개를 만들 데이터가 부족합니다. 필터를 조금 넓혀 주세요.")
        return

    scaled_points = StandardScaler().fit_transform(cluster_source[[raw_x, raw_y]])
    kmeans = KMeans(n_clusters=n_cl, random_state=42, n_init=10)
    cluster_source["Cluster"] = kmeans.fit_predict(scaled_points)
    cluster_summary = cluster_source.groupby("Cluster")[[raw_x, raw_y]].mean()
    cluster_summary["score"] = cluster_summary.mean(axis=1)
    cluster_labels = build_cluster_labels(cluster_summary, x_ax)
    cluster_source["유형"] = cluster_source["Cluster"].map(cluster_labels)

    school_count = int(cluster_source["순수학교명"].nunique())
    region_count = int(cluster_source["시군"].nunique())
    dominant_group = cluster_source["유형"].value_counts().idxmax()
    dominant_share = round((cluster_source["유형"].value_counts().max() / len(cluster_source)) * 100, 1)

    sub_tabs = st.tabs(["종합 현황", "군집 분포 맵", "맞춤형 처방"])

    with sub_tabs[0]:
        st.markdown("### 종합 현황")
        st.markdown("현재 선택한 분석 조건을 기준으로 핵심 수치와 집단 분포를 먼저 확인합니다.")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="mini-label">Schools</div>
                    <div class="summary-value">{school_count}</div>
                    <div class="summary-help">현재 분석 범위에 포함된 학교 수입니다.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi2:
            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="mini-label">Regions</div>
                    <div class="summary-value">{region_count}</div>
                    <div class="summary-help">현재 선택된 시·군 범위입니다.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi3:
            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="mini-label">Top Group</div>
                    <div class="summary-value" style="font-size:30px;">{dominant_group}</div>
                    <div class="summary-help">가장 높은 비중을 차지하는 분석 집단입니다.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi4:
            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="mini-label">Share</div>
                    <div class="summary-value">{dominant_share}%</div>
                    <div class="summary-help">최대 비중 집단의 구성 비율입니다.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        left, right = st.columns([1.35, 1])
        with left:
            st.markdown('<div class="shell">', unsafe_allow_html=True)
            st.markdown('<h3 class="panel-title">집단 분포 브리프</h3>', unsafe_allow_html=True)
            st.markdown(
                '<p class="panel-copy">현재 조건 안에서 어떤 집단이 얼마나 많이 분포하는지 빠르게 읽을 수 있습니다.</p>',
                unsafe_allow_html=True,
            )
            share_df = (
                cluster_source["유형"]
                .value_counts(normalize=True)
                .mul(100)
                .round(1)
                .rename_axis("유형")
                .reset_index(name="비중")
            )
            bar_fig = px.bar(
                share_df,
                x="비중",
                y="유형",
                orientation="h",
                color="유형",
                text="비중",
                color_discrete_map={
                    "관리 필요군": "#d44b57",
                    "고위험군": "#d44b57",
                    "중점관리군": "#ef8b2c",
                    "일반군": "#1c9d74",
                    "우수군": "#2574ea",
                    "건강 양호군": "#2574ea",
                },
            )
            bar_fig.update_traces(texttemplate="%{text}%", textposition="outside")
            bar_fig.update_layout(
                height=360,
                margin=dict(t=10, b=10, l=10, r=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor="rgba(16,34,53,0.08)", zeroline=False),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(bar_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="shell">', unsafe_allow_html=True)
            st.markdown('<h3 class="panel-title">분석 해석 메모</h3>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="note-card">
                    <strong>현재 분석 조합</strong><br>
                    {x_ax}와 {y_ax}를 기준으로 {n_cl}개 군집을 생성했습니다.
                </div>
                <br>
                <div class="note-card">
                    <strong>해석 기준</strong><br>
                    필터된 데이터만 사용해 군집을 다시 계산하므로, 현재 화면은 전체 평균이 아니라 선택된 집단의 상대 비교 결과입니다.
                </div>
                <br>
                <div class="note-card">
                    <strong>주의</strong><br>
                    BMI는 높은 값이 항상 좋은 것으로 읽히지 않도록 별도 방향성을 적용했습니다.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### AI 군집 산포도")
        st.markdown("첫 화면에서 바로 학교별 분포와 군집 위치를 확인할 수 있도록 산포도를 함께 배치했습니다.")

        st.markdown('<div class="shell">', unsafe_allow_html=True)
        overview_fig = px.scatter(
            cluster_source,
            x=raw_x,
            y=raw_y,
            color="유형",
            text="순수학교명",
            hover_data={"연도": True, "시군": True, "학년": True, "성별": True},
            labels={raw_x: x_ax, raw_y: y_ax, "유형": "집단"},
            color_discrete_map={
                "관리 필요군": "#d44b57",
                "고위험군": "#d44b57",
                "중점관리군": "#ef8b2c",
                "일반군": "#1c9d74",
                "우수군": "#2574ea",
                "건강 양호군": "#2574ea",
            },
        )
        overview_fig.update_traces(
            marker=dict(size=16, opacity=0.88, line=dict(width=1.1, color="white")),
            textposition="top center",
            textfont=dict(size=10, color="#254258"),
        )
        overview_fig.update_layout(
            height=560,
            margin=dict(t=10, b=10, l=10, r=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.72)",
            ),
            xaxis=dict(showgrid=True, gridcolor="rgba(16,34,53,0.08)", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(16,34,53,0.08)", zeroline=False),
        )
        st.plotly_chart(overview_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with sub_tabs[1]:
        st.markdown("### 군집 분포 맵")
        st.markdown("학교별 위치와 집단 분포를 한 화면에서 읽기 쉽도록 시각화했습니다.")

        st.markdown('<div class="shell">', unsafe_allow_html=True)
        fig = px.scatter(
            cluster_source,
            x=raw_x,
            y=raw_y,
            color="유형",
            text="순수학교명",
            hover_data={"연도": True, "시군": True, "학년": True, "성별": True},
            labels={raw_x: x_ax, raw_y: y_ax, "유형": "집단"},
            color_discrete_map={
                "관리 필요군": "#d44b57",
                "고위험군": "#d44b57",
                "중점관리군": "#ef8b2c",
                "일반군": "#1c9d74",
                "우수군": "#2574ea",
                "건강 양호군": "#2574ea",
            },
        )
        fig.update_traces(
            marker=dict(size=17, opacity=0.88, line=dict(width=1.2, color="white")),
            textposition="top center",
            textfont=dict(size=10, color="#254258"),
        )
        fig.update_layout(
            height=620,
            margin=dict(t=10, b=10, l=10, r=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.72)",
            ),
            xaxis=dict(showgrid=True, gridcolor="rgba(16,34,53,0.08)", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(16,34,53,0.08)", zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with sub_tabs[2]:
        st.markdown("### 맞춤형 처방")
        st.markdown("표 대신 카드형 리포트 형식으로 집단별 운동 처방과 교육 프로그램 방향을 정리했습니다.")

        row_order = ["고위험군", "관리 필요군", "중점관리군", "일반군", "우수군", "건강 양호군"]
        visible_rows = [label for label in row_order if label in cluster_source["유형"].unique()]

        for start in range(0, len(visible_rows), 2):
            cols = st.columns(2)
            for col, label in zip(cols, visible_rows[start:start + 2]):
                tag_class = get_group_style(label)
                title_1, body_1, title_2, body_2 = get_prescription_content(label)
                subset = cluster_source[cluster_source["유형"] == label]
                with col:
                    st.markdown(
                        f"""
                        <div class="report-card">
                            <span class="report-tag {tag_class}">{label}</span>
                            <h4>{label} 맞춤 전략</h4>
                            <div class="report-stat">
                                학교 수 {len(subset)} · {x_ax} 평균 {subset[raw_x].mean():.1f} · {y_ax} 평균 {subset[raw_y].mean():.1f}
                            </div>
                            <div class="report-section">운동 처방</div>
                            <div class="report-copy"><strong>{title_1}</strong><br>{body_1}</div>
                            <div class="report-section">교육 프로그램</div>
                            <div class="report-copy"><strong>{title_2}</strong><br>{body_2}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


def render_student_view():
    st.markdown(
        """
        <div class="student-shell">
            <div class="student-badge">학생 개인 FITT 시뮬레이션</div>
            <div class="student-title">나의 PAPS CARE+ 맞춤 처방</div>
            <div class="student-copy">
                심사위원이 직접 가상의 키, 몸무게, 셔틀런 횟수를 입력하면 즉시 알로메트릭 스케일링을 적용해
                개인 체력군을 시뮬레이션하고 4주 FITT 처방 카드를 보여줍니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        height_cm = st.number_input("키 (cm)", min_value=120, max_value=210, value=165, step=1)
    with col2:
        weight_kg = st.number_input("몸무게 (kg)", min_value=25, max_value=150, value=58, step=1)
    with col3:
        shuttle_runs = st.number_input("셔틀런 횟수", min_value=1, max_value=200, value=42, step=1)

    student_result = classify_student_profile(height_cm, weight_kg, shuttle_runs)
    bmi = student_result["bmi"]
    allometric_index = student_result["allometric_index"]
    cluster_label = student_result["cluster_label"]
    tag_class = get_group_style(cluster_label)
    fitness_grade = compute_student_grade(cluster_label)
    grade_score = max(1, 6 - fitness_grade)
    stars = "★" * grade_score + "☆" * (5 - grade_score)
    first_plan = student_result["plan"][0]

    stat1, stat2, stat3 = st.columns(3)
    with stat1:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="mini-label">BMI</div>
                <div class="summary-value">{bmi:.1f}</div>
                <div class="summary-help">키와 몸무게 기반 개인 체질량지수입니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat2:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="mini-label">Allometric Shuttle Index</div>
                <div class="summary-value">{allometric_index:.2f}</div>
                <div class="summary-help">셔틀런 기록을 체중 보정한 심폐 지표입니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat3:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="mini-label">Cluster</div>
                <div class="summary-value" style="font-size:30px;">{cluster_label}</div>
                <div class="summary-help">데모용 학생 개인 체력군 분류 결과입니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([0.95, 1.05])
    with left:
        st.markdown(
            f"""
            <div class="report-card">
                <span class="report-tag {tag_class}">{cluster_label}</span>
                <h4>나의 현재 상태</h4>
                <div class="report-copy">{student_result["summary"]}</div>
                <div class="report-section">FITT 처방 핵심</div>
                <div class="report-copy">
                    Frequency는 주당 운동 빈도, Intensity는 운동 강도, Time은 1회 운동 시간,
                    Type은 실제 수행 종목을 뜻합니다. 아래 4주 계획은 단계적으로 강도가 올라가도록 설계했습니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="report-card">
                <h4>시뮬레이션 안내</h4>
                <div class="report-copy">
                    학생 개인 뷰는 발표용 데모 화면입니다. 입력값이 바뀌면 즉시 체력군과 4주 플랜이 바뀌며,
                    심폐지표에는 알로메트릭 스케일링을 적용해 체중 차이에 따른 보정 효과를 보여줍니다.
                </div>
                <div class="report-section">적용 식</div>
                <div class="report-copy">알로메트릭 지수 = 셔틀런 횟수 ÷ 몸무게<sup>0.33</sup></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 학생용 앱 화면 시뮬레이션")
    st.markdown(
        f"""
        <div class="phone-stage">
            <div class="phone-shell">
                <div class="phone-screen">
                    <div class="phone-dynamic"></div>
                    <div class="phone-top">
                        <span>9:41</span>
                        <span>Diagnosis</span>
                        <span>🔔</span>
                    </div>
                    <div class="segmented">
                        <span class="active">현재 상태</span>
                        <span>종합 평가</span>
                    </div>
                    <div class="diagnosis-layout">
                        <div class="gauge-wrap">
                            <div class="gauge-ring"></div>
                            <div class="gauge-needle"></div>
                            <div class="gauge-text">
                                <strong>{fitness_grade}급</strong>
                                <span>체력 등급 · {stars}</span>
                            </div>
                        </div>
                        <div class="avatar-card">
                            <div class="avatar-circle">🧍</div>
                            <div class="avatar-meta">
                                <div><b>입력 상태</b><strong>키 {height_cm}cm · 몸무게 {weight_kg}kg</strong></div>
                                <div><b>심폐 상태</b><strong>셔틀런 {int(shuttle_runs)}회</strong></div>
                                <div><b>현재 군집</b><strong>{cluster_label}</strong></div>
                                <div><b>보정 지표</b><strong>{allometric_index:.2f}</strong></div>
                            </div>
                        </div>
                    </div>
                    <div class="bottom-nav">
                        <span>⌂</span>
                        <span>◎</span>
                        <div class="fab">+</div>
                        <span>∿</span>
                        <span>◌</span>
                    </div>
                </div>
            </div>
            <div class="phone-arrow">≫</div>
            <div class="phone-shell">
                <div class="phone-screen">
                    <div class="phone-dynamic"></div>
                    <div class="phone-top">
                        <span>‹</span>
                        <span>Prescription</span>
                        <span>📋</span>
                    </div>
                    <div class="prescription-card">
                        <small>FITT</small>
                        <h4>{cluster_label} 학생을 위한<br>4주 맞춤 운동 플랜</h4>
                        <p>{student_result["summary"]}</p>
                        <div class="runner-stage">
                            <div class="pulse-line"></div>
                            <div class="runner-emoji">🏃</div>
                        </div>
                        <div class="fitt-row">
                            <div class="fitt-mini"><b>Frequency</b><span>{first_plan["freq"]}</span></div>
                            <div class="fitt-mini"><b>Intensity</b><span>{first_plan["intensity"]}</span></div>
                            <div class="fitt-mini"><b>Time</b><span>{first_plan["time"]}</span></div>
                            <div class="fitt-mini"><b>Type</b><span>{first_plan["type"]}</span></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 4주 FITT 맞춤 처방")
    plan_cols = st.columns(4)
    for col, plan in zip(plan_cols, student_result["plan"]):
        with col:
            st.markdown(
                f"""
                <div class="plan-card">
                    <div class="plan-week">{plan["week"]}</div>
                    <div class="plan-title">{plan["title"]}</div>
                    <div class="plan-item"><strong>F</strong> · {plan["freq"]}</div>
                    <div class="plan-item"><strong>I</strong> · {plan["intensity"]}</div>
                    <div class="plan-item"><strong>T</strong> · {plan["time"]}</div>
                    <div class="plan-item"><strong>T</strong> · {plan["type"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


raw_df, meta, load_error = load_raw_data()

if load_error:
    st.error(load_error)
    st.info("`data/PAPS_Combined_Data.xlsx` 파일을 추가한 뒤 다시 실행해 주세요.")
    st.stop()

st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-badge">PC+</div>
            <div class="brand-copy">
                <h1>PAPS CARE+</h1>
                <p>학교 체력 데이터 기반 AI 분석 대시보드</p>
            </div>
        </div>
        <div class="status-chip">강원특별자치도 학교 체력 분석 리포트</div>
    </div>
    """,
    unsafe_allow_html=True,
)

main_tabs = st.tabs(["🏫 교육청 관리자 뷰", "📱 학생 개인 뷰"])

with main_tabs[0]:
    render_admin_view(raw_df, meta)

with main_tabs[1]:
    render_student_view()
