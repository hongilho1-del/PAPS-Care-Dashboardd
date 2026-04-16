import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="PAPS Care+ Intelligence",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    :root {
        --bg: #f2f6fb;
        --surface: rgba(255, 255, 255, 0.88);
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
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
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
        grid-template-columns: 1.6fr 0.8fr;
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

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.92);
        border: 1px solid var(--stroke);
        border-radius: 24px;
        padding: 16px 18px;
        box-shadow: 0 16px 36px rgba(15, 39, 64, 0.07);
    }

    @media (max-width: 1100px) {
        .hero-grid {
            grid-template-columns: 1fr;
        }
        .hero-highlight {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_raw_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data", "PAPS_Combined_Data.xlsx")

    if not os.path.exists(file_path):
        return None, {}, f"데이터 파일을 찾을 수 없습니다: {file_path}"

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
        "심폐지구력": find_col(["왕복", "오래달리기", "심폐"]),
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

    school_col = find_col(["추출학교명", "학교명"])
    year_col = find_col(["연도"])
    region_col = find_col(["시군"])
    gender_col = find_col(["성별", "남여"])
    grade_col = find_col(["학년"])

    df["순수학교명"] = (
        df[school_col].astype(str).str.strip()
        if school_col
        else df.iloc[:, 0].astype(str).str.strip()
    )
    df["연도"] = (
        pd.to_numeric(df[year_col], errors="coerce").fillna(0).astype(int)
        if year_col
        else 0
    )
    df["시군"] = df[region_col].astype(str).str.strip() if region_col else "미상"
    df["성별"] = df[gender_col].astype(str).str.strip() if gender_col else "전체"
    df["학년"] = df[grade_col].astype(str).str.strip() if grade_col else "전체"
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

with st.sidebar:
    st.markdown("## 데이터 필터")
    st.caption("선택한 조건을 기준으로 차트와 리포트가 다시 계산됩니다.")

    s_year = st.multiselect("연도", sorted(raw_df["연도"].dropna().unique()))
    s_region = st.multiselect("시·군", sorted(raw_df["시군"].dropna().unique()))
    s_grade = st.multiselect("학년", sorted(raw_df["학년"].dropna().unique()))
    s_gender = st.multiselect("성별", sorted(raw_df["성별"].dropna().unique()))

    school_base_df = apply_filters(raw_df, s_year, s_region, s_grade, s_gender, [])
    school_options = sorted(school_base_df["순수학교명"].dropna().unique())
    s_school = st.multiselect("학교", school_options)

    st.markdown("---")
    st.markdown("## 분석 설정")
    metric_options = list(meta["valid"].keys())
    x_ax = st.selectbox("수평축", metric_options, index=0)
    y_ax = st.selectbox("수직축", metric_options, index=1 if len(metric_options) > 1 else 0)
    n_cl = st.slider("군집 수", 2, 4, 3)

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

st.markdown('<div class="shell shell-dark">', unsafe_allow_html=True)
st.markdown('<h3 class="panel-title">현재 분석 기준</h3>', unsafe_allow_html=True)
st.markdown(
    f'<p class="panel-copy">연도 {format_selection(s_year)} · 지역 {format_selection(s_region)} · 학년 {format_selection(s_grade)} · 성별 {format_selection(s_gender)} · 학교 {format_selection(s_school)}</p>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

filtered_df = apply_filters(raw_df, s_year, s_region, s_grade, s_gender, s_school)
if filtered_df.empty:
    st.warning("선택한 조건에 맞는 데이터가 없습니다. 필터를 조정해 주세요.")
    st.stop()

group_cols = ["순수학교명", "연도", "시군", "학년", "성별"]
agg_map = {column: "mean" for column in meta["valid"].values()}
df_agg = filtered_df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()

raw_x = meta["valid"][x_ax]
raw_y = meta["valid"][y_ax]
cluster_source = df_agg.dropna(subset=[raw_x, raw_y]).copy()

if len(cluster_source) < n_cl:
    st.warning(f"현재 조건에서는 군집 {n_cl}개를 만들 데이터가 부족합니다. 필터를 조금 넓혀 주세요.")
    st.stop()

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

tabs = st.tabs(["종합 현황", "군집 분포 맵", "맞춤형 처방"])

with tabs[0]:
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

with tabs[1]:
    st.markdown("### 군집 분포 맵")
    st.markdown("학교별 위치와 집단 분포를 한 화면에서 읽기 쉽도록 시각화했습니다.")

    st.markdown('<div class="shell">', unsafe_allow_html=True)
    fig = px.scatter(
        cluster_source,
        x=raw_x,
        y=raw_y,
        color="유형",
        size_max=18,
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

with tabs[2]:
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
