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
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    :root {
        --bg: #eef3f7;
        --panel: rgba(255, 255, 255, 0.82);
        --panel-strong: #ffffff;
        --ink: #102437;
        --muted: #5d7286;
        --line: rgba(16, 36, 55, 0.10);
        --navy: #16324f;
        --blue: #2f6fed;
        --cyan: #45b9d6;
        --gold: #e4b04a;
        --danger: #d24d57;
        --warning: #f08b37;
        --success: #26a27b;
        --info: #377dff;
        --shadow: 0 24px 60px rgba(18, 42, 66, 0.10);
    }

    html, body, [class*="css"] {
        font-family: 'Pretendard', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(69, 185, 214, 0.18), transparent 32%),
            radial-gradient(circle at top right, rgba(228, 176, 74, 0.16), transparent 24%),
            linear-gradient(180deg, #f4f8fb 0%, #edf2f7 100%);
        color: var(--ink);
    }

    #MainMenu, header, footer {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10263d 0%, #173452 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f4f8fb !important;
    }

    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        border-radius: 32px;
        padding: 34px 38px;
        background:
            radial-gradient(circle at 82% 22%, rgba(255,255,255,0.18), transparent 18%),
            linear-gradient(135deg, #10263d 0%, #1d4b73 55%, #2779a7 100%);
        color: #f6fbff;
        box-shadow: 0 28px 70px rgba(13, 31, 48, 0.28);
        margin-bottom: 24px;
    }

    .hero-shell::after {
        content: "";
        position: absolute;
        inset: auto -80px -120px auto;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(228,176,74,0.28) 0%, rgba(228,176,74,0.02) 70%);
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 999px;
        padding: 8px 14px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.14);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 18px 0 10px;
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.08;
    }

    .hero-copy {
        max-width: 720px;
        font-size: 16px;
        line-height: 1.7;
        color: rgba(246, 251, 255, 0.86);
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.8fr 1fr;
        gap: 18px;
        align-items: end;
    }

    .hero-note {
        display: flex;
        flex-direction: column;
        gap: 14px;
        padding: 20px;
        border-radius: 24px;
        background: rgba(255,255,255,0.09);
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(12px);
    }

    .hero-note h4 {
        margin: 0;
        font-size: 15px;
        font-weight: 700;
    }

    .hero-note p {
        margin: 0;
        color: rgba(246, 251, 255, 0.82);
        line-height: 1.65;
        font-size: 13px;
    }

    .section-title {
        margin: 26px 0 4px;
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--ink);
    }

    .section-copy {
        color: var(--muted);
        margin-bottom: 16px;
        font-size: 14px;
    }

    .metric-card {
        background: var(--panel);
        border: 1px solid rgba(255,255,255,0.55);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 22px;
        box-shadow: var(--shadow);
        min-height: 150px;
    }

    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        font-weight: 700;
    }

    .metric-value {
        margin-top: 14px;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: var(--ink);
    }

    .metric-help {
        margin-top: 10px;
        font-size: 13px;
        line-height: 1.6;
        color: var(--muted);
    }

    .panel-card {
        background: var(--panel);
        border: 1px solid rgba(255,255,255,0.65);
        backdrop-filter: blur(12px);
        border-radius: 28px;
        padding: 24px;
        box-shadow: var(--shadow);
    }

    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 18px;
        margin-bottom: 16px;
    }

    .panel-title {
        margin: 0;
        font-size: 22px;
        font-weight: 800;
        color: var(--ink);
    }

    .panel-subtitle {
        margin: 6px 0 0;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
    }

    .control-shell {
        background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(245,248,252,0.92) 100%);
        border: 1px solid rgba(17, 36, 56, 0.08);
        border-radius: 26px;
        padding: 18px;
        box-shadow: 0 18px 45px rgba(13, 31, 48, 0.08);
        margin-bottom: 22px;
    }

    .pill-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 12px;
    }

    .pill {
        border-radius: 999px;
        padding: 8px 12px;
        background: #edf4ff;
        color: #2854aa;
        font-size: 12px;
        font-weight: 700;
    }

    .report-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
    }

    .report-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(17, 36, 56, 0.08);
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 16px 36px rgba(13, 31, 48, 0.08);
    }

    .report-tag {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 16px;
    }

    .tag-red { background: rgba(210, 77, 87, 0.14); color: #aa2c3a; }
    .tag-orange { background: rgba(240, 139, 55, 0.16); color: #b65d11; }
    .tag-green { background: rgba(38, 162, 123, 0.14); color: #127453; }
    .tag-blue { background: rgba(55, 125, 255, 0.14); color: #2056c9; }

    .report-title {
        margin: 0 0 12px;
        font-size: 18px;
        font-weight: 800;
        color: var(--ink);
    }

    .report-stat {
        margin-bottom: 14px;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.7;
    }

    .report-block-title {
        margin: 14px 0 8px;
        font-size: 13px;
        font-weight: 800;
        color: var(--ink);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .report-copy {
        color: #304558;
        line-height: 1.8;
        font-size: 14px;
    }

    .stAlert {
        border-radius: 18px;
    }

    @media (max-width: 1100px) {
        .hero-grid {
            grid-template-columns: 1fr;
        }
        .report-grid {
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

    meta = {
        "valid": valid_targets,
        "file_path": file_path,
    }
    return df, meta, None


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
    if x_label == "BMI":
        ordered = cluster_summary.sort_values("score", ascending=True).index.tolist()
    else:
        ordered = cluster_summary.sort_values("score", ascending=False).index.tolist()

    label_sets = {
        2: ["관리 필요군", "건강 양호군"],
        3: ["고위험군", "일반군", "우수군"],
        4: ["고위험군", "중점관리군", "일반군", "우수군"],
    }
    names = label_sets[len(ordered)]
    return {cluster_id: names[index] for index, cluster_id in enumerate(ordered)}


def get_group_style(label):
    if "고위험" in label or "관리 필요" in label:
        return "tag-red", "#d24d57"
    if "중점관리" in label:
        return "tag-orange", "#f08b37"
    if "일반" in label:
        return "tag-green", "#26a27b"
    return "tag-blue", "#377dff"


def get_prescription_content(label):
    if "고위험" in label or "관리 필요" in label:
        return (
            "기초 체력 회복 중심 처방",
            "저충격 유산소와 기초 근력 운동을 병행해 활동량을 안전하게 확보하고, 무리 없는 주간 루틴으로 운동 적응도를 높입니다.",
            "집중 지원 프로그램",
            "소그룹 건강체력교실, 영양 상담, 가정 연계형 생활습관 코칭을 묶어 지속 관리가 가능하도록 설계합니다.",
        )
    if "중점관리" in label:
        return (
            "참여도 강화형 처방",
            "뉴스포츠, 순환 운동, 기록 기반 목표 관리를 통해 흥미를 유지하면서 심폐지구력과 근지구력을 단계적으로 끌어올립니다.",
            "방과 후 성장 프로그램",
            "또래 참여형 스포츠클럽과 성취 피드백을 연결해 운동 지속성과 자기효능감을 함께 높입니다.",
        )
    if "일반" in label:
        return (
            "균형 유지형 처방",
            "근력, 유연성, 지구력의 밸런스를 유지하는 주간 루틴을 운영하며 부상 예방을 위한 스트레칭과 기초 회복 습관을 강화합니다.",
            "자율 체육 습관화",
            "1인 1운동, 선택형 종목 체험, 자기주도 기록 관리를 통해 생활 속 운동 습관을 안정적으로 정착시킵니다.",
        )
    return (
        "심화 성장형 처방",
        "인터벌 트레이닝, 개인 강점 보완 훈련, 종목 특화 루틴을 통해 상위 체력군의 성장을 정교하게 이어갑니다.",
        "리더십 연계 프로그램",
        "교내 스포츠 리더, 멘토링 활동, 지역 심화 프로그램 연계를 통해 우수군의 역할과 동기를 함께 확장합니다.",
    )


def format_selection(values, empty_text="전체"):
    if not values:
        return empty_text
    if len(values) <= 2:
        return ", ".join(map(str, values))
    return f"{', '.join(map(str, values[:2]))} 외 {len(values) - 2}개"


raw_df, meta, load_error = load_raw_data()

st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-grid">
            <div>
                <div class="eyebrow">PAPS CARE+ Intelligence Suite</div>
                <div class="hero-title">학교 체력 데이터를<br>기관용 분석 서비스처럼 보여주는 대시보드</div>
                <div class="hero-copy">
                    강원특별자치도 학교 데이터를 기반으로 체력 지표 분포를 군집화하고,
                    집단별 관리 우선순위와 실행 가능한 프로그램 제안을 한 화면에서 제공합니다.
                </div>
            </div>
            <div class="hero-note">
                <h4>Professional View</h4>
                <p>기본 스트림릿 화면이 아니라 기관 보고용 정보 위계로 재구성했습니다.</p>
                <p>상단 요약, 분석 설정, 산점도, 집단별 처방 보드가 자연스럽게 이어지도록 설계했습니다.</p>
                <p>필터는 실제 분석에 반영되며, 선택된 조건만 기준으로 AI 군집을 재계산합니다.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.error(load_error)
    st.info("`data/PAPS_Combined_Data.xlsx` 파일을 추가한 뒤 다시 실행해 주세요.")
    st.stop()


with st.sidebar:
    st.markdown("## 분석 조건")
    st.caption("필터를 선택하면 차트와 처방 보드가 같은 기준으로 다시 계산됩니다.")

    s_year = st.multiselect("연도", sorted(raw_df["연도"].dropna().unique()))
    s_region = st.multiselect("시·군", sorted(raw_df["시군"].dropna().unique()))
    s_grade = st.multiselect("학년", sorted(raw_df["학년"].dropna().unique()))
    s_gender = st.multiselect("성별", sorted(raw_df["성별"].dropna().unique()))

    school_base_df = apply_filters(raw_df, s_year, s_region, s_grade, s_gender, [])
    school_options = sorted(school_base_df["순수학교명"].dropna().unique())
    s_school = st.multiselect("학교명", school_options)

metric_options = list(meta["valid"].keys())

st.markdown("### 현재 분석 범위")
st.markdown(
    f"""
    <div class="pill-row">
        <div class="pill">연도: {format_selection(s_year)}</div>
        <div class="pill">지역: {format_selection(s_region)}</div>
        <div class="pill">학년: {format_selection(s_grade)}</div>
        <div class="pill">성별: {format_selection(s_gender)}</div>
        <div class="pill">학교: {format_selection(s_school)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

filtered_df = apply_filters(raw_df, s_year, s_region, s_grade, s_gender, s_school)

if filtered_df.empty:
    st.warning("선택한 조건에 맞는 데이터가 없습니다. 필터를 조정해 주세요.")
    st.stop()

group_cols = ["순수학교명", "연도", "시군", "학년", "성별"]
agg_map = {column: "mean" for column in meta["valid"].values()}
df_agg = filtered_df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()

st.markdown(
    """
    <div class="control-shell">
        <div class="panel-header">
            <div>
                <h3 class="panel-title">분석 프레임 설정</h3>
                <p class="panel-subtitle">기관 보고서처럼 보이도록 핵심 비교 지표와 군집 세분화 수준을 먼저 고정합니다.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.2, 1])
with ctrl1:
    x_ax = st.selectbox("수평축 지표", metric_options, index=0)
with ctrl2:
    y_ax = st.selectbox("수직축 지표", metric_options, index=1 if len(metric_options) > 1 else 0)
with ctrl3:
    n_cl = st.slider("군집 수", 2, 4, 3)

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
year_span = f"{int(cluster_source['연도'].min())} - {int(cluster_source['연도'].max())}"
top_group = cluster_source["유형"].value_counts().idxmax()

st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">상단 요약은 현재 필터 기준으로 다시 계산된 결과입니다.</div>',
    unsafe_allow_html=True,
)

metric1, metric2, metric3, metric4 = st.columns(4)
metric_cards = [
    ("대상 학교 수", f"{school_count}", "현재 조건에 포함된 학교 단위 집계 수"),
    ("분석 지역 수", f"{region_count}", "시·군 기준으로 확인된 지역 범위"),
    ("분석 연도 범위", year_span, "필터 적용 후 실제 분석에 사용된 기간"),
    ("최대 비중 집단", top_group, "현재 군집 결과에서 가장 큰 비중의 그룹"),
]
for column, (label, value, help_text) in zip((metric1, metric2, metric3, metric4), metric_cards):
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-help">{help_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

left, right = st.columns([1.65, 1])

with left:
    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-header">
                <div>
                    <h3 class="panel-title">Cluster Position Map</h3>
                    <p class="panel-subtitle">선택한 두 지표를 기준으로 학교별 상대 위치와 집단 분포를 시각화합니다.</p>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    color_map = {
        "관리 필요군": "#d24d57",
        "고위험군": "#d24d57",
        "중점관리군": "#f08b37",
        "일반군": "#26a27b",
        "우수군": "#377dff",
        "건강 양호군": "#377dff",
    }
    fig = px.scatter(
        cluster_source,
        x=raw_x,
        y=raw_y,
        color="유형",
        text="순수학교명",
        labels={raw_x: x_ax, raw_y: y_ax, "유형": "집단"},
        hover_data={"연도": True, "시군": True, "학년": True, "성별": True},
        color_discrete_map=color_map,
    )
    fig.update_traces(
        marker=dict(size=16, opacity=0.88, line=dict(width=1.2, color="white")),
        textposition="top center",
        textfont=dict(size=10, color="#27415a"),
    )
    fig.update_layout(
        height=560,
        margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="rgba(255,255,255,0)",
        paper_bgcolor="rgba(255,255,255,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.70)",
        ),
        xaxis=dict(showgrid=True, gridcolor="rgba(16,36,55,0.08)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(16,36,55,0.08)", zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-header">
                <div>
                    <h3 class="panel-title">Analysis Note</h3>
                    <p class="panel-subtitle">해석 시 확인해야 할 핵심 포인트를 함께 제공합니다.</p>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    top_counts = cluster_source["유형"].value_counts()
    for label, count in top_counts.items():
        tag_class, tag_color = get_group_style(label)
        share = round((count / len(cluster_source)) * 100, 1)
        st.markdown(
            f"""
            <div class="report-card" style="padding:18px; margin-bottom:12px; box-shadow:none;">
                <span class="report-tag {tag_class}">{label}</span>
                <div class="report-stat">구성 비중 {share}% · 학교 수 {int(count)}</div>
                <div class="report-copy">선택 지표 조합에서 상대적으로 유사한 위치를 보이는 학교 집단입니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "안내: 군집 라벨은 선택한 지표 조합에 대한 상대 비교입니다. BMI는 높은 값이 항상 우수하다고 해석되지 않도록 별도 방향성을 적용했습니다."
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section-title">맞춤형 처방 보드</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">집단별 평균 지표와 권장 실행 방향을 카드형 리포트로 정리했습니다.</div>',
    unsafe_allow_html=True,
)

sum_df = cluster_source.groupby("유형")[[raw_x, raw_y]].mean().round(1)
row_order = ["고위험군", "관리 필요군", "중점관리군", "일반군", "우수군", "건강 양호군"]
visible_rows = [label for label in row_order if label in sum_df.index]

report_html = ['<div class="report-grid">']
for label in visible_rows:
    tag_class, _ = get_group_style(label)
    title_1, body_1, title_2, body_2 = get_prescription_content(label)
    report_html.append(
        f"""
        <div class="report-card">
            <span class="report-tag {tag_class}">{label}</span>
            <h4 class="report-title">{label} 맞춤 전략</h4>
            <div class="report-stat">{x_ax} 평균 {sum_df.loc[label, raw_x]} · {y_ax} 평균 {sum_df.loc[label, raw_y]}</div>
            <div class="report-block-title">운동 처방</div>
            <div class="report-copy"><strong>{title_1}</strong><br>{body_1}</div>
            <div class="report-block-title">교육 프로그램</div>
            <div class="report-copy"><strong>{title_2}</strong><br>{body_2}</div>
        </div>
        """
    )
report_html.append("</div>")
st.markdown("".join(report_html), unsafe_allow_html=True)
