import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="PAPS CARE+",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Noto Sans KR', sans-serif;
    background-color: #f5f6fa !important;
    color: #1a2233;
}

#MainMenu, header, footer {
    visibility: hidden;
}

[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e8eaf0;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0;
}

.sidebar-logo {
    background: #1a2233;
    color: #ffffff;
    padding: 18px 20px;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 8px;
}

.sidebar-section {
    font-size: 10px;
    font-weight: 600;
    color: #9ca3b0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 14px 20px 4px 20px;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}

.breadcrumb {
    font-size: 12px;
    color: #9ca3b0;
    margin-bottom: 6px;
}

.breadcrumb span {
    color: #4a5568;
    font-weight: 500;
}

.page-title {
    font-size: 24px;
    font-weight: 700;
    color: #1a2233;
    margin-bottom: 18px;
    letter-spacing: -0.01em;
}

.notice-card {
    background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
    border: 1px solid #e8eaf0;
    border-radius: 12px;
    padding: 16px 18px;
    color: #475467;
    font-size: 13px;
    line-height: 1.8;
    margin-bottom: 18px;
}

.top-header {
    background: #ffffff;
    border-bottom: 1px solid #e8eaf0;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 20px;
    border-radius: 10px;
}

.top-header .breadcrumb-nav {
    font-size: 13px;
    color: #9ca3b0;
}

.top-header .breadcrumb-nav b {
    color: #374151;
}

[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid #e8eaf0;
    gap: 0;
}

[data-testid="stTabs"] [role="tab"] {
    font-size: 13px;
    font-weight: 500;
    color: #6b7280;
    padding: 10px 18px;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    background: none;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #1a2233;
    font-weight: 700;
    border-bottom: 2px solid #1a2233;
}

.panel-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
    border: 1px solid #eef1f5;
}

.map-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 4px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    overflow: hidden;
    border: 1px solid #eef1f5;
}

.report-card {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    height: 100%;
}

.report-tag {
    display: inline-flex;
    align-items: center;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 10px;
}

.tag-red { background: rgba(212,75,87,0.12); color: #b22d3c; }
.tag-orange { background: rgba(239,139,44,0.14); color: #b96215; }
.tag-green { background: rgba(28,157,116,0.14); color: #0f7658; }
.tag-blue { background: rgba(37,116,234,0.14); color: #1f56ba; }

.report-card h4 {
    margin: 0 0 8px 0;
    font-size: 18px;
    font-weight: 700;
}

.report-card p {
    margin: 6px 0 0 0;
    font-size: 14px;
    line-height: 1.8;
    color: #475467;
}

.phone-card {
    background: linear-gradient(180deg, #fff8ef 0%, #ffffff 100%);
    border: 1px solid #ece7dd;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 14px 28px rgba(0,0,0,0.06);
}

.phone-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(240,155,91,0.14);
    color: #bc6a24;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 12px;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
</style>
""",
    unsafe_allow_html=True,
)


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
        (centroids["bmi"] - bmi) ** 2
        + ((centroids["allometric"] - allometric_index) * 1.2) ** 2
    ) ** 0.5
    matched = centroids.loc[distances.idxmin()]
    return bmi, allometric_index, matched["label"]


raw_df, meta, load_error = load_raw_data()
if load_error:
    st.error(load_error)
    st.stop()


with st.sidebar:
    st.markdown('<div class="sidebar-logo">🏃 PAPS CARE+</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">필터</div>', unsafe_allow_html=True)
    s_year = st.multiselect("연도", sorted(raw_df["연도"].dropna().unique()))
    s_region = st.multiselect("시·군", sorted(raw_df["시군"].dropna().unique()))
    s_grade = st.multiselect("학년", sorted(raw_df["학년"].dropna().unique()))
    s_gender = st.multiselect("성별", sorted(raw_df["성별"].dropna().unique()))
    school_base_df = apply_filters(raw_df, s_year, s_region, s_grade, s_gender, [])
    s_school = st.multiselect("학교", sorted(school_base_df["순수학교명"].dropna().unique()))

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">대시보드 메뉴</div>', unsafe_allow_html=True)
    nav_options = [
        "📊 통합 대시보드 (Overview)",
        "도내 체력 현황 요약",
        "체력 취약망 지도 (Heatmap)",
        "🤖 AI 체육 데이터 분석 (Analytics)",
        "체격 보정 평가 모델 (Allometric)",
        "AI 다차원 군집 분석",
        "종목/학년별 상세 통계",
        "🏃‍♂️ 맞춤형 체력 증진 (Prescription)",
        "집단별 FITT 처방",
        "학교별 교육 프로그램 추천",
        "🏢 체육 행정 및 정책 지원 (Administration)",
        "체육 강사 우선 배치망",
        "지역별 예산 집행 타당성",
        "📱 학생/학부모 서비스 (B2C Portal)",
        "나의 AI 체력 진단",
        "4주 맞춤 운동 플랜 발급",
    ]
    current_page = st.radio(
        "이동",
        nav_options,
        index=0,
        label_visibility="collapsed",
    )


filtered_df = apply_filters(raw_df, s_year, s_region, s_grade, s_gender, s_school)
if filtered_df.empty:
    st.warning("선택한 조건에 맞는 데이터가 없습니다. 필터를 조정해 주세요.")
    st.stop()

group_cols = ["순수학교명", "연도", "시군", "학년", "성별"]
agg_map = {column: "mean" for column in meta["valid"].values()}
df_agg = filtered_df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()

metric_options = list(meta["valid"].keys())
x_ax = metric_options[0]
y_ax = metric_options[1 if len(metric_options) > 1 else 0]
raw_x = meta["valid"][x_ax]
raw_y = meta["valid"][y_ax]

cluster_source = df_agg.dropna(subset=[raw_x, raw_y]).copy()
if len(cluster_source) < 4:
    st.warning("현재 조건에서는 AI 군집 분석을 수행하기에 데이터가 부족합니다.")
    st.stop()

scaled_points = StandardScaler().fit_transform(cluster_source[[raw_x, raw_y]])
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_source["Cluster"] = kmeans.fit_predict(scaled_points)
cluster_summary = cluster_source.groupby("Cluster")[[raw_x, raw_y]].mean()
cluster_summary["score"] = cluster_summary.mean(axis=1)
cluster_labels = build_cluster_labels(cluster_summary, x_ax)
cluster_source["유형"] = cluster_source["Cluster"].map(cluster_labels)

dominant_group = cluster_source["유형"].value_counts().idxmax()
dominant_share = round((cluster_source["유형"].value_counts().max() / len(cluster_source)) * 100, 1)

st.markdown(
    '''
    <div class="top-header">
        <div class="breadcrumb-nav">교육행정 &gt; <b>PAPS CARE+ Intelligence</b></div>
    </div>
    ''',
    unsafe_allow_html=True,
)

st.markdown(
    f'''
    <div class="breadcrumb">교육행정 &gt; <span>{current_page}</span></div>
    <div class="page-title">체육행정</div>
    ''',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice-card">
        <b>PAPS CARE+ Intelligence</b><br>
        본 시스템은 학교 체력 데이터를 기반으로 집단별 체력 수준을 AI 군집 분석하고,
        취약 지역과 맞춤형 처방 방향을 시각적으로 제공합니다.
    </div>
    """,
    unsafe_allow_html=True,
)

kangwon_coords = {
    "춘천": (37.8813, 127.7298),
    "원주": (37.3422, 127.9202),
    "강릉": (37.7518, 128.8760),
    "동해": (37.5245, 129.1143),
    "태백": (37.1640, 128.9856),
    "속초": (38.2070, 128.5918),
    "삼척": (37.4498, 129.1653),
    "홍천": (37.6970, 127.8887),
    "횡성": (37.4913, 127.9850),
    "영월": (37.1833, 128.4619),
    "평창": (37.3705, 128.3902),
    "정선": (37.3797, 128.6608),
    "철원": (38.1466, 127.3134),
    "화천": (38.1060, 127.7082),
    "양구": (38.1065, 127.9897),
    "인제": (38.0694, 128.1707),
    "고성": (38.3806, 128.4678),
    "양양": (38.0754, 128.6190),
}


def get_coords(city_name):
    clean_name = str(city_name).replace("시", "").replace("군", "").strip()
    return kangwon_coords.get(clean_name, (37.8813, 127.7298))


map_df = cluster_source.copy()
map_df["lat"] = map_df["시군"].apply(lambda value: get_coords(value)[0])
map_df["lon"] = map_df["시군"].apply(lambda value: get_coords(value)[1])
weight_map = {
    "고위험군": 10,
    "관리 필요군": 8,
    "중점관리군": 5,
    "일반군": 1,
    "건강 양호군": 0.5,
    "우수군": 0.1,
}
map_df["weight"] = map_df["유형"].map(weight_map).fillna(1)


def render_heatmap():
    fig = px.density_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        z="weight",
        radius=60,
        center=dict(lat=37.75, lon=128.3),
        zoom=6.8,
        mapbox_style="carto-darkmatter",
        hover_name="시군",
        hover_data={
            "순수학교명": True,
            "유형": True,
            "lat": False,
            "lon": False,
            "weight": False,
        },
        color_continuous_scale=[
            [0.0, "rgba(10, 15, 60, 0.0)"],
            [0.2, "rgba(20, 30, 120, 0.6)"],
            [0.45, "rgba(40, 60, 180, 0.8)"],
            [0.65, "rgba(200, 80, 30, 0.9)"],
            [0.85, "rgba(240, 120, 30, 0.95)"],
            [1.0, "rgba(255, 160, 50, 1.0)"],
        ],
    )
    fig.update_traces(opacity=0.85)
    fig.add_annotation(
        x=0.98,
        y=0.96,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        text=(
            "<b>취약 체력 증목</b><br>"
            "<span style='color:#f07030'>━━━━</span>"
            "<span style='color:#3050b0'>━━━━</span>"
        ),
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.93)",
        bordercolor="rgba(210,215,225,0.8)",
        borderwidth=1,
        borderpad=12,
        font=dict(size=12, color="#1a2233", family="Noto Sans KR"),
    )
    fig.update_layout(
        height=560,
        margin=dict(t=0, b=0, l=0, r=0),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_scatter():
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
        marker=dict(size=16, opacity=0.88, line=dict(width=1.1, color="white")),
        textposition="top center",
        textfont=dict(size=10, color="#254258"),
    )
    fig.update_layout(
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
    st.plotly_chart(fig, use_container_width=True)


if current_page in ["📊 통합 대시보드 (Overview)", "도내 체력 현황 요약"]:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("전체 학교 수", f"{len(cluster_source['순수학교명'].unique())}개교")
    with col2:
        st.metric("주요 집단", dominant_group)
    with col3:
        st.metric("주요 집단 비율", f"{dominant_share}%")
    with col4:
        high_risk = len(cluster_source[cluster_source["유형"].isin(["고위험군", "관리 필요군"])])
        st.metric("관리 필요 학교", f"{high_risk}개교")

    st.markdown("---")
    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown('<div class="map-card">', unsafe_allow_html=True)
        render_heatmap()
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        render_scatter()
        st.markdown("</div>", unsafe_allow_html=True)

elif current_page == "체력 취약망 지도 (Heatmap)":
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="map-card">', unsafe_allow_html=True)
        render_heatmap()
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        render_scatter()
        st.markdown("</div>", unsafe_allow_html=True)

elif current_page in ["🤖 AI 체육 데이터 분석 (Analytics)", "AI 다차원 군집 분석"]:
    st.markdown("#### AI 다차원 군집 분석")
    render_scatter()

elif current_page == "체격 보정 평가 모델 (Allometric)":
    allometric_df = cluster_source.copy()
    allometric_df["체중 추정"] = allometric_df[raw_x].abs().fillna(allometric_df[raw_x].mean()).clip(lower=1)
    allometric_df["보정 심폐지표"] = allometric_df[raw_y] / (allometric_df["체중 추정"] ** 0.33)
    fig = px.scatter(
        allometric_df,
        x=raw_y,
        y="보정 심폐지표",
        color="유형",
        title="원점수 vs 체격 보정 점수 비교",
        labels={raw_y: "원점수", "보정 심폐지표": "체격 보정 점수"},
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

elif current_page == "종목/학년별 상세 통계":
    metric_choice = st.selectbox("상세 지표 선택", list(meta["valid"].keys()))
    metric_col = meta["valid"][metric_choice]
    detail_df = filtered_df.dropna(subset=[metric_col]).copy()
    chart_df = detail_df.groupby(["학년", "성별"])[metric_col].mean().reset_index()
    fig = px.bar(chart_df, x="학년", y=metric_col, color="성별", barmode="group", title=f"{metric_choice} 학년/성별 평균")
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(detail_df[["순수학교명", "시군", "학년", "성별", metric_col]].head(30), use_container_width=True)

elif current_page in ["🏃‍♂️ 맞춤형 체력 증진 (Prescription)", "집단별 FITT 처방"]:
    st.markdown("#### 집단별 FITT 처방")
    row_order = ["고위험군", "관리 필요군", "중점관리군", "일반군", "건강 양호군", "우수군"]
    visible_rows = [label for label in row_order if label in cluster_source["유형"].unique()]
    for start in range(0, len(visible_rows), 2):
        cols = st.columns(2)
        for col, label in zip(cols, visible_rows[start:start + 2]):
            title_1, body_1, title_2, body_2 = get_prescription_content(label)
            subset = cluster_source[cluster_source["유형"] == label]
            tag_class = get_group_style(label)
            with col:
                st.markdown(
                    f"""
                    <div class="report-card">
                        <span class="report-tag {tag_class}">{label}</span>
                        <h4>{label} 처방 가이드</h4>
                        <p><b>학교 수</b> {len(subset)}개교</p>
                        <p><b>{x_ax} 평균</b> {subset[raw_x].mean():.1f} · <b>{y_ax} 평균</b> {subset[raw_y].mean():.1f}</p>
                        <p><b>{title_1}</b><br>{body_1}</p>
                        <p><b>{title_2}</b><br>{body_2}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

elif current_page == "학교별 교육 프로그램 추천":
    risk_schools = cluster_source[cluster_source["유형"].isin(["고위험군", "관리 필요군"])].copy()
    risk_schools = risk_schools.sort_values(by=raw_y, ascending=True)
    st.dataframe(
        risk_schools[["순수학교명", "시군", "연도", "학년", "성별", "유형"]].head(30),
        use_container_width=True,
    )

elif current_page in ["🏢 체육 행정 및 정책 지원 (Administration)", "체육 강사 우선 배치망"]:
    risk_schools = cluster_source[cluster_source["유형"].isin(["고위험군", "관리 필요군"])].copy()
    priority = risk_schools.groupby("순수학교명").agg({"유형": "count", raw_y: "mean", "시군": "first"}).reset_index()
    priority.columns = ["학교명", "취약 학생군 건수", "심폐지표 평균", "시군"]
    priority = priority.sort_values(["취약 학생군 건수", "심폐지표 평균"], ascending=[False, True])
    st.dataframe(priority.head(30), use_container_width=True)

elif current_page == "지역별 예산 집행 타당성":
    budget_df = cluster_source.groupby("시군").agg(
        취약학교수=("유형", lambda x: int(x.isin(["고위험군", "관리 필요군"]).sum())),
        전체학교수=("순수학교명", "count"),
        평균심폐지표=(raw_y, "mean"),
    ).reset_index()
    budget_df["취약비율"] = (budget_df["취약학교수"] / budget_df["전체학교수"] * 100).round(1)
    fig = px.bar(budget_df.sort_values("취약비율", ascending=False), x="시군", y="취약비율", title="지역별 취약비율")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(budget_df.sort_values("취약비율", ascending=False), use_container_width=True)

elif current_page in ["📱 학생/학부모 서비스 (B2C Portal)", "나의 AI 체력 진단", "4주 맞춤 운동 플랜 발급"]:
    st.markdown("#### 학생/학부모 서비스")
    c1, c2, c3 = st.columns(3)
    with c1:
        height_cm = st.number_input("키 (cm)", min_value=120, max_value=210, value=165, step=1)
    with c2:
        weight_kg = st.number_input("몸무게 (kg)", min_value=25, max_value=150, value=58, step=1)
    with c3:
        shuttle_runs = st.number_input("셔틀런 횟수", min_value=1, max_value=200, value=42, step=1)

    bmi, allometric_index, cluster_label = classify_student_profile(height_cm, weight_kg, shuttle_runs)
    title_1, body_1, title_2, body_2 = get_prescription_content(cluster_label)
    tag_class = get_group_style(cluster_label)

    a, b, c = st.columns(3)
    with a:
        st.metric("BMI", f"{bmi:.1f}")
    with b:
        st.metric("보정 심폐지표", f"{allometric_index:.2f}")
    with c:
        st.metric("AI 체력군", cluster_label)

    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""
            <div class="phone-card">
                <div class="phone-badge">나의 AI 체력 진단</div>
                <h4 style="margin:0 0 10px 0;">현재 상태: {cluster_label}</h4>
                <p style="margin:0;color:#475467;line-height:1.8;">
                    키 {height_cm}cm, 몸무게 {weight_kg}kg, 셔틀런 {int(shuttle_runs)}회를 기준으로
                    개인 체력군을 시뮬레이션한 결과입니다.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="phone-card">
                <div class="phone-badge">4주 맞춤 운동 플랜</div>
                <h4 style="margin:0 0 10px 0;">{title_1}</h4>
                <p style="margin:0;color:#475467;line-height:1.8;">{body_1}</p>
                <p style="margin:12px 0 0 0;color:#475467;line-height:1.8;"><b>{title_2}</b><br>{body_2}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
