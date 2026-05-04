import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="체육행정",
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

.sidebar-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 20px;
    font-size: 13px;
    color: #4a5568;
    border-radius: 0;
    transition: background 0.15s;
    text-decoration: none;
}
.sidebar-item:hover {
    background: #f0f2f7;
    color: #1a2233;
}
.sidebar-item.active {
    background: #eef2ff;
    color: #3b5bdb;
    font-weight: 600;
    border-right: 3px solid #3b5bdb;
}
.sidebar-item .icon {
    font-size: 15px;
    width: 18px;
    text-align: center;
}

.main-content {
    padding: 0 8px;
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
    font-size: 22px;
    font-weight: 700;
    color: #1a2233;
    margin-bottom: 20px;
    letter-spacing: -0.01em;
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

.filter-btn {
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    border: 1.5px solid #d1d5db;
    background: white;
    color: #374151;
    text-align: center;
}
.filter-btn.active {
    background: #1a2233;
    color: white;
    border-color: #1a2233;
}

.map-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 4px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    overflow: hidden;
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

[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {
    border-radius: 8px;
    font-size: 13px;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}

.divider {
    border: none;
    border-top: 1px solid #f0f2f5;
    margin: 4px 0;
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

.prescription-card {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    height: 100%;
}

.prescription-tag {
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

.prescription-card h4 {
    margin: 0 0 8px 0;
    font-size: 18px;
    font-weight: 700;
}
.prescription-card p {
    margin: 6px 0 0 0;
    font-size: 14px;
    line-height: 1.8;
    color: #475467;
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


raw_df, meta, load_error = load_raw_data()

if load_error:
    st.error(load_error)
    st.info("`data/PAPS_Final_Master (5).xlsx` 또는 `data/PAPS_Combined_Data.xlsx` 파일을 확인해 주세요.")
    st.stop()


with st.sidebar:
    st.markdown('<div class="sidebar-logo">🏃 체육행정 시스템</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">대시보드</div>', unsafe_allow_html=True)
    st.markdown(
        '''
        <div class="sidebar-item active"><span class="icon">📊</span> 체육 행정</div>
        <div class="sidebar-item"><span class="icon">📋</span> 체력 랭킹</div>
        <div class="sidebar-item"><span class="icon">🏫</span> 학교 현황</div>
        <div class="sidebar-item"><span class="icon">📁</span> 지역 분석</div>
        <div class="sidebar-item"><span class="icon">👨‍🏫</span> 참여 교사</div>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">필터</div>', unsafe_allow_html=True)
    s_year = st.multiselect("연도", sorted(raw_df["연도"].dropna().unique()))
    s_region = st.multiselect("시·군", sorted(raw_df["시군"].dropna().unique()))
    s_grade = st.multiselect("학년", sorted(raw_df["학년"].dropna().unique()))
    s_gender = st.multiselect("성별", sorted(raw_df["성별"].dropna().unique()))

    school_base_df = apply_filters(raw_df, s_year, s_region, s_grade, s_gender, [])
    school_options = sorted(school_base_df["순수학교명"].dropna().unique())
    s_school = st.multiselect("학교", school_options)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">분석 설정</div>', unsafe_allow_html=True)
    metric_options = list(meta["valid"].keys())
    x_ax = st.selectbox("수평축", metric_options, index=0)
    y_ax = st.selectbox("수직축", metric_options, index=1 if len(metric_options) > 1 else 0)
    n_cl = st.slider("군집 수", 2, 4, 3)


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

st.markdown(
    '''
    <div class="top-header">
        <div class="breadcrumb-nav">교육행정 &gt; <b>취약체력 가름</b></div>
    </div>
    ''',
    unsafe_allow_html=True,
)

st.markdown(
    '''
    <div class="breadcrumb">교육행정 &gt; <span>취약 체력 가름</span></div>
    <div class="page-title">체육행정</div>
    ''',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice-card">
        <b>PAPS Care+ Intelligence</b><br>
        학교 체력 데이터를 기반으로 집단별 체력 수준을 AI 군집 분석하고, 취약 지역과 맞춤형 처방 방향을 시각적으로 제공합니다.
    </div>
    """,
    unsafe_allow_html=True,
)

dominant_group = cluster_source["유형"].value_counts().idxmax()
dominant_share = round((cluster_source["유형"].value_counts().max() / len(cluster_source)) * 100, 1)

sub_tabs = st.tabs(["종합 현황", "취약 체력 지역 히트맵", "맞춤형 처방"])

with sub_tabs[0]:
    st.markdown("#### 종합 현황")

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

    left, right = st.columns([1.2, 1])
    with left:
        type_counts = cluster_source["유형"].value_counts().reset_index()
        type_counts.columns = ["유형", "수"]
        color_map = {
            "고위험군": "#d44b57",
            "관리 필요군": "#e8734a",
            "중점관리군": "#ef8b2c",
            "일반군": "#1c9d74",
            "건강 양호군": "#2574ea",
            "우수군": "#1a56db",
        }
        fig_bar = px.bar(
            type_counts,
            x="유형",
            y="수",
            color="유형",
            color_discrete_map=color_map,
            title="집단별 학교 분포",
        )
        fig_bar.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(t=40, b=20, l=20, r=20),
            font=dict(family="Noto Sans KR"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        st.markdown(
            f"""
            <div class="report-card">
                <span class="report-tag {get_group_style(dominant_group)}">{dominant_group}</span>
                <h4>현재 해석 포인트</h4>
                <p>현재 필터 기준으로 가장 큰 비중을 차지하는 집단은 <b>{dominant_group}</b>이며 전체의 <b>{dominant_share}%</b>입니다.</p>
                <p>{x_ax}와 {y_ax}를 기준으로 상대 비교한 결과이며, 선택된 학교·지역 조건에 따라 결과가 다시 계산됩니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with sub_tabs[1]:
    st.markdown("#### 취약 체력 지역 히트맵")

    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        st.selectbox(
            "교체 집단",
            ["자부 성장", "전체", "고위험군만", "관리 필요군 이상"],
            index=0,
            label_visibility="collapsed",
        )
    with col_f2:
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.markdown('<div class="filter-btn active">권역별</div>', unsafe_allow_html=True)
        with btn_col2:
            st.markdown('<div class="filter-btn">레이어링</div>', unsafe_allow_html=True)
    with col_f3:
        st.markdown(
            '<div style="display:flex;gap:6px;justify-content:flex-end">'
            '<div class="filter-btn">리포트 저장</div>'
            '<div class="filter-btn" style="padding:6px 10px">⊞</div>'
            '<div class="filter-btn" style="padding:6px 10px">≡</div>'
            '</div>',
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

    map_df = cluster_source.copy()

    def get_coords(city_name):
        clean_name = str(city_name).replace("시", "").replace("군", "").strip()
        return kangwon_coords.get(clean_name, (37.8813, 127.7298))

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

    st.markdown('<div class="map-card">', unsafe_allow_html=True)

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
        height=580,
        margin=dict(t=0, b=0, l=0, r=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with sub_tabs[2]:
    st.markdown("#### 맞춤형 처방")

    st.info("선택한 집단의 체력 수준에 맞는 운동 프로그램을 확인할 수 있습니다.")

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
                    <div class="prescription-card">
                        <span class="prescription-tag {tag_class}">{label}</span>
                        <h4>{label} 처방 가이드</h4>
                        <p><b>학교 수</b> {len(subset)}개교</p>
                        <p><b>{x_ax} 평균</b> {subset[raw_x].mean():.1f} · <b>{y_ax} 평균</b> {subset[raw_y].mean():.1f}</p>
                        <p><b>{title_1}</b><br>{body_1}</p>
                        <p><b>{title_2}</b><br>{body_2}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
