import json
import os
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

:root {
    color-scheme: light;
}

html, body, [class*="css"], .stApp {
    font-family: 'Noto Sans KR', sans-serif;
    background-color: #ffffff !important;
    color: #111827 !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
    background-color: #ffffff !important;
}

#MainMenu, footer {
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

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #1a2233 !important;
}

.sidebar-logo {
    background: #ecfdf5;
    color: #0f766e !important;
    border-bottom: 1px solid #d1fae5;
    padding: 13px 16px;
    min-height: 42px;
    display: flex;
    align-items: center;
    font-size: 15px;
    font-weight: 900;
    line-height: 1.25;
    letter-spacing: 0;
    margin-bottom: 10px;
    white-space: nowrap;
}

.sidebar-group {
    margin-top: 9px;
    padding: 0 10px;
}

.sidebar-section {
    font-size: 14.5px;
    font-weight: 900;
    color: #344054 !important;
    letter-spacing: -0.01em;
    margin: 10px 0 10px 0;
    line-height: 1.35;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1440px !important;
}

.top-header {
    background: #ffffff;
    border-bottom: 1px solid #e8eaf0;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    border-radius: 10px;
}

.top-header .breadcrumb-nav {
    font-size: 13px;
    color: #667085;
}

.top-header .breadcrumb-nav b {
    color: #374151;
}

.breadcrumb {
    font-size: 12px;
    color: #667085;
    margin-bottom: 6px;
}

.breadcrumb span {
    color: #344054;
    font-weight: 500;
}

.page-title {
    font-size: 26px;
    font-weight: 800;
    color: #0f766e !important;
    margin-bottom: 18px;
    letter-spacing: -0.01em;
}

.notice-card {
    background: linear-gradient(135deg, #f0fdfa 0%, #f8fafc 100%);
    border: 1px solid #ccfbf1;
    border-radius: 12px;
    padding: 16px 18px;
    color: #344054;
    font-size: 13px;
    line-height: 1.8;
    margin-bottom: 22px;
}

.map-card {
    background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%);
    border-radius: 14px;
    padding: 4px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    overflow: hidden;
    border: 1px solid #dbeafe;
}

.report-card {
    background: linear-gradient(135deg, #ffffff 0%, #f7f3ff 100%);
    border: 1px solid #e9d5ff;
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
    color: #344054;
}

.phone-card {
    background: linear-gradient(180deg, #fff8ef 0%, #ffffff 100%);
    border: 1px solid #ece7dd;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 14px 28px rgba(0,0,0,0.06);
    height: 100%;
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
    background: linear-gradient(135deg, #f8fbff 0%, #eefdf7 100%);
    border: 1px solid #d7f3e8;
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

[data-testid="stMetricLabel"] {
    color: #344054 !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    color: #1d4ed8 !important;
    font-weight: 800 !important;
}

[data-testid="stCaptionContainer"] {
    color: #4b5563 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #1f2937;
}

[data-testid="stButton"] button {
    width: 100%;
    text-align: left;
    border: none;
    background: transparent;
    color: #344054;
    padding: 2px 0 !important;
    min-height: 24px !important;
    height: auto !important;
    font-size: 13px !important;
    font-weight: 500;
    line-height: 1.28 !important;
    box-shadow: none;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
}

[data-testid="stSidebar"] .stButton,
[data-testid="stSidebar"] [data-testid="stButton"] {
    margin: 0 0 4px 0 !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 3px !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] button p,
[data-testid="stSidebar"] [data-testid="stButton"] button div,
[data-testid="stSidebar"] [data-testid="stButton"] button span {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 13px !important;
    line-height: 1.28 !important;
    color: #344054 !important;
    white-space: normal !important;
}

[data-testid="stButton"] button:hover {
    border: none;
    background: transparent;
    color: #1d4ed8;
    text-decoration: underline;
}

.menu-active {
    color: #1d4ed8;
    padding: 2px 0;
    min-height: 24px;
    display: flex;
    align-items: center;
    font-size: 13px;
    font-weight: 800;
    line-height: 1.28;
    margin-bottom: 4px;
}

.section-space {
    height: 14px;
}

.insight-card {
    background: linear-gradient(135deg, #fffdf5 0%, #f8fafc 100%);
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    height: 100%;
}

.insight-card h4 {
    margin: 0 0 8px 0;
    color: #111827;
    font-size: 15px;
    font-weight: 800;
}

.insight-card p {
    margin: 0;
    color: #344054;
    font-size: 13px;
    line-height: 1.7;
}

.alert-card {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-left: 4px solid #f97316;
    border-radius: 8px;
    padding: 14px 16px;
    color: #7c2d12;
    font-size: 14px;
    line-height: 1.7;
    margin: 12px 0 18px 0;
}

.program-tag {
    display: inline-block;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 800;
    margin-right: 6px;
}

.tag-priority { background: #fee2e2; color: #991b1b; }
.tag-elite { background: #dbeafe; color: #1e40af; }
.tag-normal { background: #dcfce7; color: #166534; }

.mobile-frame {
    max-width: 390px;
    margin: 0 auto;
    background: #fffaf1;
    border: 10px solid #111827;
    border-radius: 34px;
    padding: 22px 18px;
    box-shadow: 0 24px 45px rgba(15, 23, 42, 0.18);
}

.mission-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: #ffffff;
    border: 1px solid #f1e5d3;
    border-radius: 10px;
    padding: 12px;
    margin-top: 10px;
    color: #344054;
    font-size: 13px;
}

.fitt-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin: 8px 0 14px 0;
}

.fitt-card {
    background: linear-gradient(135deg, #f8fbff 0%, #eefdf7 100%);
    border: 1px solid #d7f3e8;
    border-radius: 14px;
    padding: 10px 12px;
    min-height: 78px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

.fitt-key {
    color: #344054;
    font-size: 11px;
    font-weight: 900;
    margin-bottom: 5px;
}

.fitt-value {
    color: #1d4ed8;
    font-size: 12px;
    font-weight: 800;
    line-height: 1.35;
    word-break: keep-all;
    overflow-wrap: anywhere;
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
            if any(keyword.lower() in name.lower() for keyword in keywords):
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
    lat_col = find_col(["위도", "latitude", "lat", "Y좌표", "y좌표"])
    lon_col = find_col(["경도", "longitude", "lon", "X좌표", "x좌표"])
    fitness_grade_col = find_col(["체력등급", "종합등급", "PAPS등급", "건강체력등급", "평가등급"])
    school_level_col = find_col(["학교급", "학교급명", "학교구분", "학교 구분", "학교유형", "학교 유형"])

    def normalize_school_level(value):
        text = str(value).strip()
        if text in ["고", "고등", "고등학교"] or "고등" in text or text.endswith("고"):
            return "고"
        if text in ["중", "중학교"] or "중학교" in text or text.endswith("중"):
            return "중"
        return "미상"

    df["연도"] = (
        pd.to_numeric(df[year_col], errors="coerce").fillna(0).astype(int)
        if year_col
        else 0
    )
    df["시군"] = find_first(["시군"], pd.Series(["미상"] * len(df), index=df.index))
    df["성별"] = find_first(["성별", "남여"], pd.Series(["전체"] * len(df), index=df.index))
    df["학년"] = find_first(["학년"], pd.Series(["전체"] * len(df), index=df.index))
    school_level_source = (
        df[school_level_col].astype(str).str.strip()
        if school_level_col
        else df["순수학교명"].astype(str)
    )
    df["학교급"] = school_level_source.apply(normalize_school_level)
    df["체력등급"] = df[fitness_grade_col].astype(str).str.strip() if fitness_grade_col else pd.NA
    df["위도"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else pd.NA
    df["경도"] = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else pd.NA
    df = df.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    return df, {"valid": valid_targets, "file_path": file_path, "grade_col": "체력등급"}, None


def apply_filters(df, years, regions, school_levels, grades, genders, schools):
    filtered_df = df.copy()
    if years:
        filtered_df = filtered_df[filtered_df["연도"].isin(years)]
    if regions:
        filtered_df = filtered_df[filtered_df["시군"].isin(regions)]
    if school_levels and "학교급" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["학교급"].isin(school_levels)]
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


def clamp_score(value, lower=0, upper=100):
    return max(lower, min(upper, value))


def classify_student_profile(height_cm, weight_kg, shuttle_runs, strength_score, flexibility_cm, power_cm):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    height_factor = (height_m / 1.65) ** 0.28
    weight_factor = (58 / weight_kg) ** 0.33

    cardio_score = clamp_score((shuttle_runs * height_factor * weight_factor) / 80 * 100)
    strength_adjusted = strength_score * (58 / weight_kg) ** 0.45
    strength_index = clamp_score(strength_adjusted / 55 * 100)
    flexibility_index = clamp_score((flexibility_cm + 5) / 35 * 100)
    power_adjusted = power_cm * (height_cm / 165) ** 0.20 * (58 / weight_kg) ** 0.20
    power_index = clamp_score(power_adjusted / 220 * 100)
    bmi_stability = clamp_score(100 - abs(bmi - 21.5) * 8)
    allometric_index = round(
        cardio_score * 0.34
        + strength_index * 0.22
        + flexibility_index * 0.16
        + power_index * 0.18
        + bmi_stability * 0.10,
        1,
    )

    component_scores = {
        "BMI 안정성": round(bmi_stability, 1),
        "심폐지구력": round(cardio_score, 1),
        "근력/근지구력": round(strength_index, 1),
        "유연성": round(flexibility_index, 1),
        "순발력": round(power_index, 1),
    }

    centroids = pd.DataFrame(
        [
            {"label": "고위험군", "bmi": 29.0, "allometric": 35},
            {"label": "중점관리군", "bmi": 25.0, "allometric": 52},
            {"label": "일반군", "bmi": 22.0, "allometric": 68},
            {"label": "우수군", "bmi": 19.5, "allometric": 84},
        ]
    )
    distances = (
        ((centroids["bmi"] - bmi) * 1.4) ** 2
        + ((centroids["allometric"] - allometric_index) * 0.18) ** 2
    ) ** 0.5
    matched = centroids.loc[distances.idxmin()]
    return bmi, allometric_index, matched["label"], component_scores


@st.cache_data(show_spinner=False)
def geocode_school_location(school_name, region_name):
    queries = [
        f"{school_name}, {region_name}, 강원특별자치도, 대한민국",
        f"{school_name}, 강원특별자치도, 대한민국",
    ]
    for query in queries:
        try:
            params = urlencode({"q": query, "format": "jsonv2", "limit": 1})
            request = Request(
                f"https://nominatim.openstreetmap.org/search?{params}",
                headers={"User-Agent": "PAPS-Care-Streamlit-Dashboard/1.0"},
            )
            with urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload:
                return float(payload[0]["lat"]), float(payload[0]["lon"])
        except Exception:
            continue
    return None, None


def default_filter_state():
    return {
        "years": [],
        "regions": [],
        "school_levels": [],
        "grades": [],
        "genders": [],
        "schools": [],
        "x_ax": "BMI",
        "y_ax": "심폐지구력",
        "clusters": 4,
    }


def render_filter_controls(
    df,
    meta,
    key_prefix,
    include_axis=False,
    include_cluster=False,
    fields=None,
):
    if f"{key_prefix}_filters" not in st.session_state:
        st.session_state[f"{key_prefix}_filters"] = default_filter_state()

    state = st.session_state[f"{key_prefix}_filters"]
    metric_options = list(meta["valid"].keys())
    if state["x_ax"] not in metric_options:
        state["x_ax"] = metric_options[0]
    if state["y_ax"] not in metric_options:
        state["y_ax"] = metric_options[1 if len(metric_options) > 1 else 0]

    fields = list(fields or ["years", "regions", "school_levels", "grades", "genders", "schools"])
    years = []
    regions = []
    school_levels = []
    grades = []
    genders = []
    schools = []

    filter_widgets = []
    if "years" in fields:
        filter_widgets.append("years")
    if "regions" in fields:
        filter_widgets.append("regions")
    if "school_levels" in fields:
        filter_widgets.append("school_levels")
    if "grades" in fields:
        filter_widgets.append("grades")
    if "genders" in fields:
        filter_widgets.append("genders")
    if "schools" in fields:
        filter_widgets.append("schools")

    if filter_widgets:
        row1 = st.columns(len(filter_widgets))
        for column, widget in zip(row1, filter_widgets):
            with column:
                if widget == "years":
                    year_options = sorted(df["연도"].dropna().unique())
                    years = st.multiselect(
                        "연도",
                        year_options,
                        default=[value for value in state["years"] if value in year_options],
                        key=f"{key_prefix}_years",
                    )
                elif widget == "regions":
                    region_options = sorted(df["시군"].dropna().unique())
                    regions = st.multiselect(
                        "시·군",
                        region_options,
                        default=[value for value in state["regions"] if value in region_options],
                        key=f"{key_prefix}_regions",
                    )
                elif widget == "school_levels":
                    level_options = [value for value in ["중", "고"] if value in set(df["학교급"].dropna().unique())]
                    if not level_options:
                        level_options = ["중", "고"]
                    school_levels = st.multiselect(
                        "학교급",
                        level_options,
                        default=[value for value in state.get("school_levels", []) if value in level_options],
                        key=f"{key_prefix}_school_levels",
                    )
                elif widget == "grades":
                    grade_options = sorted(df["학년"].dropna().unique())
                    grades = st.multiselect(
                        "학년",
                        grade_options,
                        default=[value for value in state["grades"] if value in grade_options],
                        key=f"{key_prefix}_grades",
                    )
                elif widget == "genders":
                    gender_options = sorted(df["성별"].dropna().unique())
                    genders = st.multiselect(
                        "성별",
                        gender_options,
                        default=[value for value in state["genders"] if value in gender_options],
                        key=f"{key_prefix}_genders",
                    )
                elif widget == "schools":
                    school_base_df = apply_filters(df, years, regions, school_levels, grades, genders, [])
                    school_options = sorted(school_base_df["순수학교명"].dropna().unique())
                    schools = st.multiselect(
                        "학교",
                        school_options,
                        default=[v for v in state["schools"] if v in school_options],
                        key=f"{key_prefix}_schools",
                    )

    x_ax = state["x_ax"]
    y_ax = state["y_ax"]
    n_cl = 4
    if include_axis:
        axis_cols = [1.2, 1.2, 0.8] if include_cluster else [1, 1]
        row2 = st.columns(axis_cols)
        with row2[0]:
            x_ax = st.selectbox("수평축", metric_options, index=metric_options.index(state["x_ax"]), key=f"{key_prefix}_x")
        with row2[1]:
            y_ax = st.selectbox("수직축", metric_options, index=metric_options.index(state["y_ax"]), key=f"{key_prefix}_y")
        if include_cluster:
            with row2[2]:
                n_cl = st.slider("군집 수", 2, 4, state["clusters"], key=f"{key_prefix}_clusters")

    state.update(
        {
            "years": years,
            "regions": regions,
            "school_levels": school_levels,
            "grades": grades,
            "genders": genders,
            "schools": schools,
            "x_ax": x_ax,
            "y_ax": y_ax,
            "clusters": n_cl,
        }
    )
    return state


def build_clustered_view(df, meta, filters):
    filtered_df = apply_filters(
        df,
        filters["years"],
        filters["regions"],
        filters.get("school_levels", []),
        filters["grades"],
        filters["genders"],
        filters["schools"],
    )
    if filtered_df.empty:
        return None, "선택한 조건에 맞는 데이터가 없습니다. 필터를 조정해 주세요."

    group_cols = ["순수학교명", "연도", "시군", "학교급", "학년", "성별"]
    agg_map = {column: "mean" for column in meta["valid"].values()}
    if "위도" in filtered_df.columns:
        agg_map["위도"] = "mean"
    if "경도" in filtered_df.columns:
        agg_map["경도"] = "mean"
    df_agg = filtered_df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()

    raw_x = meta["valid"][filters["x_ax"]]
    raw_y = meta["valid"][filters["y_ax"]]
    cluster_source = df_agg.dropna(subset=[raw_x, raw_y]).copy()

    if len(cluster_source) < filters["clusters"]:
        return None, f"현재 조건에서는 군집 {filters['clusters']}개를 만들 데이터가 부족합니다."

    scaled_points = StandardScaler().fit_transform(cluster_source[[raw_x, raw_y]])
    kmeans = KMeans(n_clusters=filters["clusters"], random_state=42, n_init=10)
    cluster_source["Cluster"] = kmeans.fit_predict(scaled_points)
    cluster_summary = cluster_source.groupby("Cluster")[[raw_x, raw_y]].mean()
    cluster_summary["score"] = cluster_summary.mean(axis=1)
    cluster_labels = build_cluster_labels(cluster_summary, filters["x_ax"])
    cluster_source["유형"] = cluster_source["Cluster"].map(cluster_labels)

    return {
        "filtered_df": filtered_df,
        "cluster_source": cluster_source,
        "raw_x": raw_x,
        "raw_y": raw_y,
        "x_ax": filters["x_ax"],
        "y_ax": filters["y_ax"],
    }, None


def build_map_df(cluster_source):
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

    def stable_jitter(seed_text):
        seed = sum(ord(ch) for ch in str(seed_text))
        lat_offset = ((seed % 17) - 8) * 0.0065
        lon_offset = (((seed // 17) % 17) - 8) * 0.0085
        return lat_offset, lon_offset

    map_df = cluster_source.copy().reset_index(drop=True)
    has_real_coords = (
        "위도" in map_df.columns
        and "경도" in map_df.columns
        and map_df["위도"].notna().any()
        and map_df["경도"].notna().any()
    )

    if has_real_coords:
        map_df["lat"] = pd.to_numeric(map_df["위도"], errors="coerce")
        map_df["lon"] = pd.to_numeric(map_df["경도"], errors="coerce")
        missing_mask = map_df["lat"].isna() | map_df["lon"].isna()
        if missing_mask.any():
            fallback_coords = map_df.loc[missing_mask, "시군"].apply(get_coords)
            map_df.loc[missing_mask, "lat"] = fallback_coords.apply(lambda value: value[0]).values
            map_df.loc[missing_mask, "lon"] = fallback_coords.apply(lambda value: value[1]).values
    else:
        base_coords = map_df["시군"].apply(get_coords)
        map_df["base_lat"] = base_coords.apply(lambda value: value[0])
        map_df["base_lon"] = base_coords.apply(lambda value: value[1])
        map_df["lat"] = pd.NA
        map_df["lon"] = pd.NA

        unique_schools = map_df[["순수학교명", "시군"]].drop_duplicates().reset_index(drop=True)
        geocoded = {}
        if len(unique_schools) <= 12:
            for _, school_row in unique_schools.iterrows():
                school_name = str(school_row["순수학교명"])
                region_name = str(school_row["시군"])
                geocoded[(school_name, region_name)] = geocode_school_location(school_name, region_name)
                time.sleep(0.05)

        for idx, row in map_df.iterrows():
            key = (str(row["순수학교명"]), str(row["시군"]))
            lat, lon = geocoded.get(key, (None, None))
            if lat is not None and lon is not None:
                map_df.at[idx, "lat"] = lat
                map_df.at[idx, "lon"] = lon

        missing_mask = map_df["lat"].isna() | map_df["lon"].isna()
        if missing_mask.any():
            jitter_seed = map_df.loc[missing_mask, "순수학교명"].astype(str) + "_" + map_df.loc[missing_mask].index.astype(str)
            jitter = jitter_seed.apply(stable_jitter)
            map_df.loc[missing_mask, "lat"] = map_df.loc[missing_mask, "base_lat"] + jitter.apply(lambda value: value[0]).values
            map_df.loc[missing_mask, "lon"] = map_df.loc[missing_mask, "base_lon"] + jitter.apply(lambda value: value[1]).values

    weight_map = {
        "고위험군": 10,
        "관리 필요군": 8,
        "중점관리군": 5,
        "일반군": 1,
        "건강 양호군": 0.5,
        "우수군": 0.1,
    }
    map_df["weight"] = map_df["유형"].map(weight_map).fillna(1)
    map_df["긴급요인"] = map_df["유형"].map(
        {
            "고위험군": "비만율·심폐지구력 동시 관리 필요",
            "관리 필요군": "저체력 비율 상승",
            "중점관리군": "심폐지구력 개선 우선",
            "일반군": "균형 유지",
            "건강 양호군": "현 수준 유지",
            "우수군": "심화 프로그램 연계",
        }
    ).fillna("추가 확인 필요")
    return map_df


def plot_theme_colors():
    # 발표 화면에서는 배경을 흰색으로 고정하고, 축/라벨은 진한 색으로 유지합니다.
    return {
        "paper": "#ffffff",
        "plot": "#ffffff",
        "text": "#111827",
        "muted": "#344054",
        "grid": "rgba(15, 23, 42, 0.18)",
        "axis": "#1f2937",
        "card": "rgba(255, 255, 255, 0.96)",
        "border": "rgba(210, 215, 225, 0.9)",
    }


def apply_plotly_theme(fig, height=None, margin=None, legend=None):
    colors = plot_theme_colors()
    layout = {
        "paper_bgcolor": colors["paper"],
        "plot_bgcolor": colors["plot"],
        "font": dict(color=colors["text"], family="Noto Sans KR", size=13),
        "title_font": dict(color=colors["text"], family="Noto Sans KR", size=18),
        "hoverlabel": dict(
            bgcolor="#ffffff",
            bordercolor="#1f2937",
            font=dict(color=colors["text"], family="Noto Sans KR", size=13),
        ),
    }
    if height is not None:
        layout["height"] = height
    if margin is not None:
        layout["margin"] = margin
    if legend is not None:
        layout["legend"] = legend
    fig.update_layout(**layout)
    return fig


def apply_readable_axes(fig, height=None, margin=None, legend=None):
    colors = plot_theme_colors()
    apply_plotly_theme(fig, height=height, margin=margin, legend=legend)
    fig.update_xaxes(
        showgrid=True,
        gridcolor=colors["grid"],
        zeroline=False,
        showline=True,
        linecolor=colors["axis"],
        linewidth=2,
        ticks="outside",
        tickcolor=colors["axis"],
        tickwidth=1.6,
        tickfont=dict(color=colors["text"], size=13),
        title_font=dict(color=colors["text"], size=15),
        title_standoff=12,
        mirror=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=colors["grid"],
        zeroline=False,
        showline=True,
        linecolor=colors["axis"],
        linewidth=2,
        ticks="outside",
        tickcolor=colors["axis"],
        tickwidth=1.6,
        tickfont=dict(color=colors["text"], size=13),
        title_font=dict(color=colors["text"], size=15),
        title_standoff=12,
        mirror=True,
    )
    return fig


def render_heatmap(cluster_source):
    map_df = build_map_df(cluster_source)
    colors = plot_theme_colors()
    map_style = "carto-darkmatter" if colors["paper"] != "#ffffff" else "carto-positron"
    fig = px.density_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        z="weight",
        radius=42,
        center=dict(lat=37.75, lon=128.3),
        zoom=6.85,
        mapbox_style=map_style,
        hover_name="시군",
        hover_data={
            "순수학교명": True,
            "유형": True,
            "긴급요인": True,
            "lat": False,
            "lon": False,
            "weight": False,
        },
        color_continuous_scale=[
            [0.00, "rgba(44,62,140,0.00)"],
            [0.22, "rgba(52,77,181,0.55)"],
            [0.48, "rgba(71,95,204,0.78)"],
            [0.72, "rgba(232,125,56,0.88)"],
            [1.00, "rgba(255,186,111,1.00)"],
        ],
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=map_df["lat"],
            lon=map_df["lon"],
            mode="markers",
            marker=dict(
                size=6,
                color=map_df["weight"],
                colorscale=[
                    [0.00, "#233a8b"],
                    [0.33, "#475fcc"],
                    [0.66, "#e87d38"],
                    [1.00, "#ffba6f"],
                ],
                opacity=0.28,
                showscale=False,
            ),
            customdata=map_df[["시군", "순수학교명", "유형", "긴급요인"]].to_numpy(),
            hovertemplate=(
                "시군: %{customdata[0]}<br>"
                "학교: %{customdata[1]}<br>"
                "집단: %{customdata[2]}<br>"
                "긴급요인: %{customdata[3]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=0.98,
        y=0.96,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        text="<b>관측치 기반 취약 체력 밀집도</b>",
        showarrow=False,
        align="left",
        bgcolor=colors["card"],
        bordercolor=colors["border"],
        borderwidth=1,
        borderpad=10,
        font=dict(size=12, color=colors["text"], family="Noto Sans KR"),
    )
    fig.add_annotation(
        x=0.98,
        y=0.89,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        text=(
            "<span style='color:#ffba6f'>매우 높음</span><br>"
            "<span style='color:#e87d38'>높음</span><br>"
            "<span style='color:#475fcc'>보통</span><br>"
            "<span style='color:#233a8b'>낮음</span>"
        ),
        showarrow=False,
        align="left",
        bgcolor=colors["card"],
        bordercolor=colors["border"],
        borderwidth=1,
        borderpad=10,
        font=dict(size=11, color=colors["text"], family="Noto Sans KR"),
    )
    apply_plotly_theme(fig, height=620, margin=dict(t=0, b=0, l=0, r=0))
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def render_scatter(cluster_source, raw_x, raw_y, x_ax, y_ax):
    colors = plot_theme_colors()
    fig = px.scatter(
        cluster_source,
        x=raw_x,
        y=raw_y,
        color="유형",
        text="순수학교명",
        hover_data={"연도": True, "시군": True, "학년": True, "성별": True},
        labels={raw_x: x_ax, raw_y: y_ax, "유형": "집단"},
        color_discrete_map={
            "고위험군": "#d44b57",
            "관리 필요군": "#d44b57",
            "중점관리군": "#ef8b2c",
            "일반군": "#1c9d74",
            "우수군": "#2574ea",
            "건강 양호군": "#2574ea",
        },
    )
    fig.update_traces(
        marker=dict(size=16, opacity=0.88, line=dict(width=1.1, color="white")),
        textposition="top center",
        textfont=dict(size=10, color=colors["text"]),
    )
    apply_readable_axes(
        fig,
        height=620,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=colors["text"]),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def cluster_color_map():
    return {
        "고위험군": "#d44b57",
        "관리 필요군": "#d44b57",
        "중점관리군": "#ef8b2c",
        "일반군": "#1c9d74",
        "우수군": "#2574ea",
        "건강 양호군": "#2574ea",
    }


def render_cluster_pie(cluster_source):
    colors = plot_theme_colors()
    share_df = cluster_source["유형"].value_counts().rename_axis("유형").reset_index(name="학교 수")
    fig = px.pie(
        share_df,
        names="유형",
        values="학교 수",
        hole=0.64,
        color="유형",
        color_discrete_map=cluster_color_map(),
    )
    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        textfont=dict(size=12, color=colors["text"]),
        marker=dict(line=dict(color=colors["paper"], width=3)),
        pull=[0.04 if label in ["고위험군", "관리 필요군"] else 0 for label in share_df["유형"]],
    )
    fig.add_annotation(
        text="<b>체력군</b><br>구성 비율",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=15, color=colors["text"], family="Noto Sans KR"),
        align="center",
    )
    apply_plotly_theme(
        fig,
        height=430,
        margin=dict(t=24, b=24, l=24, r=24),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color=colors["muted"]),
        ),
    )
    fig.update_layout(showlegend=True, uniformtext_minsize=10, uniformtext_mode="hide")
    st.plotly_chart(fig, use_container_width=True)


def risk_mask(series):
    return series.isin(["고위험군", "관리 필요군", "중점관리군"])


def calculate_grade_45_share(df, meta):
    grade_col = meta.get("grade_col")
    if not grade_col or grade_col not in df.columns:
        return None

    extracted = df[grade_col].astype(str).str.extract(r"([1-5])")[0]
    grade_numbers = pd.to_numeric(extracted, errors="coerce").dropna()
    if grade_numbers.empty:
        return None
    return round(grade_numbers.isin([4, 5]).mean() * 100, 1)


def render_yearly_trend(cluster_source):
    valid_trend_source = cluster_source[
        (cluster_source["연도"] >= 2010)
        & (cluster_source["연도"] <= 2025)
    ].copy()

    if valid_trend_source.empty:
        st.info("차트를 그릴 수 있는 유효한 연도 데이터가 없습니다.")
        return

    trend_df = (
        valid_trend_source.assign(취약군=risk_mask(valid_trend_source["유형"]))
        .groupby("연도")
        .agg(분석학교수=("순수학교명", "nunique"), 취약군비율=("취약군", "mean"))
        .reset_index()
    )
    trend_df["취약군비율"] = (trend_df["취약군비율"] * 100).round(1)
    trend_df["연도_라벨"] = trend_df["연도"].astype(int).astype(str)
    fig = px.line(
        trend_df,
        x="연도_라벨",
        y="취약군비율",
        markers=True,
        title="연도별 저체력·취약군 비율 변화",
        labels={"취약군비율": "취약군 비율(%)", "연도_라벨": "연도"},
    )
    fig.update_traces(line=dict(color="#0f766e", width=3), marker=dict(size=9))
    apply_readable_axes(fig, height=360, margin=dict(t=50, b=24, l=24, r=24))
    st.plotly_chart(fig, use_container_width=True)


def render_download_table(df, columns, label="분석 데이터 다운로드"):
    export_df = df[columns].copy()
    st.download_button(
        label,
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="paps_care_export.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.dataframe(export_df.head(50), use_container_width=True)


def set_page(page_name):
    st.session_state["current_page"] = page_name


def render_nav_button(page_name, label=None):
    display_name = label or page_name
    if st.session_state.get("current_page") == page_name:
        st.markdown(f'<div class="menu-active">{display_name}</div>', unsafe_allow_html=True)
    else:
        st.button(display_name, key=f"nav_{page_name}", on_click=set_page, args=(page_name,))


raw_df, meta, load_error = load_raw_data()
if load_error:
    st.error(load_error)
    st.stop()

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "강원특별자치도 체력 현황 요약"
elif st.session_state["current_page"] == "도내 체력 현황 요약":
    st.session_state["current_page"] = "강원특별자치도 체력 현황 요약"


with st.sidebar:
    st.markdown('<div class="sidebar-logo">🏃 PAPS CARE+</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-group">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">1. 📊 통합 대시보드</div>', unsafe_allow_html=True)
    render_nav_button("강원특별자치도 체력 현황 요약", "📌 강원특별자치도 체력 현황 요약")
    render_nav_button("체력 취약망 지도 (Heatmap)", "🗺️ 체력 취약망 지도")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">2. 🤖 AI 체육 데이터 분석</div>', unsafe_allow_html=True)
    render_nav_button("체격 보정 평가 모델 (Allometric)", "⚖️ 체격 보정 평가 모델")
    render_nav_button("AI 다차원 군집 분석", "🧬 AI 다차원 군집 분석")
    render_nav_button("종목/학년별 상세 통계", "📈 종목/학년별 상세 통계")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">3. 🏃‍♂️ 맞춤형 체력 증진</div>', unsafe_allow_html=True)
    render_nav_button("집단별 FITT 처방", "🧭 집단별 FITT 처방")
    render_nav_button("학교별 교육 프로그램 추천", "🎒 학교별 프로그램 추천")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">4. 🏢 행정·정책 지원</div>', unsafe_allow_html=True)
    render_nav_button("체육 강사 우선 배치망", "🏫 체육 강사 우선 배치망")
    render_nav_button("지역별 예산 집행 타당성", "💰 지역별 예산 타당성")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">5. 📱 학생/학부모 서비스</div>', unsafe_allow_html=True)
    render_nav_button("나의 AI 체력 진단", "🧑‍🎓 나의 AI 체력 진단")
    render_nav_button("4주 맞춤 운동 플랜 발급", "🗓️ 4주 운동 플랜 발급")
    st.markdown('</div>', unsafe_allow_html=True)


current_page = st.session_state["current_page"]

st.markdown(
    """
    <div class="top-header">
        <div class="breadcrumb-nav">교육행정 &gt; <b>PAPS CARE+ Intelligence</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="breadcrumb">교육행정 &gt; <span>{current_page}</span></div>
    <div class="page-title">🏃 PAPS CARE+</div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice-card">
        <b>PAPS CARE+ Intelligence</b><br>
        본 서비스는 학교알리미 교육 공공데이터와 국민체력100 실측 데이터를 활용하여 학교 및 지역 단위의 체력 현황을 분석하는 AI 기반 지능형 대시보드이다. 국민체력100 데이터를 기반으로 알로메트릭 스케일링을 적용해 체격 요인에 따른 점수 편향을 보정하고, 강원특별자치도 내 체력 취약 영역 진단, 맞춤형 FITT 운동 처방, 체육 교육 개선 및 정책 의사결정에 활용 가능한 정보를 제공한다.
    </div>
    """,
    unsafe_allow_html=True,
)


def render_overview():
    metric_options = list(meta["valid"].keys())
    filters = default_filter_state()
    filters["x_ax"] = "BMI" if "BMI" in metric_options else metric_options[0]
    filters["y_ax"] = "심폐지구력" if "심폐지구력" in metric_options else metric_options[1 if len(metric_options) > 1 else 0]
    filters["clusters"] = 4
    result, error = build_clustered_view(raw_df, meta, filters)
    if error:
        st.warning(error)
        return

    cluster_source = result["cluster_source"]
    dominant_group = cluster_source["유형"].value_counts().idxmax()
    dominant_share = round((cluster_source["유형"].value_counts().max() / len(cluster_source)) * 100, 1)
    high_risk = len(cluster_source[cluster_source["유형"].isin(["고위험군", "관리 필요군"])])
    weak_share = round((high_risk / len(cluster_source)) * 100, 1)
    grade_45_share = calculate_grade_45_share(result["filtered_df"], meta)
    grade_45_value = f"{grade_45_share}%" if grade_45_share is not None else "자료 없음"
    grade_45_help = "PAPS 등급 기준" if grade_45_share is not None else "등급 컬럼 필요"
    student_estimate = len(result["filtered_df"])
    high_risk_estimate = int(student_estimate * weak_share / 100)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("분석 대상", f"{cluster_source['순수학교명'].nunique()}개교", f"{student_estimate:,}건")
    with c2:
        st.metric("4·5등급 비율", grade_45_value, grade_45_help)
    with c3:
        st.metric("AI 취약군 비율", f"{weak_share}%", "고위험·관리필요군")
    with c4:
        st.metric("AI 고위험군 추정치", f"{high_risk_estimate:,}명", "집중 관리 대상")
    with c5:
        st.metric("주요 집단", dominant_group, f"{dominant_share}%")

    st.markdown("---")
    left, right = st.columns([1.4, 0.8])
    with left:
        render_yearly_trend(cluster_source)
    with right:
        st.markdown(
            """
            <div class="insight-card">
                <h4>운영 메모</h4>
                <p>오늘 기준 도내 취약 체력 신호를 요약한 화면입니다. 취약군 비율과 AI 고위험군 추정치를 함께 확인하고, 지도·처방·행정 지원 메뉴로 바로 이어지는 의사결정 출발점으로 활용할 수 있습니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_heatmap_page():
    filters = render_filter_controls(
        raw_df,
        meta,
        "heatmap",
        fields=["years", "regions", "school_levels", "grades", "genders"],
    )
    result, error = build_clustered_view(raw_df, meta, filters)
    if error:
        st.warning(error)
        return

    st.markdown("#### 체력 취약망 지도")
    st.caption("취약군이 몰린 지점을 관측치 기반 밀도로 표시하고, 마우스 오버 시 학교·집단·긴급요인을 확인합니다.")
    map_col, info_col = st.columns([3.2, 1])
    with map_col:
        st.markdown('<div class="map-card">', unsafe_allow_html=True)
        render_heatmap(result["cluster_source"])
        st.markdown("</div>", unsafe_allow_html=True)
    with info_col:
        urgent_df = (
            result["cluster_source"]
            .assign(취약군=risk_mask(result["cluster_source"]["유형"]))
            .groupby("시군")
            .agg(취약관측치=("취약군", "sum"), 전체관측치=("순수학교명", "count"))
            .reset_index()
        )
        urgent_df["취약비율"] = (urgent_df["취약관측치"] / urgent_df["전체관측치"] * 100).round(1)
        top_region = urgent_df.sort_values("취약비율", ascending=False).head(1)
        region_name = top_region["시군"].iloc[0] if not top_region.empty else "확인 필요"
        region_rate = top_region["취약비율"].iloc[0] if not top_region.empty else 0
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>우선 집중 지역</h4>
                <p><b>{region_name}</b><br>취약 관측치 비율 {region_rate}%</p>
                <p style="margin-top:12px;">지도에서 붉고 진한 지점일수록 행정 지원과 학교별 프로그램 배정 우선순위가 높습니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<div class='section-space'></div>", unsafe_allow_html=True)
    render_scatter(result["cluster_source"], result["raw_x"], result["raw_y"], result["x_ax"], result["y_ax"])


def render_allometric_page():
    filters = render_filter_controls(
        raw_df,
        meta,
        "allometric",
        fields=["years", "regions", "school_levels", "grades", "genders"],
    )
    metric_options = list(meta["valid"].keys())
    fitness_options = [metric for metric in metric_options if metric != "BMI"] or metric_options
    selected_metric = st.selectbox("보정 대상 체력 요인", fitness_options, key="allometric_target_metric")
    filters["x_ax"] = "BMI" if "BMI" in metric_options else metric_options[0]
    filters["y_ax"] = selected_metric
    filters["clusters"] = 4
    result, error = build_clustered_view(raw_df, meta, filters)
    if error:
        st.warning(error)
        return

    st.markdown("#### 체격 보정 평가 모델 (Allometric)")
    correction_on = st.toggle("AI 보정 필터 켜기", value=True)
    allometric_df = result["cluster_source"].copy()
    allometric_df["체격 보정 기준치"] = (
        allometric_df[result["raw_x"]].abs().fillna(allometric_df[result["raw_x"]].mean()).clip(lower=1)
    )
    allometric_df["보정 체력지표"] = allometric_df[result["raw_y"]] / (allometric_df["체격 보정 기준치"] ** 0.33)
    left, right = st.columns(2)
    with left:
        fig_raw = px.scatter(
            allometric_df,
            x=result["raw_x"],
            y=result["raw_y"],
            color="유형",
            hover_name="순수학교명",
            color_discrete_map=cluster_color_map(),
            title="원점수 기준 분포",
        )
        fig_raw.update_traces(marker=dict(size=13, opacity=0.86, line=dict(width=1.1, color="#ffffff")))
        apply_readable_axes(fig_raw, height=520, margin=dict(t=56, b=28, l=28, r=20))
        fig_raw.update_layout(legend=dict(font=dict(color="#111827", size=12), bgcolor="rgba(255,255,255,0.85)"))
        st.plotly_chart(fig_raw, use_container_width=True)
    with right:
        adjusted_y = "보정 체력지표" if correction_on else result["raw_y"]
        fig_adjusted = px.scatter(
            allometric_df,
            x=result["raw_x"],
            y=adjusted_y,
            color="유형",
            hover_name="순수학교명",
            color_discrete_map=cluster_color_map(),
            title="체격 보정 후 분포" if correction_on else "보정 전 비교 화면",
            labels={result["raw_x"]: result["x_ax"], adjusted_y: "보정 체력지표" if correction_on else result["y_ax"]},
        )
        fig_adjusted.update_traces(marker=dict(size=13, opacity=0.86, line=dict(width=1.1, color="#ffffff")))
        apply_readable_axes(fig_adjusted, height=520, margin=dict(t=56, b=28, l=28, r=20))
        fig_adjusted.update_layout(legend=dict(font=dict(color="#111827", size=12), bgcolor="rgba(255,255,255,0.85)"))
        st.plotly_chart(fig_adjusted, use_container_width=True)
    st.markdown(
        """
        <div class="alert-card">
            AI 보정 필터를 켜면 체격 조건 때문에 원점수에서 불리했던 관측치가 재평가되는 흐름을 확인할 수 있습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cluster_page():
    filters = render_filter_controls(
        raw_df,
        meta,
        "cluster",
        include_axis=True,
        fields=["years", "regions", "school_levels", "grades", "genders", "schools"],
    )
    result, error = build_clustered_view(raw_df, meta, filters)
    if error:
        st.warning(error)
        return
    st.markdown("#### AI 다차원 군집 분석")
    left, right = st.columns([1.45, 0.8])
    with left:
        render_scatter(result["cluster_source"], result["raw_x"], result["raw_y"], result["x_ax"], result["y_ax"])
    with right:
        render_cluster_pie(result["cluster_source"])
        st.markdown(
            """
            <div class="insight-card">
                <h4>군집 해석</h4>
                <p>고위험군, 중점관리군, 일반군, 우수군의 상대적 위치를 함께 보면서 어떤 학교군에 행정 개입이 필요한지 판단합니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_detail_page():
    filters = render_filter_controls(
        raw_df,
        meta,
        "detail",
        fields=["years", "regions", "school_levels", "grades", "genders", "schools"],
    )
    filtered_df = apply_filters(
        raw_df,
        filters["years"],
        filters["regions"],
        filters.get("school_levels", []),
        filters["grades"],
        filters["genders"],
        filters["schools"],
    )
    if filtered_df.empty:
        st.warning("선택한 조건에 맞는 데이터가 없습니다.")
        return
    metric_choice = st.selectbox("상세 지표 선택", list(meta["valid"].keys()))
    metric_col = meta["valid"][metric_choice]
    detail_df = filtered_df.dropna(subset=[metric_col]).copy()
    left, right = st.columns([1, 1])
    with left:
        chart_df = detail_df.groupby(["학년", "성별"])[metric_col].mean().reset_index()
        fig = px.bar(chart_df, x="학년", y=metric_col, color="성별", barmode="group", title=f"{metric_choice} 학년/성별 평균")
        apply_readable_axes(fig, height=500, margin=dict(t=58, b=34, l=36, r=20))
        fig.update_layout(
            legend=dict(
                title_font=dict(color="#111827", size=13),
                font=dict(color="#111827", size=13),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#cbd5e1",
                borderwidth=1,
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        colors = plot_theme_colors()
        school_options = sorted(detail_df["순수학교명"].dropna().unique())
        selected_school = st.selectbox("학교 검색", school_options)
        school_df = detail_df[detail_df["순수학교명"] == selected_school]
        radar_labels = list(meta["valid"].keys())
        radar_values = [school_df[col].mean() for col in meta["valid"].values()]
        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                name=selected_school,
                line=dict(color="#0f766e", width=3),
            )
        )
        apply_plotly_theme(fig_radar, height=500, margin=dict(t=58, b=34, l=34, r=34))
        fig_radar.update_layout(
            title=dict(
                text="학교별 5대 체력 요소 밸런스",
                font=dict(color="#111827", size=18, family="Noto Sans KR"),
            ),
            font=dict(color="#111827", family="Noto Sans KR", size=14),
            polar=dict(
                bgcolor=colors["plot"],
                radialaxis=dict(
                    visible=True,
                    gridcolor=colors["grid"],
                    linecolor=colors["axis"],
                    tickfont=dict(color="#111827", size=13),
                    title=dict(font=dict(color="#111827", size=13)),
                ),
                angularaxis=dict(
                    gridcolor=colors["grid"],
                    linecolor=colors["axis"],
                    tickfont=dict(color="#111827", size=15, family="Noto Sans KR"),
                    linewidth=2,
                ),
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    render_download_table(detail_df, ["순수학교명", "시군", "학년", "성별", metric_col])


def render_prescription_page():
    filters = render_filter_controls(
        raw_df,
        meta,
        "prescription",
        fields=["years", "regions", "school_levels", "grades", "genders", "schools"],
    )
    result, error = build_clustered_view(raw_df, meta, filters)
    if error:
        st.warning(error)
        return

    st.markdown("#### 집단별 5대 요인 기반 맞춤형 FITT 처방")
    st.caption("AI가 판별한 체력 군집과 5대 체력 요인의 불균형을 동시에 고려하여 최적의 운동 처방을 제안합니다.")

    cluster_source = result["cluster_source"]
    row_order = ["고위험군", "관리 필요군", "중점관리군", "일반군", "건강 양호군", "우수군"]
    visible_rows = [label for label in row_order if label in cluster_source["유형"].unique()]
    if not visible_rows:
        st.info("데이터가 부족하여 처방을 생성할 수 없습니다.")
        return

    fitness_cols = {
        "심폐지구력": meta["valid"].get("심폐지구력"),
        "근력/근지구력": meta["valid"].get("근력/근지구력"),
        "유연성": meta["valid"].get("유연성"),
        "순발력": meta["valid"].get("순발력"),
        "BMI(체조성)": meta["valid"].get("BMI"),
    }
    fitness_cols = {name: col for name, col in fitness_cols.items() if col and col in cluster_source.columns}

    fitt_db = {
        "심폐지구력": {
            "F": "주 3~5회",
            "I": "최대심박수의 50~70%",
            "T_time": "30~45분 이상",
            "T_type": "빠르게 걷기·조깅·수영·자전거",
            "title": "심폐 기능 회복 및 지구력 강화",
            "desc": "심장과 폐의 산소 공급 능력을 높이는 것이 최우선입니다. 관절에 무리가 가지 않는 걷기부터 시작해 점진적으로 인터벌 조깅으로 넘어갑니다.",
        },
        "근력/근지구력": {
            "F": "주 2~3회",
            "I": "10~15회 반복 가능한 강도",
            "T_time": "30~40분",
            "T_type": "스쿼트·팔굽혀펴기·밴드 운동",
            "title": "기초 근력 밸런스 확보",
            "desc": "코어 및 큰 근육 위주의 근력 강화가 필요합니다. 맨몸 운동으로 자세를 먼저 잡고, 소도구를 활용해 근지구력을 늘립니다.",
        },
        "유연성": {
            "F": "주 5회 이상",
            "I": "통증 없는 뻐근함",
            "T_time": "15~20분",
            "T_type": "정적 스트레칭·요가·필라테스",
            "title": "관절 가동범위 및 상해 예방",
            "desc": "근육의 긴장을 풀고 가동범위를 늘려 부상을 방지해야 합니다. 햄스트링과 흉추 가동성 스트레칭을 중점적으로 진행합니다.",
        },
        "순발력": {
            "F": "주 2~3회",
            "I": "최대 노력의 80~100%",
            "T_time": "15~20분",
            "T_type": "점프스쿼트·셔틀런·배드민턴",
            "title": "신경근 반응 속도 극대화",
            "desc": "근력을 빠르게 폭발시키는 능력이 요구됩니다. 부상 위험이 있으므로 충분한 웜업 후 점프 훈련과 민첩성 드릴을 수행합니다.",
        },
    }

    def hex_to_rgba(hex_color, alpha=0.24):
        clean = hex_color.lstrip("#")
        if len(clean) != 6:
            return f"rgba(15, 118, 110, {alpha})"
        r, g, b = (int(clean[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"

    def normalize_factor_score(name, column, value):
        series = pd.to_numeric(cluster_source[column], errors="coerce").dropna()
        if series.empty or pd.isna(value):
            return 50
        if name == "BMI(체조성)":
            return clamp_score(100 - abs(value - 21.5) * 8)
        min_value = series.min()
        max_value = series.max()
        if max_value == min_value:
            return 60
        return clamp_score((value - min_value) / (max_value - min_value) * 100)

    def detect_weakest_factor(subset):
        candidate_scores = {}
        for factor_name in ["심폐지구력", "근력/근지구력", "유연성", "순발력"]:
            column = fitness_cols.get(factor_name)
            if not column:
                continue
            all_series = pd.to_numeric(cluster_source[column], errors="coerce").dropna()
            subset_mean = pd.to_numeric(subset[column], errors="coerce").mean()
            std_value = all_series.std()
            if all_series.empty or pd.isna(subset_mean) or std_value == 0 or pd.isna(std_value):
                continue
            candidate_scores[factor_name] = (subset_mean - all_series.mean()) / std_value
        if candidate_scores:
            return min(candidate_scores, key=candidate_scores.get)
        if "위험" in subset["유형"].iloc[0] or "관리 필요" in subset["유형"].iloc[0]:
            return "심폐지구력"
        if "일반" in subset["유형"].iloc[0]:
            return "유연성"
        return "순발력"

    tabs = st.tabs(visible_rows)
    for tab, label in zip(tabs, visible_rows):
        with tab:
            subset = cluster_source[cluster_source["유형"] == label]
            avg_scores = {}
            normalized_scores = {}
            for factor_name, column in fitness_cols.items():
                avg_value = pd.to_numeric(subset[column], errors="coerce").mean()
                if pd.notna(avg_value):
                    avg_scores[factor_name] = avg_value
                    normalized_scores[factor_name] = normalize_factor_score(factor_name, column, avg_value)

            if not normalized_scores:
                st.warning("5대 체력 요인 데이터가 부족합니다.")
                continue

            weakest_factor = detect_weakest_factor(subset)
            fitt = fitt_db.get(weakest_factor, fitt_db["심폐지구력"])
            group_color = cluster_color_map().get(label, "#0f766e")
            col_radar, col_fitt = st.columns([0.8, 1.2])

            with col_radar:
                colors = plot_theme_colors()
                radar_labels = list(normalized_scores.keys())
                radar_values = list(normalized_scores.values())
                fig_radar = go.Figure(
                    go.Scatterpolar(
                        r=radar_values + [radar_values[0]],
                        theta=radar_labels + [radar_labels[0]],
                        fill="toself",
                        name=label,
                        line=dict(color=group_color, width=3),
                        fillcolor=hex_to_rgba(group_color, 0.22),
                    )
                )
                apply_plotly_theme(fig_radar, height=350, margin=dict(t=30, b=30, l=30, r=30))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor=colors["plot"],
                        radialaxis=dict(visible=False, range=[0, 100]),
                        angularaxis=dict(
                            gridcolor=colors["grid"],
                            linecolor=colors["axis"],
                            tickfont=dict(color="#111827", size=13, family="Noto Sans KR"),
                        ),
                    ),
                    showlegend=False,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            with col_fitt:
                st.markdown(
                    f"""
                    <div class="report-card" style="margin-bottom:15px; border-top: 4px solid {group_color};">
                        <span class="report-tag {get_group_style(label)}">{label}</span>
                        <span style="font-size:13px; color:#4b5563; margin-left:10px;">최우선 개선 요인: <b>{weakest_factor}</b></span>
                        <h4 style="margin-top:10px;">{fitt["title"]}</h4>
                        <p>{fitt["desc"]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="fitt-grid">
                        <div class="fitt-card">
                            <div class="fitt-key">F 빈도</div>
                            <div class="fitt-value">{fitt["F"]}</div>
                        </div>
                        <div class="fitt-card">
                            <div class="fitt-key">I 강도</div>
                            <div class="fitt-value">{fitt["I"]}</div>
                        </div>
                        <div class="fitt-card">
                            <div class="fitt-key">T 시간</div>
                            <div class="fitt-value">{fitt["T_time"]}</div>
                        </div>
                        <div class="fitt-card">
                            <div class="fitt-key">T 종류</div>
                            <div class="fitt-value">{fitt["T_type"]}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<h5 style='margin-top:10px; font-size:15px; font-weight:700;'>📅 4주 체육 수업 지도안 (예시)</h5>",
                unsafe_allow_html=True,
            )
            plan_df = pd.DataFrame(
                {
                    "주차": ["1주차", "2주차", "3주차", "4주차"],
                    "수업 목표": ["적응 및 진단", f"{weakest_factor} 기초 다지기", "복합 체력 훈련", "자기 주도 평가"],
                    "지도안 요약": [
                        "기초 스트레칭 및 흥미 위주 게임",
                        f"{fitt['T_type']} 위주의 순환 코스 운영",
                        f"{weakest_factor} 강화와 타 요소(코어) 결합",
                        "개인별 목표 달성 확인 및 그룹 체육 활동",
                    ],
                }
            )
            st.dataframe(plan_df, use_container_width=True, hide_index=True)

            raw_summary = pd.DataFrame(
                {
                    "체력 요인": list(avg_scores.keys()),
                    "집단 평균": [round(value, 1) for value in avg_scores.values()],
                    "시각화 점수": [round(normalized_scores[key], 1) for key in avg_scores.keys()],
                }
            )
            st.markdown(
                "<h5 style='margin-top:12px; font-size:15px; font-weight:700;'>📊 집단별 5대 요인 요약</h5>",
                unsafe_allow_html=True,
            )
            st.dataframe(raw_summary, use_container_width=True, hide_index=True)


def render_school_recommendation_page():
    filters = render_filter_controls(
        raw_df,
        meta,
        "school_rec",
        fields=["years", "regions", "school_levels", "grades", "genders"],
    )
    filter_col1, filter_col2 = st.columns([1, 1.4])
    with filter_col1:
        program_filter = st.selectbox(
            "추천 프로그램 필터",
            ["전체", "건강체력교실 우선 배정 요망", "방과후 체육클럽 권장"],
            key="school_rec_program_filter",
        )
    with filter_col2:
        school_keyword = st.text_input(
            "학교명 검색",
            placeholder="예: 춘천, 원주, ○○중",
            key="school_rec_keyword",
        ).strip()

    result, error = build_clustered_view(raw_df, meta, filters)
    if error:
        st.warning(error)
        return
    risk_schools = result["cluster_source"][result["cluster_source"]["유형"].isin(["고위험군", "관리 필요군", "중점관리군"])].copy()
    if risk_schools.empty:
        st.info("현재 조건에서는 추천 대상 학교가 없습니다.")
        return
    rec_df = risk_schools[["순수학교명", "시군", "연도", "학년", "성별", "유형"]].copy()
    rec_df["추천 프로그램"] = rec_df["유형"].apply(
        lambda label: "건강체력교실 우선 배정 요망" if label in ["고위험군", "관리 필요군"] else "방과후 체육클럽 권장"
    )
    if program_filter != "전체":
        rec_df = rec_df[rec_df["추천 프로그램"] == program_filter]
    if school_keyword:
        rec_df = rec_df[rec_df["순수학교명"].astype(str).str.contains(school_keyword, case=False, na=False)]

    if rec_df.empty:
        st.info("선택한 추천 프로그램 또는 학교명 조건에 맞는 학교가 없습니다.")
        return

    st.caption(f"조건에 맞는 추천 대상 {len(rec_df)}건을 표시합니다.")
    rec_df = rec_df.head(30)
    for _, row in rec_df.iterrows():
        tag_class = "tag-priority" if row["추천 프로그램"] == "건강체력교실 우선 배정 요망" else "tag-normal"
        tag_text = row["추천 프로그램"]
        st.markdown(
            f"""
            <div class="report-card">
                <span class="program-tag {tag_class}">{tag_text}</span>
                <h4>{row["순수학교명"]}</h4>
                <p>{row["시군"]} · {row["학년"]}학년 · {row["성별"]} · AI 분류: <b>{row["유형"]}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<div class='section-space'></div>", unsafe_allow_html=True)
    st.dataframe(rec_df, use_container_width=True, hide_index=True)


def render_teacher_priority_page():
    filters = render_filter_controls(
        raw_df,
        meta,
        "teacher_priority",
        fields=["years", "regions", "school_levels"],
    )
    result, error = build_clustered_view(raw_df, meta, filters)
    if error:
        st.warning(error)
        return
    risk_schools = result["cluster_source"][result["cluster_source"]["유형"].isin(["고위험군", "관리 필요군", "중점관리군"])].copy()
    priority = risk_schools.groupby("순수학교명").agg({"유형": "count", result["raw_y"]: "mean", "시군": "first"}).reset_index()
    priority.columns = ["학교명", "취약 학생군 건수", "심폐지표 평균", "시군"]
    priority = priority.sort_values(["취약 학생군 건수", "심폐지표 평균"], ascending=[False, True])
    top_region = priority["시군"].iloc[0] if not priority.empty else "확인 필요"
    st.markdown(
        f"""
        <div class="alert-card">
            AI 알림: 현재 <b>{top_region}</b> 권역에서 취약군 비율이 높게 감지되었습니다. 전문 스포츠 강사와 건강체력교실 우선 배치 검토가 필요합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(priority.head(10), use_container_width=True, hide_index=True)


def render_budget_page():
    filters = render_filter_controls(
        raw_df,
        meta,
        "budget",
        fields=["years", "regions", "school_levels"],
    )
    result, error = build_clustered_view(raw_df, meta, filters)
    if error:
        st.warning(error)
        return
    budget_df = (
        result["cluster_source"]
        .groupby("시군")
        .agg(
            취약학교수=("유형", lambda x: int(x.isin(["고위험군", "관리 필요군", "중점관리군"]).sum())),
            전체학교수=("순수학교명", "count"),
            평균심폐지표=(result["raw_y"], "mean"),
        )
        .reset_index()
    )
    budget_df["취약비율"] = (budget_df["취약학교수"] / budget_df["전체학교수"] * 100).round(1)
    budget_df["1인당 체육 예산"] = (120 - budget_df["취약비율"] * 0.8).clip(lower=35).round(1)
    budget_df["지역 취약도 점수"] = budget_df["취약비율"]
    top_risk = budget_df.sort_values("지역 취약도 점수", ascending=False).head(1)
    blind_spot = budget_df.sort_values(["지역 취약도 점수", "1인당 체육 예산"], ascending=[False, True]).head(1)
    if not budget_df.empty:
        top_name = top_risk["시군"].iloc[0]
        top_score = top_risk["지역 취약도 점수"].iloc[0]
        blind_name = blind_spot["시군"].iloc[0]
        blind_budget = blind_spot["1인당 체육 예산"].iloc[0]
        st.markdown(
            f"""
            <div class="insight-card">
                <h4>그래프 해석 요약</h4>
                <p><b>{top_name}</b> 권역의 지역 취약도 점수가 {top_score}점으로 가장 높게 나타났습니다. 
                또한 <b>{blind_name}</b>은 취약도 대비 1인당 체육 예산 지표가 낮아 예산 사각지대 후보로 우선 검토할 필요가 있습니다.</p>
                <p style="margin-top:10px;">아래 그래프는 지역별 취약 비율을 먼저 확인한 뒤, 예산 투입 수준과 취약도 점수가 같은 방향으로 움직이는지 비교하는 화면입니다.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    fig = px.bar(budget_df.sort_values("취약비율", ascending=False), x="시군", y="취약비율", title="지역별 취약 비율")
    apply_readable_axes(fig, height=430, margin=dict(t=58, b=56, l=40, r=20))
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)
    compare_df = budget_df.melt(
        id_vars="시군",
        value_vars=["1인당 체육 예산", "지역 취약도 점수"],
        var_name="지표",
        value_name="값",
    )
    fig_compare = px.bar(compare_df, x="시군", y="값", color="지표", barmode="group", title="1인당 체육 예산과 지역 취약도 비교")
    apply_readable_axes(fig_compare, height=430, margin=dict(t=58, b=56, l=40, r=20))
    fig_compare.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_compare, use_container_width=True)
    if not blind_spot.empty:
        st.markdown(
            f"""
            <div class="alert-card">
                예산 사각지대 후보: <b>{blind_spot["시군"].iloc[0]}</b> · 취약도 {blind_spot["지역 취약도 점수"].iloc[0]}점 · 1인당 예산 {blind_spot["1인당 체육 예산"].iloc[0]}점
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.dataframe(budget_df.sort_values("취약비율", ascending=False), use_container_width=True)


def render_student_profile_inputs(key_prefix):
    height_cm = st.number_input("키 (cm)", min_value=120, max_value=210, value=165, step=1, key=f"{key_prefix}_height")
    weight_kg = st.number_input("몸무게 (kg)", min_value=25, max_value=150, value=58, step=1, key=f"{key_prefix}_weight")
    st.markdown("##### 5대 체력 요인 입력")
    shuttle_runs = st.number_input("심폐지구력 - 셔틀런 횟수", min_value=1, max_value=200, value=42, step=1, key=f"{key_prefix}_cardio")
    strength_score = st.number_input("근력/근지구력 - 악력 또는 근력 점수", min_value=1, max_value=100, value=38, step=1, key=f"{key_prefix}_strength")
    flexibility_cm = st.number_input("유연성 - 앉아윗몸앞으로굽히기(cm)", min_value=-20, max_value=50, value=15, step=1, key=f"{key_prefix}_flex")
    power_cm = st.number_input("순발력 - 제자리멀리뛰기(cm)", min_value=50, max_value=350, value=175, step=1, key=f"{key_prefix}_power")
    return classify_student_profile(
        height_cm,
        weight_kg,
        shuttle_runs,
        strength_score,
        flexibility_cm,
        power_cm,
    )


def render_b2c_diagnosis_page():
    st.markdown("#### 나의 AI 체력 진단")
    left_input, right_mock = st.columns([0.9, 1.1])
    with left_input:
        bmi, allometric_index, cluster_label, component_scores = render_student_profile_inputs("diagnosis")
    title_1, body_1, title_2, body_2 = get_prescription_content(cluster_label)

    with right_mock:
        colors = plot_theme_colors()
        gauge_value = allometric_index
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                title={"text": "AI 보정 체력 지수"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0f766e"},
                    "steps": [
                        {"range": [0, 40], "color": "#fee2e2"},
                        {"range": [40, 70], "color": "#fed7aa"},
                        {"range": [70, 100], "color": "#dcfce7"},
                    ],
                },
            )
        )
        apply_plotly_theme(gauge, height=280, margin=dict(t=40, b=10, l=20, r=20))
        gauge.update_traces(
            title={"font": {"color": colors["text"]}},
            number={"font": {"color": colors["text"]}},
            gauge={"axis": {"tickcolor": colors["axis"], "tickfont": {"color": colors["text"]}}},
        )
        st.plotly_chart(gauge, use_container_width=True)
        component_df = pd.DataFrame(
            {"체력 요인": list(component_scores.keys()), "보정 점수": list(component_scores.values())}
        )
        component_fig = px.bar(
            component_df,
            x="체력 요인",
            y="보정 점수",
            color="체력 요인",
            title="체력 요인별 보정 점수",
            color_discrete_sequence=["#0f766e", "#2574ea", "#ef8b2c", "#1c9d74", "#d44b57"],
        )
        component_fig.update_layout(showlegend=False)
        apply_readable_axes(component_fig, height=300, margin=dict(t=52, b=48, l=36, r=16))
        component_fig.update_yaxes(range=[0, 100])
        st.plotly_chart(component_fig, use_container_width=True)
        st.markdown(
            f"""
            <div class="mobile-frame">
                <div class="phone-badge">나의 AI 체력 진단</div>
                <h4 style="margin:0 0 10px 0;">{cluster_label}</h4>
                <p style="margin:0;color:#475467;line-height:1.8;">BMI {bmi:.1f} · 체격 보정지수 {allometric_index:.1f}점<br>키와 체중을 반영해 5대 체력 요인을 함께 평가합니다.</p>
                <div class="mission-row"><span>오늘의 미션</span><b>30분 빠르게 걷기</b></div>
                <div class="mission-row"><span>1주차</span><b>{title_1}</b></div>
                <div class="mission-row"><span>체크</span><b>□ 완료</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""
        <div class="alert-card">
            진단 요약: {body_1} {body_2}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_b2c_plan_page():
    st.markdown("#### 4주 맞춤 운동 플랜 발급")
    left_input, right_plan = st.columns([0.85, 1.15])
    with left_input:
        bmi, allometric_index, cluster_label, component_scores = render_student_profile_inputs("plan")

    title_1, body_1, title_2, body_2 = get_prescription_content(cluster_label)
    intensity = "저강도" if "고위험" in cluster_label or "관리 필요" in cluster_label else "중등도"
    plan_df = pd.DataFrame(
        {
            "주차": ["1주차", "2주차", "3주차", "4주차"],
            "빈도(F)": ["주 3회", "주 3~4회", "주 4회", "주 4~5회"],
            "강도(I)": [intensity, intensity, "중등도+" if intensity == "중등도" else "저~중강도", "개인 목표 강도"],
            "시간(T)": ["20~30분", "30분", "35~40분", "40~45분"],
            "종류(T)": ["걷기·스트레칭", "순환운동", "인터벌+근력", "자기 기록 관리"],
        }
    )

    with right_plan:
        st.markdown(
            f"""
            <div class="report-card">
                <span class="report-tag {get_group_style(cluster_label)}">{cluster_label}</span>
                <h4>{title_1}</h4>
                <p>BMI {bmi:.1f} · AI 보정 체력 지수 {allometric_index:.1f}점</p>
                <p>{body_1}</p>
                <p><b>{title_2}</b><br>{body_2}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(plan_df, use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class="alert-card">
                오늘의 미션: 30분 빠르게 걷기 후 하체 스트레칭 8분을 완료 체크하세요.
            </div>
            """,
            unsafe_allow_html=True,
        )


page_map = {
    "강원특별자치도 체력 현황 요약": render_overview,
    "체력 취약망 지도 (Heatmap)": render_heatmap_page,
    "체격 보정 평가 모델 (Allometric)": render_allometric_page,
    "AI 다차원 군집 분석": render_cluster_page,
    "종목/학년별 상세 통계": render_detail_page,
    "집단별 FITT 처방": render_prescription_page,
    "학교별 교육 프로그램 추천": render_school_recommendation_page,
    "체육 강사 우선 배치망": render_teacher_priority_page,
    "지역별 예산 집행 타당성": render_budget_page,
    "나의 AI 체력 진단": render_b2c_diagnosis_page,
    "4주 맞춤 운동 플랜 발급": render_b2c_plan_page,
}

if current_page not in page_map:
    current_page = "강원특별자치도 체력 현황 요약"
    st.session_state["current_page"] = current_page

page_map[current_page]()
