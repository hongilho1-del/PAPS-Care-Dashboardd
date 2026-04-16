import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="PAPS Care+ Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8fafc; }
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', -apple-system, sans-serif; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero-section {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        padding: 40px;
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.2);
    }
    .hero-title { font-size: 32px; font-weight: 800; margin-bottom: 5px; letter-spacing: -1px; }
    .hero-sub { font-size: 16px; opacity: 0.9; font-weight: 300; }

    .analysis-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
    }

    .paps-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        background: white;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    .paps-table th {
        background-color: #f1f5f9;
        color: #475569;
        padding: 20px;
        text-align: left;
        font-weight: 700;
        border-bottom: 2px solid #e2e8f0;
    }
    .paps-table td {
        padding: 25px 20px;
        border-bottom: 1px solid #f1f5f9;
        vertical-align: top;
    }

    .tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .tag-red { background: #fee2e2; color: #b91c1c; }
    .tag-orange { background: #ffedd5; color: #c2410c; }
    .tag-green { background: #dcfce7; color: #15803d; }
    .tag-blue { background: #dbeafe; color: #1d4ed8; }

    .content-box { font-size: 14.5px; line-height: 1.6; color: #334155; }
    .content-box b { color: #1e293b; display: block; margin-bottom: 8px; font-size: 15px; }
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
        2: ["🔴 관리필요", "🔵 건강양호"],
        3: ["🔴 고위험", "🟢 일반", "🔵 우수"],
        4: ["🔴 고위험", "🟠 중점관리", "🟢 일반", "🔵 우수"],
    }
    names = label_sets[len(ordered)]
    return {cluster_id: names[index] for index, cluster_id in enumerate(ordered)}


def get_prescription_content(label):
    category = label[0]
    return {
        "🔴": (
            "기초 체력 증진 루틴",
            "- 관절 부담이 적은 걷기, 수영 중심 활동<br>- 저강도 유산소 운동으로 주당 활동량 확보<br>- 생활 속 움직임을 늘리는 습관 형성",
            "건강체력교실 집중 케어",
            "- 기초 체력 향상 중심의 소그룹 프로그램<br>- 영양 상담과 가정 연계형 생활습관 코칭",
        ),
        "🟠": (
            "흥미 기반 단계형 운동",
            "- 뉴스포츠와 순환운동으로 참여도 향상<br>- 심폐지구력과 근지구력 향상 루틴 병행<br>- 기록 기반의 점진적 강도 조절",
            "방과 후 스포츠클럽 권장",
            "- 또래 참여형 팀 스포츠로 지속성 확보<br>- 성취 경험을 만드는 체육 활동 피드백 제공",
        ),
        "🟢": (
            "전신 밸런스 유지",
            "- 근력, 유연성, 지구력의 균형 유지 운동<br>- 부상 예방을 위한 스트레칭 정례화<br>- 현재 수준을 유지할 수 있는 주간 활동 계획",
            "자율 체육 활동 습관화",
            "- 1인 1운동 생활 습관 정착 지원<br>- 다양한 종목 체험으로 운동 흥미 유지",
        ),
        "🔵": (
            "고강도 심화 트레이닝",
            "- 인터벌 트레이닝과 전문 기술 연습 병행<br>- 개인별 강점 지표 중심의 심화 프로그램 구성<br>- 리더십 역할과 자기주도 운동 설계 강화",
            "학생 스포츠 리더 프로그램",
            "- 교내 체육 활동 멘토 역할 기회 제공<br>- 지역 연계형 심화 스포츠 프로그램 추천",
        ),
    }[category]


raw_df, meta, load_error = load_raw_data()

st.markdown(
    """
    <div class="hero-section">
        <div class="hero-title">PAPS CARE+</div>
        <div class="hero-sub">강원특별자치도 학교 데이터 AI 분석 시스템</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.error(load_error)
    st.info("`data/PAPS_Combined_Data.xlsx` 파일을 추가한 뒤 다시 실행해 주세요.")
    st.stop()


with st.sidebar:
    st.markdown("### 데이터 필터링")

    s_year = st.multiselect("연도", sorted(raw_df["연도"].dropna().unique()))
    s_region = st.multiselect("시·군", sorted(raw_df["시군"].dropna().unique()))
    s_grade = st.multiselect("학년", sorted(raw_df["학년"].dropna().unique()))
    s_gender = st.multiselect("성별", sorted(raw_df["성별"].dropna().unique()))

    school_base_df = apply_filters(raw_df, s_year, s_region, s_grade, s_gender, [])
    school_options = sorted(school_base_df["순수학교명"].dropna().unique())
    s_school = st.multiselect("학교명 검색", school_options)

metric_options = list(meta["valid"].keys())

st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    x_ax = st.selectbox("수평축 지표 (X)", metric_options, index=0)
with col2:
    y_default_index = 1 if len(metric_options) > 1 else 0
    y_ax = st.selectbox("수직축 지표 (Y)", metric_options, index=y_default_index)
with col3:
    n_cl = st.slider("AI 군집 세분화", 2, 4, 3)
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
    st.warning(
        f"현재 필터 조건에서는 군집 {n_cl}개를 만들기 위한 데이터가 부족합니다. "
        f"최소 {n_cl}개 이상의 집계 데이터가 필요합니다."
    )
    st.stop()

scaler = StandardScaler()
scaled_points = scaler.fit_transform(cluster_source[[raw_x, raw_y]])

kmeans = KMeans(n_clusters=n_cl, random_state=42, n_init=10)
cluster_source["Cluster"] = kmeans.fit_predict(scaled_points)

cluster_summary = cluster_source.groupby("Cluster")[[raw_x, raw_y]].mean()
cluster_summary["score"] = cluster_summary.mean(axis=1)
cluster_labels = build_cluster_labels(cluster_summary, x_ax)
cluster_source["유형"] = cluster_source["Cluster"].map(cluster_labels)

fig = px.scatter(
    cluster_source,
    x=raw_x,
    y=raw_y,
    color="유형",
    text="순수학교명",
    labels={raw_x: x_ax, raw_y: y_ax},
    hover_data={
        "연도": True,
        "시군": True,
        "학년": True,
        "성별": True,
    },
    color_discrete_map={
        "🔴 관리필요": "#ef4444",
        "🔴 고위험": "#ef4444",
        "🟠 중점관리": "#f97316",
        "🟢 일반": "#22c55e",
        "🔵 우수": "#3b82f6",
        "🔵 건강양호": "#3b82f6",
    },
)
fig.update_traces(textposition="top center")
fig.update_layout(
    height=520,
    margin=dict(t=10, b=10, l=10, r=10),
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend_title_text="분석 집단군",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "<h3 style='margin-top:40px;'>그룹별 맞춤형 처방 리포트</h3>",
    unsafe_allow_html=True,
)

sum_df = cluster_source.groupby("유형")[[raw_x, raw_y]].mean().round(1)
row_order = ["🔴 고위험", "🔴 관리필요", "🟠 중점관리", "🟢 일반", "🔵 우수", "🔵 건강양호"]
visible_rows = [label for label in row_order if label in sum_df.index]

table_html = """
<table class="paps-table">
    <tr>
        <th style="width: 20%;">분석 집단군</th>
        <th style="width: 40%;">맞춤형 운동 처방</th>
        <th style="width: 40%;">교육 프로그램 추천</th>
    </tr>
"""

for label in visible_rows:
    tag_class = (
        "tag-red"
        if "🔴" in label
        else "tag-orange"
        if "🟠" in label
        else "tag-green"
        if "🟢" in label
        else "tag-blue"
    )
    prescription = get_prescription_content(label)
    table_html += f"""
    <tr>
        <td>
            <span class="tag {tag_class}">{label}</span><br>
            <div style="font-size:13px; color:#64748b;">{x_ax} 평균: {sum_df.loc[label, raw_x]}</div>
            <div style="font-size:13px; color:#64748b;">{y_ax} 평균: {sum_df.loc[label, raw_y]}</div>
        </td>
        <td><div class="content-box"><b>[{prescription[0]}]</b>{prescription[1]}</div></td>
        <td><div class="content-box"><b>[{prescription[2]}]</b>{prescription[3]}</div></td>
    </tr>
    """

table_html += "</table>"
st.markdown(table_html, unsafe_allow_html=True)

st.caption(
    "안내: 군집 라벨은 선택한 지표 조합을 기준으로 상대 비교한 결과입니다. "
    "특히 BMI는 높은 값이 항상 우수함을 의미하지 않도록 별도 방향성을 적용했습니다."
)
