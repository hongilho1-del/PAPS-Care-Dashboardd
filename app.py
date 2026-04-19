import streamlit as st
import pandas as pd
import os
import glob
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── 1. 웹페이지 설정 및 스타일링 ────────────────────────────────────────────────
st.set_page_config(
    page_title="PAPS Care+ Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    "<h1 style='margin-top: -20px;'>📊 <b>PAPS CARE+</b> "
    "<span style='font-size:0.55em; color:#666; font-weight:normal;'>| 강원특별자치도 학교 데이터 AI 분석 시스템</span></h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='color: #7f8c8d; font-size: 0.95em; margin-top: -15px; margin-bottom: 25px;'>"
    "* 본 시스템은 <b>학교알리미</b> 공시 데이터를 기반으로 학생들의 건강체력평가(PAPS)를 AI로 분석한 결과를 제공합니다."
    "</p>",
    unsafe_allow_html=True
)

# ─── 2. 공통 유틸 ───────────────────────────────────────────────────────────────
def normalize_text_series(series, default="전체"):
    s = series.astype(str).str.strip()
    s = s.replace({"": default, "nan": default, "None": default})
    return s.fillna(default)

def normalize_grade_value(v):
    if pd.isna(v):
        return "전체"
    v = str(v).strip()
    if v in ["", "nan", "None"]:
        return "전체"
    v = v.replace("학년", "").strip()
    return f"{v}학년" if v else "전체"

def infer_school_level(name):
    if pd.isna(name):
        return "전체"
    name = str(name).strip()
    if name.endswith("중"):
        return "중"
    if name.endswith("고"):
        return "고"
    return "전체"

def apply_filters(df, filters):
    result = df.copy()

    if filters.get("year"):
        result = result[result["연도"].isin(filters["year"])]
    if filters.get("region"):
        result = result[result["시군"].isin(filters["region"])]
    if filters.get("school_level"):
        result = result[result["학교급"].isin(filters["school_level"])]
    if filters.get("grade"):
        result = result[result["학년"].isin(filters["grade"])]
    if filters.get("gender"):
        result = result[result["성별"].isin(filters["gender"])]
    if filters.get("school"):
        result = result[result["순수학교명"].isin(filters["school"])]

    return result

# ─── 3. 데이터 로드 및 전처리 ──────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    patterns = [
        os.path.join(data_dir, "PAPS_Final_Master*.xlsx"),
        os.path.join(data_dir, "PAPS_Combined_Data*.xlsx"),
    ]

    matched_files = []
    for pattern in patterns:
        matched_files.extend(glob.glob(pattern))

    if not matched_files:
        return None, {
            "error": f"데이터 파일을 찾지 못했습니다. 확인 폴더: {data_dir}"
        }

    file_path = max(matched_files, key=os.path.getmtime)

    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.map(str).str.strip()

        def find_col(keywords):
            for c in df.columns:
                c_str = str(c)
                for kw in keywords:
                    if kw in c_str:
                        return c
            return None

        target_mapping = {
            "BMI": find_col(["BMI", "비만", "체질량"]),
            "심폐지구력 (왕복오래달리기)": find_col(["왕복", "오래달리기", "심폐"]),
            "근력 (악력)": find_col(["악력"]),
            "근력 (팔굽혀/말아올리기)": find_col(["팔굽혀", "말아올리기", "윗몸말아올리기"]),
            "유연성 (앉아윗몸)": find_col(["앉아윗몸", "유연성"]),
            "순발력 (제자리멀리뛰기)": find_col(["제자리멀리", "멀리뛰기", "순발력"]),
        }
        valid_cols = {k: v for k, v in target_mapping.items() if v is not None}

        for col in valid_cols.values():
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                errors="coerce"
            )

        if "순수학교명" in df.columns:
            school_col = "순수학교명"
        elif "추출학교명" in df.columns:
            school_col = "추출학교명"
        else:
            school_col = df.columns[0]

        df["순수학교명"] = normalize_text_series(df[school_col], default="학교명미상")

        if "연도" in df.columns:
            df["연도"] = pd.to_numeric(df["연도"], errors="coerce").fillna(0).astype(int)
        else:
            extracted_year = df[school_col].astype(str).str.extract(r"(20\d{2}|19\d{2})")[0]
            df["연도"] = pd.to_numeric(extracted_year, errors="coerce").fillna(0).astype(int)

        if "시군" in df.columns:
            df["시군"] = normalize_text_series(df["시군"], default="강원")
        else:
            df["시군"] = "강원"

        if "학교급" in df.columns:
            df["학교급"] = normalize_text_series(df["학교급"], default="전체")
        else:
            df["학교급"] = df["순수학교명"].apply(infer_school_level)

        gender_col = find_col(["성별", "남여", "구분_성별"])
        if gender_col:
            df["성별"] = normalize_text_series(df[gender_col], default="전체")
        else:
            df["성별"] = "전체"

        grade_col = find_col(["학년", "구분_학년"])
        if grade_col:
            df["학년"] = df[grade_col].apply(normalize_grade_value)
        else:
            df["학년"] = "전체"

        df["표시용이름"] = df.apply(
            lambda row: f"{row['순수학교명']} ({row['연도']})" if row["연도"] > 0 else row["순수학교명"],
            axis=1
        )

        return df, {
            "school_col": school_col,
            "valid_cols": valid_cols,
            "file_path": file_path,
            "matched_files": matched_files,
        }

    except Exception as e:
        return None, {"error": f"데이터 로드 중 오류 발생: {e}"}

# ─── 4. AI 군집 분석 함수 ─────────────────────────────────────────────────────
def get_clustered_df(tab_df, valid_cols, x_axis, y_axis, n_clusters):
    raw_x = valid_cols[x_axis]
    raw_y = valid_cols[y_axis]

    agg_dict = {v: "mean" for v in valid_cols.values()}
    agg_dict["연도"] = "first"
    agg_dict["시군"] = "first"
    agg_dict["학교급"] = "first"

    df_agg = tab_df.groupby(["순수학교명", "표시용이름"]).agg(agg_dict).reset_index()

    X = df_agg[[raw_x, raw_y]].dropna()
    if len(X) < n_clusters:
        return pd.DataFrame()

    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_agg.loc[X.index, "Cluster"] = kmeans.fit_predict(X_scaled)
    df_agg = df_agg.dropna(subset=["Cluster"])

    cluster_means = df_agg.groupby("Cluster")[raw_x].mean().sort_values(ascending=False)
    rank_map = {cluster_idx: i for i, cluster_idx in enumerate(cluster_means.index)}

    if n_clusters == 2:
        names = ["🔴 관리 필요군", "🔵 건강 양호군"]
    elif n_clusters == 3:
        names = ["🔴 고위험군", "🟢 일반군", "🔵 건강 우수군"]
    else:
        names = ["🔴 고위험군", "🟠 중점 관리군", "🟢 일반군", "🔵 건강 우수군"]

    df_agg["유형"] = df_agg["Cluster"].map(rank_map).apply(
        lambda x: names[int(x)] if x < len(names) else "⚪ 기타"
    )

    inv_map = {v: k for k, v in valid_cols.items()}
    df_agg = df_agg.rename(columns=inv_map).rename(columns={"표시용이름": "학교(연도)"})

    return df_agg

# ─── 5. 통합 렌더링 함수 ───────────────────────────────────────────────────────
def render_dashboard(raw_df, valid_cols, filters):
    if len(valid_cols) < 2:
        st.warning("분석 가능한 수치 지표가 2개 이상 필요합니다. 엑셀 컬럼명을 확인해주세요.")
        return

    tab_df = apply_filters(raw_df, filters)

    if tab_df.empty:
        st.info("💡 선택하신 필터 조건에 맞는 데이터가 없습니다.")
        return

    col_set1, col_set2 = st.columns([1, 3])
    metrics = list(valid_cols.keys())

    with col_set1:
        st.write("### ⚙️ 분석 지표")
        x_axis = st.selectbox("X축 (주로 BMI)", metrics, index=0)
        y_default = 1 if len(metrics) > 1 else 0
        y_axis = st.selectbox("Y축 (주로 체력지표)", metrics, index=y_default)
        n_clusters = st.slider("군집 세분화 (개)", 2, 4, 3)
        is_mobile = st.toggle("📱 모바일 최적화")

    plot_df = get_clustered_df(tab_df, valid_cols, x_axis, y_axis, n_clusters)

    if plot_df.empty:
        st.warning("⚠️ 분석을 수행하기 위한 데이터가 부족합니다. 필터를 조금 넓히거나 군집 수를 줄여보세요.")
        return

    with col_set2:
        color_map = {
            "🔴 고위험군": "#EF5350",
            "🔴 관리 필요군": "#EF5350",
            "🟠 중점 관리군": "#FFB74D",
            "🟢 일반군": "#66BB6A",
            "🔵 건강 우수군": "#42A5F5",
            "🔵 건강 양호군": "#42A5F5",
        }

        fig = px.scatter(
            plot_df,
            x=x_axis,
            y=y_axis,
            color="유형",
            text="학교(연도)",
            hover_name="학교(연도)",
            hover_data={"학교급": True, "시군": True},
            color_discrete_map=color_map,
            title="🏫 통합 건강 데이터 AI 분석 결과"
        )
        fig.update_traces(
            textposition="top center",
            marker=dict(
                size=12 if is_mobile else 22,
                line=dict(width=2, color="white")
            )
        )
        fig.update_layout(
            height=550,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.write("### 📋 그룹별 맞춤형 운동 처방 및 교육 프로그램")

    sum_df = plot_df.groupby("유형")[[x_axis, y_axis]].mean().round(1)
    counts = plot_df["유형"].value_counts()

    head_col1, head_col2, head_col3 = st.columns([1.2, 2, 2])
    with head_col1:
        st.markdown("##### 📊 분석 집단군")
    with head_col2:
        st.markdown("##### 🏃‍♂️ 맞춤형 운동 처방")
    with head_col3:
        st.markdown("##### 💊 교육 프로그램 추천")
    st.markdown("<hr style='margin-top: 0px; margin-bottom: 20px;'>", unsafe_allow_html=True)

    for idx in sum_df.index:
        col1, col2, col3 = st.columns([1.2, 2, 2])

        with col1:
            st.metric(
                label=idx,
                value=f"{counts.get(idx, 0)}건",
                delta=f"{x_axis} 평균: {sum_df.loc[idx, x_axis]}",
                delta_color="off"
            )

        with col2:
            if "🔴" in idx:
                st.error("**저강도 유산소 위주 구성**\n\n관절에 무리가 가지 않는 걷기, 수영, 실내 자전거 등을 통한 기초 체력 증진 및 체지방 감소 유도.")
            elif "🟠" in idx:
                st.warning("**뉴스포츠 등 활동량 증대**\n\n흥미를 유발할 수 있는 신체 활동을 통해 일상적인 움직임을 늘리고 과체중 진입 예방.")
            elif "🟢" in idx:
                st.success("**전신 근력 및 유연성 밸런스**\n\n현재의 체력을 유지하면서 신체 각 부위를 골고루 발달시킬 수 있는 밸런스 운동.")
            elif "🔵" in idx:
                st.info("**고강도 심화 트레이닝**\n\n우수한 체력을 바탕으로 심폐지구력과 순발력을 극대화하는 고강도 인터벌 트레이닝(HIIT) 및 스포츠 기술 습득.")

        with col3:
            if "🔴" in idx:
                st.error("**건강체력교실 우선 배정**\n\n전문 강사의 집중 관리, 가정통신문 연계 식습관 및 영양 상담 정기 진행.")
            elif "🟠" in idx:
                st.warning("**교내 걷기 챌린지 참여**\n\n방과 후 스포츠클럽 가입 적극 권장 및 또래와 함께하는 재미 위주의 교내 프로그램 도입.")
            elif "🟢" in idx:
                st.success("**정규 체육 수업 충실**\n\n1일 1시간 이상 일상적 신체활동 습관화 및 다양한 스포츠 종목 체험을 통한 체력 유지.")
            elif "🔵" in idx:
                st.info("**학생 스포츠 리더 선발**\n\n체육 동아리 멘토 위촉, 학교 대표 선수단 선발 및 지역 엘리트 체육 프로그램 연계 지원.")

        st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("🔍 상세 데이터 테이블 보기"):
        st.dataframe(
            plot_df.drop(columns=["Cluster"], errors="ignore").sort_values(["유형", "학교(연도)"]),
            use_container_width=True
        )

# ─── 6. 메인 실행 로직 ─────────────────────────────────────────────────────────
raw_df, meta = load_raw_data()

if raw_df is None:
    st.error(meta.get("error", "데이터를 불러오지 못했습니다."))
else:
    if not meta["valid_cols"]:
        st.warning("체력 분석에 사용할 수치형 컬럼을 찾지 못했습니다. 엑셀 컬럼명을 확인해주세요.")
    else:
        st.markdown("### 📍 분석 데이터 필터링")

        years = sorted([y for y in raw_df["연도"].unique() if y > 0])
        sigungus = sorted(raw_df["시군"].dropna().astype(str).unique())
        school_levels = sorted([s for s in raw_df["학교급"].dropna().astype(str).unique() if s != "전체"])
        grades = sorted([g for g in raw_df["학년"].dropna().astype(str).unique() if g != "전체"])
        genders = sorted([g for g in raw_df["성별"].dropna().astype(str).unique() if g != "전체"])

        f1, f2, f3, f4, f5, f6 = st.columns(6)

        with f1:
            s_year = st.multiselect("📅 연도", options=years, placeholder="전체")
        with f2:
            s_region = st.multiselect("📍 시·군", options=sigungus, placeholder="전체")
        with f3:
            s_school_level = st.multiselect("🏷️ 학교급", options=school_levels, placeholder="전체")
        with f4:
            s_grade = st.multiselect("🎓 학년", options=grades, placeholder="전체")
        with f5:
            s_gender = st.multiselect("👫 성별", options=genders, placeholder="남/여 전체")

        school_filter_base = apply_filters(raw_df, {
            "year": s_year,
            "region": s_region,
            "school_level": s_school_level,
            "grade": s_grade,
            "gender": s_gender,
            "school": []
        })

        f_schools = sorted(school_filter_base["순수학교명"].dropna().astype(str).unique())

        with f6:
            s_school = st.multiselect("🏫 학교명", options=f_schools, placeholder="전체")

        filters = {
            "year": s_year,
            "region": s_region,
            "school_level": s_school_level,
            "grade": s_grade,
            "gender": s_gender,
            "school": s_school,
        }

        render_dashboard(raw_df, meta["valid_cols"], filters)
