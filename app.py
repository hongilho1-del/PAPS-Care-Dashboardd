import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── 1. 웹페이지 설정 및 스타일링 ────────────────────────────────────────────────
st.set_page_config(page_title="PAPS Care+ Dashboard", layout="wide", initial_sidebar_state="expanded")

# 제목 강조 및 부제목/출처 문구 구성 (실데이터 -> 데이터)
st.markdown(
    "<h1 style='margin-top: -20px;'>📊 <b>PAPS CARE+</b> <span style='font-size:0.55em; color:#666; font-weight:normal;'>| 강원특별자치도 학교 데이터 AI 분석 시스템</span></h1>", 
    unsafe_allow_html=True
)

st.markdown(
    "<p style='color: #7f8c8d; font-size: 0.95em; margin-top: -15px; margin-bottom: 25px;'>"
    "* 본 시스템은 <b>학교알리미</b> 공시 데이터를 기반으로 학생들의 건강체력평가(PAPS)를 AI로 분석한 결과를 제공합니다."
    "</p>", 
    unsafe_allow_html=True
)

# ─── 2. 데이터 로드 및 전처리 ──────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    # 데이터 경로 설정 (파일이 data 폴더 안에 있다고 가정)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', 'PAPS_Combined_Data.xlsx')

    if not os.path.exists(file_path):
        return None, {}

    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip() # 열 이름 공백 제거
        
        # 유연한 컬럼 매칭 함수
        def find_col(keywords):
            for c in df.columns:
                for kw in keywords:
                    if kw in str(c): return c
            return None

        # 지표별 컬럼 자동 탐색
        bmi_col      = find_col(['BMI', '비만', '체질량'])
        run_col      = find_col(['왕복', '오래달리기', '심폐'])
        grip_col     = find_col(['악력'])
        push_col     = find_col(['팔굽혀', '말아올리기']) 
        flex_col     = find_col(['앉아윗몸', '유연성'])
        jump_col     = find_col(['제자리멀리', '순발력'])

        target_cols = {
            'BMI': bmi_col, 
            '심폐지구력 (왕복오래달리기)': run_col, 
            '근력 (악력)': grip_col,
            '근력 (팔굽혀/말아올리기)': push_col,
            '유연성 (앉아윗몸)': flex_col, 
            '순발력 (제자리멀리뛰기)': jump_col
        }
        valid_cols = {k: v for k, v in target_cols.items() if v}

        # 숫자 데이터 정제
        for k, col in valid_cols.items():
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

        # 기초 정보 필드 정리
        school_col = '추출학교명' if '추출학교명' in df.columns else df.columns[0]
        df['순수학교명'] = df[school_col].astype(str).str.strip()
        
        if '연도' in df.columns:
            df['연도'] = pd.to_numeric(df['연도'], errors='coerce').fillna(0).astype(int)
        else:
            df['연도'] = df[school_col].astype(str).str.extract(r'(20\d{2}|19\d{2})')[0]
            df['연도'] = pd.to_numeric(df['연도'], errors='coerce').fillna(0).astype(int)

        df['표시용이름'] = df.apply(lambda row: f"{row['순수학교명']} ({row['연도']})" if row['연도'] > 0 else row['순수학교명'], axis=1)
        df['시군'] = df['시군'].astype(str).str.strip() if '시군' in df.columns else '강원'

        # 성별 및 학년 표준화
        gender_col = find_col(['성별', '남여', '구분_성별'])
        df['성별'] = df[gender_col].astype(str).str.strip() if gender_col else '전체'

        grade_col = find_col(['학년', '구분_학년'])
        if grade_col:
            df['학년'] = df[grade_col].astype(str).str.replace('학년', '', regex=False).str.strip() + '학년'
        else:
            df['학년'] = '전체'

        return df, {'school_col': school_col, 'valid_cols': valid_cols}
    except Exception as e:
        st.error(f"데이터를 로드하는 중 오류가 발생했습니다: {e}")
        return None, {}

# ─── 3. AI 군집 분석 및 등급 할당 ──────────────────────────────────────────────
def get_clustered_df(tab_df, valid_cols, x_axis, y_axis, n_clusters):
    agg_dict = {v: 'mean' for v in valid_cols.values()}
    agg_dict['연도'] = 'first'
    agg_dict['시군'] = 'first'
    
    # 분석용 요약 데이터 생성
    df = tab_df.groupby(['순수학교명', '표시용이름']).agg(agg_dict).reset_index()
    numeric_cols = list(valid_cols.values())
    df[numeric_cols] = df[numeric_cols].round(1)
    df.columns = ['순수학교명', '학교(연도)'] + list(valid_cols.keys()) + ['연도', '시군']
    df = df.dropna(subset=[x_axis, y_axis])

    if len(df) < n_clusters: return df

    # K-Means 클러스터링
    X = df[[x_axis, y_axis]]
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # BMI 기준으로 등급 명칭 동적 할당
    sort_metric = 'BMI' if 'BMI' in df.columns else x_axis
    cluster_means = df.groupby('Cluster')[sort_metric].mean().sort_values(ascending=False)
    rank_map = {cluster_idx: i for i, cluster_idx in enumerate(cluster_means.index)}
    
    if n_clusters == 2:
        name_list = ["🔴 관리 필요군", "🔵 건강 양호군"]
    elif n_clusters == 3:
        name_list = ["🔴 고위험군", "🟢 일반군", "🔵 건강 우수군"]
    else: 
        name_list = ["🔴 고위험군", "🟠 중점 관리군", "🟢 일반군", "🔵 건강 우수군"]
        
    df['유형'] = df['Cluster'].map(rank_map).apply(lambda x: name_list[x] if x < len(name_list) else "⚪ 기타")
    return df

# ─── 4. 메인 대시보드 렌더링 ────────────────────────────────────────────────────
def render_dashboard(tab_df, valid_cols, filters):
    col_set1, col_set2 = st.columns([1, 3])
    metrics = list(valid_cols.keys())

    with col_set1:
        st.write("### ⚙️ 분석 지표")
        x_axis = st.selectbox("X축 (주로 BMI)", metrics, index=0)
        y_axis = st.selectbox("Y축 (주로 체력지표)", metrics, index=min(1, len(metrics)-1))
        n_clusters = st.slider("군집 세분화 (개)", 2, 4, 3)
        is_mobile = st.toggle("📱 모바일 최적화")
        
    # [성별/학년] 필터는 평균 계산 전에 미리 적용
    if filters['gender']: tab_df = tab_df[tab_df['성별'].isin(filters['gender'])]
    if filters['grade']: tab_df = tab_df[tab_df['학년'].isin(filters['grade'])]
    
    plot_df = get_clustered_df(tab_df, valid_cols, x_axis, y_axis, n_clusters)

    if plot_df.empty or len(plot_df) < n_clusters:
        st.warning("⚠️ 선택하신 조건에 분석할 데이터가 부족합니다.")
        return

    # 축 범위 고정 (필터 변경 시에도 좌표 유지)
    x_min, x_max = plot_df[x_axis].min(), plot_df[x_axis].max()
    y_min, y_max = plot_df[y_axis].min(), plot_df[y_axis].max()
    x_range = [x_min - (x_max-x_min)*0.1, x_max + (x_max-x_min)*0.1]
    y_range = [y_min - (y_max-y_min)*0.1, y_max + (y_max-y_min)*0.1]

    # [연도/지역/학교] 필터는 시각화 직전에 적용 (상대적 위치 보존용)
    if filters['year']: plot_df = plot_df[plot_df['연도'].isin(filters['year'])]
    if filters['region']: plot_df = plot_df[plot_df['시군'].isin(filters['region'])]
    if filters['school']: plot_df = plot_df[plot_df['순수학교명'].isin(filters['school'])]
        
    if plot_df.empty:
        st.info("💡 해당 조건에 맞는 데이터가 없습니다.")
        return

    with col_set2:
        color_map = {"🔴 고위험군": "#EF5350", "🔴 관리 필요군": "#EF5350", "🟠 중점 관리군": "#FFB74D", "🟢 일반군": "#66BB6A", "🔵 건강 우수군": "#42A5F5", "🔵 건강 양호군": "#42A5F5"}
        fig = px.scatter(
            plot_df, x=x_axis, y=y_axis, color='유형', text='학교(연도)',
            hover_name='학교(연도)', color_discrete_map=color_map,
            title=f"🏫 조건별 건강 등급 분포 분석"
        )
        fig.update_traces(textposition='top center', marker=dict(size=12 if is_mobile else 22, line=dict(width=2, color='white')))
        fig.update_xaxes(range=x_range)
        fig.update_yaxes(range=y_range)
        fig.update_layout(height=550, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    # 처방 카드 및 상세 테이블 (기존 로직 유지)
    st.markdown("---")
    st.write("### 📋 맞춤형 처방 프로그램 및 운동 방향")
    sum_df = plot_df.groupby('유형')[[x_axis, y_axis]].mean().round(1)
    counts = plot_df['유형'].value_counts()
    
    met_cols = st.columns(len(sum_df))
    for i, (idx, row) in enumerate(sum_df.iterrows()):
        with met_cols[i]:
            st.metric(label=idx, value=f"{counts.get(idx, 0)}건", delta=f"{x_axis}: {row[x_axis]}", delta_color="off")

    st.markdown("<br>", unsafe_allow_html=True)
    card1, card2 = st.columns(2)
    for idx in sum_df.index:
        target = card1 if "🔴" in idx or "🟢" in idx else card2
        if "🔴" in idx:
            target.error(f"#### {idx}\n**🏃‍♂️ 운동 방향:** 저강도 유산소 위주 구성\n**💊 처방:** 건강체력교실 우선 배정 및 상담 병행")
        elif "🟠" in idx:
            target.warning(f"#### {idx}\n**🏃‍♂️ 운동 방향:** 뉴스포츠 등 신체 활동량 증대\n**💊 처방:** 교내 걷기 챌린지 및 스포츠클럽 참여 권장")
        elif "🟢" in idx:
            target.success(f"#### {idx}\n**🏃‍♂️ 운동 방향:** 전신 밸런스 운동 권장\n**💊 처방:** 일상적 신체활동 습관화 지속")
        elif "🔵" in idx:
            target.info(f"#### {idx}\n**🏃‍♂️ 운동 방향:** 고강도 트레이닝 및 전문 기술 습득\n**💊 처방:** 스포츠 리더 선발 및 엘리트 체육 연계")

    with st.expander("🔍 상세 데이터 테이블"):
        st.dataframe(plot_df.drop(columns=['순수학교명', '연도', '시군'], errors='ignore'), use_container_width=True)

# ─── 5. 실행 로직 ─────────────────────────────────────────────────────────────
raw_df, meta = load_raw_data()
if raw_df is not None:
    # 상단 5단계 다중 필터 구성
    years = sorted([y for y in raw_df['연도'].unique() if y > 0])
    sigungus = sorted(raw_df['시군'].unique())
    grades = sorted([g for g in raw_df['학년'].unique() if '전체' not in g])
    genders = sorted([g for g in raw_df['성별'].unique() if '전체' not in g])

    # [수정됨] 비교 기준(라디오 버튼) 제거
    st.markdown("### 📍 분석 데이터 필터링")

    f1, f2, f3, f4, f5 = st.columns(5)
    with f1: s_year = st.multiselect("📅 연도", options=years, placeholder="전체")
    with f2: s_region = st.multiselect("📍 시·군", options=sigungus, placeholder="전체")
    with f3: s_grade = st.multiselect("🎓 학년", options=grades, placeholder="전체")
    with f4: s_gender = st.multiselect("👫 성별", options=genders, placeholder="전체")
    
    # 필터링된 학교 목록 동적 생성
    tmp = raw_df.copy()
    if s_year: tmp = tmp[tmp['연도'].isin(s_year)]
    if s_region: tmp = tmp[tmp['시군'].isin(s_region)]
    if s_grade: tmp = tmp[tmp['학년'].isin(s_grade)]
    if s_gender: tmp = tmp[tmp['성별'].isin(s_gender)]
    f_schools = sorted(tmp['순수학교명'].unique())
    
    with f5: s_school = st.multiselect("🏫 학교명", options=f_schools, placeholder="전체")

    # 통합 필터 딕셔너리
    filters = {'year': s_year, 'region': s_region, 'grade': s_grade, 'gender': s_gender, 'school': s_school}

    # [수정됨] 탭 구조를 제거하고 단일 리포트로 곧바로 렌더링
    st.markdown("---")
    render_dashboard(raw_df, meta['valid_cols'], filters)
