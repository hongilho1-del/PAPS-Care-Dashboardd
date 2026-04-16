import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── 1. 웹페이지 설정 및 스타일링 ────────────────────────────────────────────────
st.set_page_config(page_title="PAPS Care+ Dashboard", layout="wide", initial_sidebar_state="expanded")

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', 'PAPS_Combined_Data.xlsx')

    if not os.path.exists(file_path):
        return None, {}

    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        
        def find_col(keywords):
            for c in df.columns:
                for kw in keywords:
                    if kw in str(c): return c
            return None

        target_mapping = {
            'BMI': find_col(['BMI', '비만', '체질량']),
            '심폐지구력 (왕복오래달리기)': find_col(['왕복', '오래달리기', '심폐']),
            '근력 (악력)': find_col(['악력']),
            '근력 (팔굽혀/말아올리기)': find_col(['팔굽혀', '말아올리기']),
            '유연성 (앉아윗몸)': find_col(['앉아윗몸', '유연성']),
            '순발력 (제자리멀리뛰기)': find_col(['제자리멀리', '순발력'])
        }
        valid_cols = {k: v for k, v in target_mapping.items() if v}

        for col in valid_cols.values():
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

        school_col = '추출학교명' if '추출학교명' in df.columns else df.columns[0]
        df['순수학교명'] = df[school_col].astype(str).str.strip()
        
        if '연도' in df.columns:
            df['연도'] = pd.to_numeric(df['연도'], errors='coerce').fillna(0).astype(int)
        else:
            df['연도'] = df[school_col].astype(str).str.extract(r'(20\d{2}|19\d{2})')[0]
            df['연도'] = pd.to_numeric(df['연도'], errors='coerce').fillna(0).astype(int)

        df['표시용이름'] = df.apply(lambda row: f"{row['순수학교명']} ({row['연도']})" if row['연도'] > 0 else row['순수학교명'], axis=1)
        df['시군'] = df['시군'].astype(str).str.strip() if '시군' in df.columns else '강원'

        gender_col = find_col(['성별', '남여', '구분_성별'])
        df['성별'] = df[gender_col].astype(str).str.strip() if gender_col else '전체'
        
        grade_col = find_col(['학년', '구분_학년'])
        if grade_col:
            df['학년'] = df[grade_col].astype(str).str.replace('학년', '', regex=False).str.strip() + '학년'
        else:
            df['학년'] = '전체'

        return df, {'school_col': school_col, 'valid_cols': valid_cols}
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None, {}

# ─── 3. AI 군집 분석 함수 ──────────────────────────────────────────
def get_clustered_df(tab_df, valid_cols, x_axis, y_axis, n_clusters):
    raw_x = valid_cols[x_axis]
    raw_y = valid_cols[y_axis]
    
    agg_dict = {v: 'mean' for v in valid_cols.values()}
    agg_dict['연도'] = 'first'
    agg_dict['시군'] = 'first'
    
    df_agg = tab_df.groupby(['순수학교명', '표시용이름']).agg(agg_dict).reset_index()
    
    X = df_agg[[raw_x, raw_y]].dropna()
    if len(X) < n_clusters: return pd.DataFrame()

    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_agg.loc[X.index, 'Cluster'] = kmeans.fit_predict(X_scaled)
    df_agg = df_agg.dropna(subset=['Cluster'])

    cluster_means = df_agg.groupby('Cluster')[raw_x].mean().sort_values(ascending=False)
    rank_map = {cluster_idx: i for i, cluster_idx in enumerate(cluster_means.index)}
    
    if n_clusters == 2:
        names = ["🔴 관리 필요군", "🔵 건강 양호군"]
    elif n_clusters == 3:
        names = ["🔴 고위험군", "🟢 일반군", "🔵 건강 우수군"]
    else:
        names = ["🔴 고위험군", "🟠 중점 관리군", "🟢 일반군", "🔵 건강 우수군"]
        
    df_agg['유형'] = df_agg['Cluster'].map(rank_map).apply(lambda x: names[int(x)] if x < len(names) else "⚪ 기타")
    
    inv_map = {v: k for k, v in valid_cols.items()}
    df_agg = df_agg.rename(columns=inv_map).rename(columns={'표시용이름': '학교(연도)'})
    
    return df_agg

# ─── 4. 통합 렌더링 함수 ────────────────────────────────────────────────────
def render_dashboard(tab_df, valid_cols, filters):
    if filters['gender']: tab_df = tab_df[tab_df['성별'].isin(filters['gender'])]
    if filters['grade']: tab_df = tab_df[tab_df['학년'].isin(filters['grade'])]
    
    col_set1, col_set2 = st.columns([1, 3])
    metrics = list(valid_cols.keys())

    with col_set1:
        st.write("### ⚙️ 분석 지표")
        x_axis = st.selectbox("X축 (주로 BMI)", metrics, index=0)
        y_axis = st.selectbox("Y축 (주로 체력지표)", metrics, index=min(1, len(metrics)-1))
        n_clusters = st.slider("군집 세분화 (개)", 2, 4, 3)
        is_mobile = st.toggle("📱 모바일 최적화")
        
    plot_df = get_clustered_df(tab_df, valid_cols, x_axis, y_axis, n_clusters)

    if plot_df.empty:
        st.warning("⚠️ 분석을 수행하기 위한 데이터가 부족합니다.")
        return

    if filters['year']: plot_df = plot_df[plot_df['연도'].isin(filters['year'])]
    if filters['region']: plot_df = plot_df[plot_df['시군'].isin(filters['region'])]
    if filters['school']: plot_df = plot_df[plot_df['순수학교명'].isin(filters['school'])]
        
    if plot_df.empty:
        st.info("💡 선택하신 필터 조건에 맞는 데이터가 없습니다.")
        return

    with col_set2:
        color_map = {
            "🔴 고위험군": "#EF5350", "🔴 관리 필요군": "#EF5350", "🟠 중점 관리군": "#FFB74D", 
            "🟢 일반군": "#66BB6A", "🔵 건강 우수군": "#42A5F5", "🔵 건강 양호군": "#42A5F5"
        }
        fig = px.scatter(
            plot_df, x=x_axis, y=y_axis, color='유형', text='학교(연도)',
            hover_name='학교(연도)', color_discrete_map=color_map,
            title=f"🏫 통합 건강 데이터 AI 분석 결과"
        )
        fig.update_traces(textposition='top center', marker=dict(size=12 if is_mobile else 22, line=dict(width=2, color='white')))
        fig.update_layout(height=550, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    # 💡 [핵심 해결] 브라우저 창 크기가 작아져도 깨지지 않는 완전 고정형 HTML 표 삽입
    st.markdown("---")
    st.write("### 📋 그룹별 맞춤형 운동 처방 및 교육 프로그램")

    sum_df = plot_df.groupby('유형')[[x_axis, y_axis]].mean().round(1)
    counts = plot_df['유형'].value_counts()

    # CSS 스타일을 적용한 테이블 뼈대 생성
    html_table = '''
    <style>
    .presc-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; font-family: 'Malgun Gothic', sans-serif; }
    .presc-table th { background-color: #f8f9fa; padding: 15px; text-align: left; font-size: 16px; border-bottom: 2px solid #dee2e6; color: #333; }
    .presc-table td { padding: 15px; border-bottom: 1px solid #dee2e6; vertical-align: top; font-size: 14px; line-height: 1.6; color: #444; }
    .t-title { font-weight: bold; margin-bottom: 10px; display: inline-block; font-size: 15px; }
    .t-red { color: #c62828; background: #ffebee; padding: 5px 10px; border-radius: 5px; }
    .t-orange { color: #ef6c00; background: #fff3e0; padding: 5px 10px; border-radius: 5px; }
    .t-green { color: #2e7d32; background: #e8f5e9; padding: 5px 10px; border-radius: 5px; }
    .t-blue { color: #1565c0; background: #e3f2fd; padding: 5px 10px; border-radius: 5px; }
    .t-ul { margin-top: 5px; margin-bottom: 0; padding-left: 20px; }
    .t-ul li { margin-bottom: 6px; }
    </style>
    <table class="presc-table">
        <tr>
            <th style="width: 20%;">📊 분석 집단군</th>
            <th style="width: 40%;">🏃‍♂️ 맞춤형 운동 처방</th>
            <th style="width: 40%;">💊 교육 프로그램 추천</th>
        </tr>
    '''

    # 각 그룹별로 행(Row) 생성
    for idx in sum_df.index:
        count_val = counts.get(idx, 0)
        mean_val = sum_df.loc[idx, x_axis]
        
        # 색상 및 내용 매핑
        if "🔴" in idx:
            badge = "t-red"
            ex_title = "저강도 유산소 위주 구성"
            ex_list = "<li>관절 무리 없는 걷기, 수영 권장</li><li>실내 자전거로 기초 체력 증진</li><li>무리한 근력 운동 지양</li>"
            ed_title = "건강체력교실 우선 배정"
            ed_list = "<li>전문 강사 집중 체력 관리</li><li>가정통신문 연계 모니터링</li><li>식습관 및 영양 상담 정기 진행</li>"
        elif "🟠" in idx:
            badge = "t-orange"
            ex_title = "활동량 증대 집중"
            ex_list = "<li>흥미 유발 신체 활동 병행</li><li>일상적인 움직임 늘리기</li><li>비만 단계 진입 적극 예방</li>"
            ed_title = "교내 걷기 챌린지 참여"
            ed_list = "<li>방과 후 스포츠클럽 가입 권장</li><li>또래와 재미 위주 활동 도입</li><li>신체 활동 마일리지 보상</li>"
        elif "🟢" in idx:
            badge = "t-green"
            ex_title = "전신 밸런스 유지"
            ex_list = "<li>현재 기초 체력 수준 유지</li><li>신체 부위별 골고루 발달</li><li>주기적인 체력 모니터링</li>"
            ed_title = "정규 체육 수업 충실"
            ed_list = "<li>1일 1시간 이상 신체활동 권장</li><li>다양한 교내 스포츠 종목 체험</li><li>자발적이고 꾸준한 운동 습관화</li>"
        elif "🔵" in idx:
            badge = "t-blue"
            ex_title = "고강도 심화 트레이닝"
            ex_list = "<li>심폐지구력/순발력 극대화</li><li>인터벌 트레이닝(HIIT) 소화</li><li>전문 개별 스포츠 기술 습득</li>"
            ed_title = "학생 스포츠 리더 선발"
            ed_list = "<li>교내 체육 동아리 멘토 위촉</li><li>학교 대표 선수단 선발 시 우대</li><li>지역 엘리트 체육 프로그램 연계</li>"
        else:
            continue

        # HTML 행(Row) 추가
        row_html = f'''
        <tr>
            <td>
                <div class="t-title {badge}">{idx}</div>
                <div style="margin-top: 10px; font-size: 13.5px; color: #555;">
                    <b>{count_val}개 데이터</b><br>{x_axis} 평균: {mean_val}
                </div>
            </td>
            <td>
                <div class="t-title">[{ex_title}]</div>
                <ul class="t-ul">{ex_list}</ul>
            </td>
            <td>
                <div class="t-title">[{ed_title}]</div>
                <ul class="t-ul">{ed_list}</ul>
            </td>
        </tr>
        '''
        html_table += row_html

    html_table += "</table>"
    
    # 생성된 완전 고정형 표 출력
    st.markdown(html_table, unsafe_allow_html=True)

    with st.expander("🔍 상세 데이터 테이블 보기"):
        st.dataframe(plot_df.drop(columns=['순수학교명', '연도', '시군'], errors='ignore').sort_values(['유형', '학교(연도)']), use_container_width=True)

# ─── 5. 메인 실행 로직 ─────────────────────────────────────────────────────────────
raw_df, meta = load_raw_data()
if raw_df is not None:
    st.markdown("### 📍 분석 데이터 필터링")
    
    years = sorted([y for y in raw_df['연도'].unique() if y > 0])
    sigungus = sorted(raw_df['시군'].unique())
    grades = sorted([g for g in raw_df['학년'].unique() if '전체' not in g])
    genders = sorted([g for g in raw_df['성별'].unique() if '전체' not in g])

    f1, f2, f3, f4, f5 = st.columns(5)
    with f1: s_year = st.multiselect("📅 연도", options=years, placeholder="전체")
    with f2: s_region = st.multiselect("📍 시·군", options=sigungus, placeholder="전체")
    with f3: s_grade = st.multiselect("🎓 학년", options=grades, placeholder="전체")
    with f4: s_gender = st.multiselect("👫 성별", options=genders, placeholder="남/여 전체")
    
    tmp = raw_df.copy()
    if s_year: tmp = tmp[tmp['연도'].isin(s_year)]
    if s_region: tmp = tmp[tmp['시군'].isin(s_region)]
    if s_grade: tmp = tmp[tmp['학년'].isin(s_grade)]
    if s_gender: tmp = tmp[tmp['성별'].isin(s_gender)]
    f_schools = sorted(tmp['순수학교명'].unique())
    
    with f5: s_school = st.multiselect("🏫 학교명", options=f_schools, placeholder="전체")

    filters = {'year': s_year, 'region': s_region, 'grade': s_grade, 'gender': s_gender, 'school': s_school}

    # 필터가 적용된 통합 리포트 렌더링
    render_dashboard(raw_df, meta['valid_cols'], filters)
