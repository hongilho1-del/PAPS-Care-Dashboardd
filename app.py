import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── 1. 웹페이지 설정 및 고급 스타일링 (CSS) ──────────────────────────────────
st.set_page_config(page_title="PAPS Care+ Dashboard", layout="wide", initial_sidebar_state="expanded")

# 전문적인 사이트 느낌을 위한 커스텀 스타일 적용
st.markdown("""
    <style>
    /* 전체 배경색 및 폰트 설정 */
    .main { background-color: #f4f7f9; }
    h1, h2, h3 { color: #1e3a8a; font-family: 'Pretendard', sans-serif; }
    
    /* 최상단 헤더 바 디자인 */
    .header-bar {
        background-color: #1e3a8a;
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* 카드(컨테이너) 디자인 */
    .stMetric, .stPlotlyChart, .presc-table {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* 필터 사이드바 디자인 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e6ed;
    }
    
    /* 표(Table) 스타일 고도화 */
    .presc-table { width: 100%; border-collapse: collapse; overflow: hidden; }
    .presc-table th { background-color: #f8fafd; padding: 15px; text-align: left; border-bottom: 2px solid #edf2f7; color: #4a5568; }
    .presc-table td { padding: 20px 15px; border-bottom: 1px solid #edf2f7; vertical-align: top; }
    
    /* 뱃지 스타일 */
    .badge { padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 13px; display: inline-block; margin-bottom: 8px; }
    .bg-red { background-color: #ffebee; color: #c62828; }
    .bg-orange { background-color: #fff3e0; color: #e65100; }
    .bg-green { background-color: #e8f5e9; color: #1b5e20; }
    .bg-blue { background-color: #e3f2fd; color: #0d47a1; }
    </style>
""", unsafe_allow_html=True)

# 헤더 영역
st.markdown("""
    <div class="header-bar">
        <h1 style="color: white; margin: 0;">📊 PAPS CARE+</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.8;">강원특별자치도 학생 건강 데이터 분석 시스템</p>
    </div>
    <p style="color: #7f8c8d; font-size: 0.9em; margin-bottom: 30px;">
        * 본 시스템은 <b>학교알리미</b> 공시 데이터를 기반으로 학생들의 건강체력평가(PAPS)를 AI로 분석한 결과를 제공합니다.
    </p>
""", unsafe_allow_html=True)

# ─── 2. 데이터 로드 ──────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', 'PAPS_Combined_Data.xlsx')
    if not os.path.exists(file_path): return None, {}
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        def find_col(kws):
            for c in df.columns:
                for kw in kws:
                    if kw in str(c): return c
            return None
        target_map = {
            'BMI': find_col(['BMI', '비만', '체질량']),
            '심폐지구력': find_col(['왕복', '오래달리기', '심폐']),
            '근력(악력)': find_col(['악력']),
            '근력(복근)': find_col(['팔굽혀', '말아올리기']),
            '유연성': find_col(['앉아윗몸', '유연성']),
            '순발력': find_col(['제자리멀리', '순발력'])
        }
        valid_cols = {k: v for k, v in target_map.items() if v}
        for col in valid_cols.values():
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
        school_col = '추출학교명' if '추출학교명' in df.columns else df.columns[0]
        df['순수학교명'] = df[school_col].astype(str).str.strip()
        df['연도'] = pd.to_numeric(df['연도'], errors='coerce').fillna(0).astype(int) if '연도' in df.columns else 0
        df['표시용이름'] = df.apply(lambda r: f"{r['순수학교명']} ({r['연도']})" if r['연도']>0 else r['순수학교명'], axis=1)
        df['시군'] = df['시군'].astype(str).str.strip() if '시군' in df.columns else '강원'
        df['성별'] = df[find_col(['성별','남여'])] if find_col(['성별','남여']) else '전체'
        df['학년'] = df[find_col(['학년'])] if find_col(['학년']) else '전체'
        return df, {'valid_cols': valid_cols}
    except: return None, {}

# ─── 3. AI 군집 분석 ──────────────────────────────────────────
def get_clustered_df(tab_df, valid_cols, x_axis, y_axis, n_clusters):
    raw_x, raw_y = valid_cols[x_axis], valid_cols[y_axis]
    df_agg = tab_df.groupby(['순수학교명', '표시용이름']).agg({v:'mean' for v in valid_cols.values()}).reset_index()
    X = df_agg[[raw_x, raw_y]].dropna()
    if len(X) < n_clusters: return pd.DataFrame()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_agg.loc[X.index, 'Cluster'] = kmeans.fit_predict(StandardScaler().fit_transform(X))
    df_agg = df_agg.dropna(subset=['Cluster'])
    means = df_agg.groupby('Cluster')[raw_x].mean().sort_values(ascending=False)
    rank = {idx: i for i, idx in enumerate(means.index)}
    names = {2:["🔴 관리 필요", "🔵 건강 양호"], 3:["🔴 고위험", "🟢 일반", "🔵 우수"], 4:["🔴 고위험", "🟠 중점관리", "🟢 일반", "🔵 우수"]}[n_clusters]
    df_agg['유형'] = df_agg['Cluster'].map(rank).apply(lambda x: names[int(x)] if x < len(names) else "⚪ 기타")
    return df_agg.rename(columns={v:k for k,v in valid_cols.items()})

# ─── 4. 사이드바 필터 ──────────────────────────────────────────
raw_df, meta = load_raw_data()
if raw_df is not None:
    with st.sidebar:
        st.markdown("### 🔍 데이터 필터링")
        s_year = st.multiselect("📅 연도", sorted(raw_df['연도'].unique()), placeholder="전체")
        s_region = st.multiselect("📍 시군", sorted(raw_df['시군'].unique()), placeholder="전체")
        s_grade = st.multiselect("🎓 학년", sorted(raw_df['학년'].unique()), placeholder="전체")
        s_gender = st.multiselect("👫 성별", sorted(raw_df['성별'].unique()), placeholder="전체")
        
        tmp = raw_df.copy()
        if s_year: tmp = tmp[tmp['연도'].isin(s_year)]; 
        if s_region: tmp = tmp[tmp['시군'].isin(s_region)]
        s_school = st.multiselect("🏫 학교명", sorted(tmp['순수학교명'].unique()), placeholder="전체")

    # ─── 5. 메인 대시보드 ──────────────────────────────────────
    metrics = list(meta['valid_cols'].keys())
    c1, c2, c3 = st.columns([2,2,2])
    with c1: x_ax = st.selectbox("수평축 지표(X)", metrics, index=0)
    with c2: y_ax = st.selectbox("수직축 지표(Y)", metrics, index=1)
    with c3: n_cl = st.slider("군집 세분화", 2, 4, 3)

    plot_df = get_clustered_df(raw_df, meta['valid_cols'], x_ax, y_ax, n_cl)
    
    if not plot_df.empty:
        # 필터 적용
        if s_year: plot_df = plot_df[plot_df['연도'].isin(s_year)]
        if s_region: plot_df = plot_df[plot_df['시군'].isin(s_region)]
        if s_school: plot_df = plot_df[plot_df['순수학교명'].isin(s_school)]

        # 차트 영역
        fig = px.scatter(plot_df, x=x_ax, y=y_ax, color='유형', text='순수학교명', 
                         color_discrete_map={"🔴 관리 필요":"#EF5350","🔴 고위험":"#EF5350","🟠 중점관리":"#FFB74D","🟢 일반":"#66BB6A","🔵 우수":"#42A5F5","🔵 건강 양호":"#42A5F5"})
        fig.update_layout(height=500, margin=dict(t=10, b=10, l=10, r=10), plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

        # 처방 테이블 영역
        st.markdown("### 📋 그룹별 맞춤 처방 및 프로그램")
        sum_df = plot_df.groupby('유형')[[x_ax, y_ax]].mean().round(1)
        
        html_code = '<table class="presc-table"><tr><th>📊 분석 집단군</th><th>🏃‍♂️ 맞춤형 운동 처방</th><th>💊 교육 프로그램 추천</th></tr>'
        for idx in sum_df.index:
            color_class = "bg-red" if "🔴" in idx else "bg-orange" if "🟠" in idx else "bg-green" if "🟢" in idx else "bg-blue"
            presc = {
                "🔴": ["저강도 유산소(걷기, 수영)","건강체력교실 및 식단 상담"],
                "🟠": ["뉴스포츠 활동량 증대","방과후 스포츠클럽 권장"],
                "🟢": ["전신 밸런스 및 근력 유지","정규 체육 수업 및 자율 운동"],
                "🔵": ["고강도 심화 트레이닝","학생 리더 선발 및 엘리트 연계"]
            }[idx[0]]
            
            html_code += f'<tr><td><span class="badge {color_class}">{idx}</span><br><small>{x_ax} 평균: {sum_df.loc[idx, x_ax]}</small></td>'
            html_code += f'<td><b>{presc[0]}</b><br>심폐지구력 향상과 기초 대사량 증진을 목표로 구성합니다.</td>'
            html_code += f'<td><b>{presc[1]}</b><br>지속 가능한 건강 습관 형성을 위한 교육 연계 프로그램입니다.</td></tr>'
        
        st.markdown(html_code + '</table>', unsafe_allow_html=True)
