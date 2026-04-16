import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── 1. 웹페이지 설정 및 고급 UI 스타일링 ──────────────────────────────────────────
st.set_page_config(page_title="PAPS Care+ Dashboard", layout="wide", initial_sidebar_state="expanded")

# 전문 사이트 느낌을 위한 CSS (헤더, 카드 디자인, 폰트, 여백 최적화)
st.markdown("""
    <style>
    /* 전체 배경색 및 폰트 */
    .main { background-color: #f8f9fa; }
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', sans-serif; }
    
    /* 최상단 전문 헤더 디자인 */
    .top-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* 분석 지표 설정 박스 (카드 UI) */
    .setting-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }
    
    /* 처방 리포트 테이블 스타일 */
    .report-table { width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .report-table th { background-color: #f1f5f9; padding: 18px; text-align: left; font-size: 15px; border-bottom: 2px solid #e2e8f0; color: #475569; }
    .report-table td { padding: 22px 18px; border-bottom: 1px solid #f1f5f9; vertical-align: top; }
    
    /* 텍스트 스타일링 */
    .badge { padding: 5px 12px; border-radius: 8px; font-weight: 700; font-size: 13px; display: inline-block; margin-bottom: 10px; }
    .status-red { background-color: #fee2e2; color: #dc2626; }
    .status-orange { background-color: #ffedd5; color: #ea580c; }
    .status-green { background-color: #dcfce7; color: #16a34a; }
    .status-blue { background-color: #dbeafe; color: #2563eb; }
    
    /* 리스트 스타일 */
    .content-list { margin: 0; padding-left: 20px; color: #4b5563; font-size: 14px; line-height: 1.7; }
    .content-list li { margin-bottom: 8px; }
    </style>
""", unsafe_allow_html=True)

# 1줄 헤더 구성
st.markdown("""
    <div class="top-header">
        <h1 style="color: white; margin: 0; font-size: 2.2em;">📊 <b>PAPS CARE+</b> <span style="font-size: 0.5em; font-weight: 300; opacity: 0.9;">| 강원특별자치도 학교 데이터 AI 분석 시스템</span></h1>
    </div>
    <p style="color: #64748b; font-size: 0.95em; margin-top: -15px;">
        * 본 시스템은 <b>학교알리미</b> 공시 데이터를 기반으로 학생들의 건강체력평가(PAPS)를 AI로 분석한 결과를 제공합니다.
    </p>
""", unsafe_allow_html=True)

# ─── 2. 데이터 로드 및 전처리 ──────────────────────────────────────────────────
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
            '근력/근지구력': find_col(['악력', '팔굽혀', '말아올리기']),
            '유연성': find_col(['앉아윗몸', '유연성']),
            '순발력': find_col(['제자리멀리', '순발력'])
        }
        valid_cols = {k: v for k, v in target_map.items() if v}
        for col in valid_cols.values():
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
        
        df['순수학교명'] = df['추출학교명'].astype(str).str.strip() if '추출학교명' in df.columns else df.iloc[:,0]
        df['연도'] = pd.to_numeric(df['연도'], errors='coerce').fillna(0).astype(int) if '연도' in df.columns else 0
        df['시군'] = df['시군'].astype(str).str.strip() if '시군' in df.columns else '강원'
        df['성별'] = df[find_col(['성별','남여'])] if find_col(['성별','남여']) else '전체'
        df['학년'] = df[find_col(['학년'])] if find_col(['학년']) else '전체'
        return df, {'valid_cols': valid_cols}
    except: return None, {}

# ─── 3. AI 분석 로직 ──────────────────────────────────────────────────────────
def get_clustered_df(tab_df, valid_cols, x_axis, y_axis, n_clusters):
    raw_x, raw_y = valid_cols[x_axis], valid_cols[y_axis]
    df_agg = tab_df.groupby(['순수학교명', '연도', '시군']).agg({v:'mean' for v in valid_cols.values()}).reset_index()
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

# ─── 4. 사이드바 제어 (전문적인 필터링 창) ──────────────────────────────────────────
raw_df, meta = load_raw_data()
if raw_df is not None:
    with st.sidebar:
        st.markdown("### 🔍 데이터 필터링")
        st.write("분석 범위를 설정하세요.")
        s_year = st.multiselect("📅 연도", sorted(raw_df['연도'].unique()), placeholder="전체")
        s_region = st.multiselect("📍 시·군", sorted(raw_df['시군'].unique()), placeholder="전체")
        s_grade = st.multiselect("🎓 학년", sorted(raw_df['학년'].unique()), placeholder="전체")
        s_gender = st.multiselect("👫 성별", sorted(raw_df['성별'].unique()), placeholder="전체")
        
        tmp = raw_df.copy()
        if s_year: tmp = tmp[tmp['연도'].isin(s_year)]
        if s_region: tmp = tmp[tmp['시군'].isin(s_region)]
        s_school = st.multiselect("🏫 특정 학교명 검색", sorted(tmp['순수학교명'].unique()), placeholder="전체")

    # ─── 5. 메인 분석 리포트 ──────────────────────────────────────────────────────
    valid_keys = list(meta['valid_cols'].keys())
    
    # 지표 설정 카드
    with st.container():
        st.markdown('<div class="setting-card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 2, 2])
        with c1: x_ax = st.selectbox("수평축 지표 (X)", valid_keys, index=0)
        with c2: y_ax = st.selectbox("수직축 지표 (Y)", valid_keys, index=1)
        with c3: n_cl = st.slider("AI 군집 세분화 수준", 2, 4, 3)
        st.markdown('</div>', unsafe_allow_html=True)

    plot_df = get_clustered_df(raw_df, meta['valid_cols'], x_ax, y_ax, n_cl)
    
    if not plot_df.empty:
        # 필터링 적용
        if s_year: plot_df = plot_df[plot_df['연도'].isin(s_year)]
        if s_region: plot_df = plot_df[plot_df['시군'].isin(s_region)]
        if s_school: plot_df = plot_df[plot_df['순수학교명'].isin(s_school)]

        # 메인 시각화
        fig = px.scatter(plot_df, x=x_ax, y=y_ax, color='유형', text='순수학교명', hover_name='순수학교명',
                         color_discrete_map={"🔴 관리 필요":"#EF5350","🔴 고위험":"#EF5350","🟠 중점관리":"#FFB74D","🟢 일반":"#66BB6A","🔵 우수":"#42A5F5","🔵 건강 양호":"#42A5F5"})
        fig.update_layout(height=550, plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        # 가로형 맞춤 처방 테이블 (HTML 고정형)
        st.markdown("<br>### 📋 그룹별 맞춤 처방 및 프로그램", unsafe_allow_html=True)
        sum_df = plot_df.groupby('유형')[[x_ax, y_ax]].mean().round(1)
        
        table_html = '''
        <table class="report-table">
            <tr>
                <th style="width: 20%;">📊 분석 집단군</th>
                <th style="width: 40%;">🏃‍♂️ 맞춤형 운동 처방</th>
                <th style="width: 40%;">💊 교육 프로그램 추천</th>
            </tr>
        '''
        
        for idx in sum_df.index:
            badge_class = "status-red" if "🔴" in idx else "status-orange" if "🟠" in idx else "status-green" if "🟢" in idx else "status-blue"
            
            # 내용 매핑
            presc_data = {
                "🔴": ("저강도 유산소 및 기초 체력", "<li>무리가 가지 않는 걷기, 수영 권장</li><li>실내 자전거를 통한 유산소 운동</li><li>급격한 근력 운동보다 체지방 감소 유도</li>", 
                       "건강체력교실 우선 배정", "<li>전문 강사의 1:1 집중 관리</li><li>가정 연계 식습관 모니터링</li><li>영양 교육 및 생활 습관 상담</li>"),
                "🟠": ("뉴스포츠를 통한 활동량 증대", "<li>재미 중심의 뉴스포츠 활동</li><li>일상적인 신체 활동량 늘리기</li><li>기초 대사량 향상을 위한 움직임 유도</li>", 
                       "방과 후 스포츠클럽 가입", "<li>동료 학생들과 함께하는 팀 스포츠</li><li>신체 활동 마일리지 시스템 활용</li><li>자존감 향상을 위한 성취 경험 제공</li>"),
                "🟢": ("전신 밸런스 및 체력 유지", "<li>신체 각 부위별 균형 발달 운동</li><li>전신 유연성 및 근력 강화 활동</li><li>현재의 우수한 체력 수준 꾸준히 유지</li>", 
                       "정규 체육 수업 및 자발적 운동", "<li>1일 1시간 이상 신체 활동 습관화</li><li>다양한 종목 체험 기회 확대</li><li>주기적인 개인 체력 측정 관리</li>"),
                "🔵": ("심화 트레이닝 및 전문 기술", "<li>고강도 인터벌 트레이닝(HIIT)</li><li>순발력과 순발력을 극대화하는 운동</li><li>개별 관심 종목의 전문 기술 습득</li>", 
                       "학생 리더 및 선수단 연계", "<li>체육 동아리 멘토 리더로 선발</li><li>학교 대표 선수단 선발 시 가산점</li><li>지역 사회 엘리트 체육 프로그램 연계</li>")
            }[idx[0]]
            
            table_html += f'''
            <tr>
                <td>
                    <span class="badge {badge_class}">{idx}</span><br>
                    <small style="color:#64748b;">{x_ax} 평균: {sum_df.loc[idx, x_ax]}<br>{len(plot_df[plot_df['유형']==idx])}개 데이터</small>
                </td>
                <td>
                    <div style="font-weight:700; color:#334155; margin-bottom:8px;">[{presc_data[0]}]</div>
                    <ul class="content-list">{presc_data[1]}</ul>
                </td>
                <td>
                    <div style="font-weight:700; color:#334155; margin-bottom:8px;">[{presc_data[2]}]</div>
                    <ul class="content-list">{presc_data[3]}</ul>
                </td>
            </tr>
            '''
        
        st.markdown(table_html + '</table>', unsafe_allow_html=True)
