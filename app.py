import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── 1. 웹페이지 설정 및 고급 UI 스타일링 ──────────────────────────────────────────
st.set_page_config(page_title="PAPS Care+ Dashboard", layout="wide", initial_sidebar_state="expanded")

# 전문적인 사이트 느낌을 위한 커스텀 CSS
st.markdown("""
    <style>
    /* 전체 배경색 및 폰트 */
    .main { background-color: #f0f2f6; }
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', sans-serif; }
    
    /* 최상단 헤더 디자인 */
    .top-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* 카드형 컨테이너 디자인 */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* 가로형 리포트 테이블 스타일 */
    .report-table { width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; }
    .report-table th { background-color: #f9fafb; padding: 18px; text-align: left; font-size: 15px; border-bottom: 2px solid #e5e7eb; color: #374151; }
    .report-table td { padding: 20px 18px; border-bottom: 1px solid #f3f4f6; vertical-align: top; }
    
    /* 상태 뱃지 */
    .badge { padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 12px; display: inline-block; margin-bottom: 8px; }
    .status-red { background-color: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; }
    .status-orange { background-color: #ffedd5; color: #ea580c; border: 1px solid #fdba74; }
    .status-green { background-color: #dcfce7; color: #16a34a; border: 1px solid #86efac; }
    .status-blue { background-color: #dbeafe; color: #2563eb; border: 1px solid #93c5fd; }
    
    /* 리스트 스타일 */
    .presc-list { margin: 0; padding-left: 1.2rem; color: #4b5563; font-size: 14px; line-height: 1.7; }
    </style>
""", unsafe_allow_html=True)

# ─── 2. 헤더 및 데이터 로드 ──────────────────────────────────────────────────
st.markdown("""
    <div class="top-header">
        <h1 style="color: white; margin: 0; font-size: 2.5rem; letter-spacing: -1px;">📊 <b>PAPS CARE+</b></h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 1.1rem; font-weight: 300;">강원특별자치도 학교 데이터 AI 분석 시스템</p>
    </div>
    <p style="color: #6b7280; font-size: 0.95rem; margin-top: -15px; margin-bottom: 30px;">
        * 본 분석 리포트는 <b>학교알리미</b> 공시 데이터를 기반으로 인공지능 군집 분석을 통해 도출되었습니다.
    </p>
""", unsafe_allow_html=True)

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

# ─── 3. 분석 로직 및 사이드바 ──────────────────────────────────────────────────
raw_df, meta = load_raw_data()
if raw_df is not None:
    with st.sidebar:
        st.markdown("<h2 style='font-size: 1.5rem;'>🔍 데이터 필터</h2>", unsafe_allow_html=True)
        s_year = st.multiselect("📅 연도", sorted(raw_df['연도'].unique()), placeholder="전체 연도")
        s_region = st.multiselect("📍 시군", sorted(raw_df['시군'].unique()), placeholder="전체 지역")
        s_grade = st.multiselect("🎓 학년", sorted(raw_df['학년'].unique()), placeholder="전체 학년")
        s_gender = st.multiselect("👫 성별", sorted(raw_df['성별'].unique()), placeholder="전체 성별")
        
        tmp = raw_df.copy()
        if s_year: tmp = tmp[tmp['연도'].isin(s_year)]
        if s_region: tmp = tmp[tmp['시군'].isin(s_region)]
        s_school = st.multiselect("🏫 학교명 검색", sorted(tmp['순수학교명'].unique()), placeholder="전체 학교")

    # 지표 설정 UI
    valid_keys = list(meta['valid_cols'].keys())
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1: x_ax = st.selectbox("📊 X축 지표 선택", valid_keys, index=0)
    with c2: y_ax = st.selectbox("🏃 Y축 지표 선택", valid_keys, index=1)
    with c3: n_cl = st.slider("🤖 AI 군집 세분화", 2, 4, 3)

    # 군집 분석 수행
    raw_x, raw_y = meta['valid_cols'][x_ax], meta['valid_cols'][y_ax]
    df_agg = raw_df.groupby(['순수학교명', '연도', '시군']).agg({v:'mean' for v in meta['valid_cols'].values()}).reset_index()
    X_data = df_agg[[raw_x, raw_y]].dropna()
    
    if len(X_data) >= n_cl:
        kmeans = KMeans(n_clusters=n_cl, random_state=42, n_init=10)
        df_agg.loc[X_data.index, 'Cluster'] = kmeans.fit_predict(StandardScaler().fit_transform(X_data))
        df_agg = df_agg.dropna(subset=['Cluster'])
        
        means = df_agg.groupby('Cluster')[raw_x].mean().sort_values(ascending=False)
        rank = {idx: i for i, idx in enumerate(means.index)}
        names = {2:["🔴 관리필요", "🔵 건강양호"], 3:["🔴 고위험", "🟢 일반", "🔵 우수"], 4:["🔴 고위험", "🟠 중점관리", "🟢 일반", "🔵 우수"]}[n_cl]
        df_agg['유형'] = df_agg['Cluster'].map(rank).apply(lambda x: names[int(x)] if x < len(names) else "⚪ 기타")
        
        # 필터 적용
        if s_year: df_agg = df_agg[df_agg['연도'].isin(s_year)]
        if s_region: df_agg = df_agg[df_agg['시군'].isin(s_region)]
        if s_school: df_agg = df_agg[df_agg['순수학교명'].isin(s_school)]

        # 메인 차트
        fig = px.scatter(df_agg, x=raw_x, y=raw_y, color='유형', text='순수학교명', hover_name='순수학교명',
                         labels={raw_x: x_ax, raw_y: y_ax},
                         color_discrete_map={"🔴 관리필요":"#EF5350","🔴 고위험":"#EF5350","🟠 중점관리":"#FFB74D","🟢 일반":"#66BB6A","🔵 우수":"#42A5F5","🔵 건강양호":"#42A5F5"})
        fig.update_layout(height=500, plot_bgcolor='white', margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        # 가로형 테이블 리포트 (이미지대로 수정)
        st.markdown("### 📋 그룹별 맞춤형 처방 리포트")
        sum_df = df_agg.groupby('유형')[[raw_x, raw_y]].mean().round(1)
        
        html_code = '<table class="report-table"><tr><th style="width:20%;">분석 집단군</th><th style="width:40%;">맞춤형 운동 처방</th><th style="width:40%;">교육 프로그램 추천</th></tr>'
        for idx in sum_df.index:
            b_class = "status-red" if "🔴" in idx else "status-orange" if "🟠" in idx else "status-green" if "🟢" in idx else "status-blue"
            presc = {
                "🔴": ("기초 대사량 증진 루틴", "<li>관절에 무리가 없는 수중 운동 및 걷기</li><li>실내 자전거 등 고정식 저강도 유산소</li><li>급격한 운동보다 생활 신체활동 증대 유도</li>", "건강체력교실 집중 케어", "<li>전문 강사의 1:1 맞춤형 체력 관리</li><li>영양 상담 및 가정 연계 식단 모니터링</li>"),
                "🟠": ("흥미 중심 뉴스포츠 클럽", "<li>뉴스포츠(디스크골프, 킨볼 등) 참여</li><li>일일 신체활동 마일리지 제도 활용</li><li>심폐지구력 향상을 위한 단계적 트레이닝</li>", "방과 후 스포츠클럽 권장", "<li>또래 친구들과 함께하는 단체 신체활동</li><li>체육에 대한 긍정적 인식 변화 교육</li>"),
                "🟢": ("전신 밸런스 유지 트레이닝", "<li>신체 각 부위별 균형 있는 근력 운동</li><li>유연성 확보를 위한 스트레칭 루틴</li><li>현재의 건강 체력 수준 지속 모니터링</li>", "자율 체육 활동 습관화", "<li>1인 1운동 생활 습관 정착 지원</li><li>다양한 종목 체험을 통한 흥미 유지</li>"),
                "🔵": ("고강도 엘리트 스포츠 연계", "<li>고강도 인터벌 트레이닝(HIIT) 소화</li><li>전문 스포츠 기술 습득 및 심화 과정</li><li>개인별 맞춤형 근지구력 강화 세션</li>", "학생 스포츠 리더 선발", "<li>체육 동아리 멘토링 리더 활동</li><li>지역 사회 엘리트 체육 프로그램 연계</li>")
            }[idx[0]]
            
            html_code += f'<tr><td><span class="badge {b_class}">{idx}</span><br><small style="color:#6b7280;">{x_ax} 평균: {sum_df.loc[idx, raw_x]}<br>총 {len(df_agg[df_agg["유형"]==idx])}개교</small></td>'
            html_code += f'<td><div style="font-weight:700; color:#374151; margin-bottom:8px;">[{presc[0]}]</div><ul class="presc-list">{presc[1]}</ul></td>'
            html_code += f'<td><div style="font-weight:700; color:#374151; margin-bottom:8px;">[{presc[2]}]</div><ul class="presc-list">{presc[3]}</ul></td></tr>'
        
        st.markdown(html_code + '</table>', unsafe_allow_html=True)
