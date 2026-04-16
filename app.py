import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── 1. 웹페이지 설정 및 하드코딩된 디자인(CSS) 적용 ──────────────────────────────────
st.set_page_config(page_title="PAPS Care+ Dashboard", layout="wide", initial_sidebar_state="expanded")

# 전문 사이트 느낌을 위한 강력한 CSS 주입
st.markdown("""
    <style>
    /* 배경 및 기본 폰트 */
    .main { background-color: #f8fafc; }
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', -apple-system, sans-serif; }

    /* 스트림릿 기본 요소 숨기기 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 상단 헤더 섹션 */
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

    /* 분석 카드 스타일 */
    .analysis-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
    }

    /* 가로형 리포트 테이블 (이미지 요청사항 반영) */
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
    
    /* 뱃지 스타일 */
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
        t_map = {
            'BMI': find_col(['BMI', '비만', '체질량']),
            '심폐지구력': find_col(['왕복', '오래달리기', '심폐']),
            '근력/근지구력': find_col(['악력', '팔굽혀', '말아올리기']),
            '유연성': find_col(['앉아윗몸', '유연성']),
            '순발력': find_col(['제자리멀리', '순발력'])
        }
        valid = {k: v for k, v in t_map.items() if v}
        for col in valid.values():
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
        df['순수학교명'] = df['추출학교명'].astype(str).str.strip() if '추출학교명' in df.columns else df.iloc[:,0]
        df['연도'] = pd.to_numeric(df['연도'], errors='coerce').fillna(0).astype(int) if '연도' in df.columns else 0
        df['시군'] = df['시군'].astype(str).str.strip() if '시군' in df.columns else '강원'
        df['성별'] = df[find_col(['성별','남여'])] if find_col(['성별','남여']) else '전체'
        df['학년'] = df[find_col(['학년'])] if find_col(['학년']) else '전체'
        return df, {'valid': valid}
    except: return None, {}

# ─── 3. 메인 로직 ─────────────────────────────────────────────────────────────
raw_df, meta = load_raw_data()
if raw_df is not None:
    # 1. 헤더 출력
    st.markdown("""
        <div class="hero-section">
            <div class="hero-title">📊 PAPS CARE+</div>
            <div class="hero-sub">강원특별자치도 학교 데이터 AI 분석 시스템</div>
        </div>
    """, unsafe_allow_html=True)

    # 2. 사이드바 (필터)
    with st.sidebar:
        st.markdown("### 🔍 데이터 필터링")
        s_year = st.multiselect("📅 연도", sorted(raw_df['연도'].unique()), placeholder="전체 연도")
        s_region = st.multiselect("📍 시·군", sorted(raw_df['시군'].unique()), placeholder="전체 지역")
        s_grade = st.multiselect("🎓 학년", sorted(raw_df['학년'].unique()), placeholder="전체 학년")
        s_gender = st.multiselect("👫 성별", sorted(raw_df['성별'].unique()), placeholder="전체 성별")
        
        tmp_df = raw_df.copy()
        if s_year: tmp_df = tmp_df[tmp_df['연도'].isin(s_year)]
        if s_region: tmp_df = tmp_df[tmp_df['시군'].isin(s_region)]
        s_school = st.multiselect("🏫 학교명 검색", sorted(tmp_df['순수학교명'].unique()), placeholder="전체 학교")

    # 3. 분석 설정 (카드 형태)
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1: x_ax = st.selectbox("수평축 지표 (X)", list(meta['valid'].keys()), index=0)
    with c2: y_ax = st.selectbox("수직축 지표 (Y)", list(meta['valid'].keys()), index=1)
    with c3: n_cl = st.slider("AI 군집 세분화", 2, 4, 3)
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. AI 군집 분석
    raw_x, raw_y = meta['valid'][x_ax], meta['valid'][y_ax]
    df_agg = raw_df.groupby(['순수학교명', '연도', '시군']).agg({v:'mean' for v in meta['valid'].values()}).reset_index()
    X = df_agg[[raw_x, raw_y]].dropna()
    
    if len(X) >= n_cl:
        kmeans = KMeans(n_clusters=n_cl, random_state=42, n_init=10)
        df_agg.loc[X.index, 'Cluster'] = kmeans.fit_predict(StandardScaler().fit_transform(X))
        df_agg = df_agg.dropna(subset=['Cluster'])
        
        # 필터 적용
        if s_year: df_agg = df_agg[df_agg['연도'].isin(s_year)]
        if s_region: df_agg = df_agg[df_agg['시군'].isin(s_region)]
        if s_school: df_agg = df_agg[df_agg['순수학교명'].isin(s_school)]

        # 차트 출력
        means = df_agg.groupby('Cluster')[raw_x].mean().sort_values(ascending=False)
        rank = {idx: i for i, idx in enumerate(means.index)}
        names = {2:["🔴 관리필요", "🔵 건강양호"], 3:["🔴 고위험", "🟢 일반", "🔵 우수"], 4:["🔴 고위험", "🟠 중점관리", "🟢 일반", "🔵 우수"]}[n_cl]
        df_agg['유형'] = df_agg['Cluster'].map(rank).apply(lambda x: names[int(x)] if x < len(names) else "⚪ 기타")

        fig = px.scatter(df_agg, x=raw_x, y=raw_y, color='유형', text='순수학교명', labels={raw_x:x_ax, raw_y:y_ax},
                         color_discrete_map={"🔴 관리필요":"#ef4444","🔴 고위험":"#ef4444","🟠 중점관리":"#f97316","🟢 일반":"#22c55e","🔵 우수":"#3b82f6","🔵 건강양호":"#3b82f6"})
        fig.update_layout(height=500, margin=dict(t=10, b=10, l=10, r=10), plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

        # 5. 가로형 테이블 리포트 (가장 중요한 부분)
        st.markdown("<h3 style='margin-top:40px;'>📋 그룹별 맞춤형 처방 리포트</h3>", unsafe_allow_html=True)
        sum_df = df_agg.groupby('유형')[[raw_x, raw_y]].mean().round(1)
        
        table_html = """
        <table class="paps-table">
            <tr>
                <th style="width: 20%;">📊 분석 집단군</th>
                <th style="width: 40%;">🏃‍♂️ 맞춤형 운동 처방</th>
                <th style="width: 40%;">💊 교육 프로그램 추천</th>
            </tr>
        """
        
        for idx in sum_df.index:
            t_class = "tag-red" if "🔴" in idx else "tag-orange" if "🟠" in idx else "tag-green" if "🟢" in idx else "tag-blue"
            p_data = {
                "🔴": ("기초 체력 증진 루틴", "- 관절 무리 없는 걷기, 수영 권장<br>- 실내 자전거 등 저강도 유산소 활동<br>- 급격한 운동보다 생활 속 움직임 유도", 
                       "건강체력교실 집중 케어", "- 전문 강사의 1:1 밀착 체력 관리<br>- 영양 상담 및 가정 식단 모니터링 연계"),
                "🟠": ("뉴스포츠 흥미 유발", "- 디스크골프, 킨볼 등 뉴스포츠 참여<br>- 일일 신체활동 마일리지 제도 활용<br>- 심폐지구력 향상을 위한 단계적 트레이닝", 
                       "방과 후 스포츠클럽 권장", "- 또래와 함께하는 팀 스포츠 활동 도입<br>- 체육에 대한 긍정적 인식 변화 교육"),
                "🟢": ("전신 밸런스 유지", "- 신체 부위별 균형 있는 근력 운동<br>- 유연성 확보를 위한 꾸준한 스트레칭<br>- 현재의 우수한 체력 수준 지속 모니터링", 
                       "자율 체육 활동 습관화", "- 1인 1운동 생활 습관 정착 지원<br>- 다양한 종목 체험을 통한 흥미 유지"),
                "🔵": ("고강도 심화 트레이닝", "- 고강도 인터벌 트레이닝(HIIT) 소화<br>- 전문 스포츠 기술 습득 및 심화 과정<br>- 개인별 맞춤형 근지구력 강화 세션", 
                       "학생 스포츠 리더 선발", "- 체육 동아리 멘토링 리더 활동 권장<br>- 지역 사회 엘리트 체육 프로그램 연계")
            }[idx[0]]

            table_html += f"""
            <tr>
                <td>
                    <span class="tag {t_class}">{idx}</span><br>
                    <div style="font-size:13px; color:#64748b;">{x_ax} 평균: {sum_df.loc[idx, raw_x]}</div>
                </td>
                <td><div class="content-box"><b>[{p_data[0]}]</b>{p_data[1]}</div></td>
                <td><div class="content-box"><b>[{p_data[2]}]</b>{p_data[3]}</div></td>
            </tr>
            """
        st.markdown(table_html + "</table>", unsafe_allow_html=True)
