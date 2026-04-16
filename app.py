import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── 1. 웹페이지 설정 ───────────────────────────────────────────────────────
st.set_page_config(page_title="PAPS Care+ Real-time Dashboard", layout="wide", initial_sidebar_state="collapsed")
st.title("📊 PAPS Care+ : 강원특별자치도 학교 실데이터 AI 분석 시스템")

# ─── 2. 데이터 로드 ──────────────────────────────────────────────────────────
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

        for k, col in valid_cols.items():
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

        school_col = '추출학교명' if '추출학교명' in df.columns else df.columns[0]
        df['순수학교명'] = df[school_col].astype(str).str.strip()
        
        if '연도' in df.columns:
            df['연도'] = pd.to_numeric(df['연도'], errors='coerce').fillna(0).astype(int)
        else:
            df['연도'] = df[school_col].astype(str).str.extract(r'(20\d{2}|19\d{2})')[0]
            df['연도'] = pd.to_numeric(df['연도'], errors='coerce').fillna(0).astype(int)

        # 시각화용 이름 생성 (예: 솔올중 (2023))
        df['표시용이름'] = df.apply(lambda row: f"{row['순수학교명']} ({row['연도']})" if row['연도'] > 0 else row['순수학교명'], axis=1)

        if '시군' not in df.columns:
            df['시군'] = '강원'

        return df, {'school_col': school_col, 'valid_cols': valid_cols}
    except Exception as e:
        st.error(f"데이터를 읽는 중 오류가 발생했습니다: {e}")
        return None, {}

# ─── 3. 학교+연도별 집계 및 군집 명칭 자동 할당 ──────────────────────────────────
def get_clustered_df(tab_df, valid_cols, x_axis, y_axis, n_clusters):
    agg_dict = {v: 'mean' for v in valid_cols.values()}
    df = tab_df.groupby('표시용이름').agg(agg_dict).reset_index().round(1)
    
    df.columns = ['학교(연도)'] + list(valid_cols.keys())
    df = df.dropna(subset=[x_axis, y_axis])

    if len(df) < n_clusters: return df

    X = df[[x_axis, y_axis]]
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    sort_metric = 'BMI' if 'BMI' in df.columns else x_axis
    # BMI가 높은 순서대로 나쁜 등급이 되도록 정렬
    cluster_means = df.groupby('Cluster')[sort_metric].mean().sort_values(ascending=False)
    
    rank_map = {cluster_idx: i for i, cluster_idx in enumerate(cluster_means.index)}
    # 논리적 순서: 고위험 -> 중점관리 -> 일반 -> 우수
    name_list = ["🔴 고위험군", "🟠 중점 관리군", "🟢 일반군", "🔵 건강 우수군", "⚪ 기타"]
    df['유형'] = df['Cluster'].map(rank_map).apply(lambda x: name_list[min(x, 4)])
    return df

# ─── 4. 탭별 렌더링 함수 ──────────────────────────────────────────────────────
def render_tab(tab_df, tab_label, valid_cols, unique_key):
    st.subheader(f"📍 {tab_label} 분석 리포트")
    
    col_set1, col_set2 = st.columns([1, 3])
    metrics = list(valid_cols.keys())

    with col_set1:
        st.write("### ⚙️ 분석 지표")
        x_axis = st.selectbox("X축 (주로 BMI)", metrics, index=0, key=f"x_{unique_key}")
        y_axis = st.selectbox("Y축 (주로 체력지표)", metrics, index=min(1, len(metrics)-1), key=f"y_{unique_key}")
        n_clusters = st.slider("군집 세분화 (개)", 2, 4, 3, key=f"n_{unique_key}")
        is_mobile = st.toggle("📱 모바일 최적화", key=f"mob_{unique_key}")
        
    plot_df = get_clustered_df(tab_df, valid_cols, x_axis, y_axis, n_clusters)

    if plot_df.empty or len(plot_df) < n_clusters:
        st.warning("⚠️ 지표 데이터가 부족하여 AI 분석을 수행할 수 없습니다.")
        return

    marker_size = 10 if is_mobile else 18
    chart_height = 400 if is_mobile else 600

    with col_set2:
        color_discrete_map = {"🔴 고위험군": "#EF5350", "🟠 중점 관리군": "#FFB74D", "🟢 일반군": "#66BB6A", "🔵 건강 우수군": "#42A5F5"}
        fig = px.scatter(
            plot_df, x=x_axis, y=y_axis, color='유형', 
            text='학교(연도)', 
            hover_name='학교(연도)', 
            color_discrete_map=color_discrete_map,
            title=f"🏫 {tab_label} 건강 등급 분포 (학교별/연도별)"
        )
        
        fig.update_traces(
            textposition='top center', 
            textfont_size=9 if is_mobile else 11,
            marker=dict(size=marker_size, line=dict(width=1, color='white')) 
        )
        fig.update_layout(
            height=chart_height,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # 💡 [복원 및 업그레이드] AI 맞춤형 처방 프로그램 영역
    st.markdown("---")
    st.write("### 📋 AI 맞춤형 처방 프로그램 및 운동 방향")
    
    sum_df = plot_df.groupby('유형')[[x_axis, y_axis]].mean().round(1)
    counts = plot_df['유형'].value_counts()
    
    # 1. 지표 요약 (가로 나열)
    cols = st.columns(len(sum_df))
    for i, (idx, row) in enumerate(sum_df.iterrows()):
        with cols[i]:
            st.metric(label=idx, value=f"{counts.get(idx, 0)}개 데이터", delta=f"{x_axis}: {row[x_axis]} / {y_axis}: {row[y_axis]}", delta_color="off")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. 상세 처방 카드 (2단 레이아웃)
    card_col1, card_col2 = st.columns(2)
    
    for idx in sum_df.index:
        val_x = sum_df.loc[idx, x_axis]
        if "🔴" in idx:
            with card_col1:
                st.error(f"#### {idx}\n"
                         f"**🔻 상태:** 비만도({x_axis} 평균 {val_x})가 상대적으로 높고, 기초 체력이 저조하여 집중 관리가 시급합니다.\n\n"
                         f"**🏃‍♂️ 운동 방향:** 관절에 무리가 없는 저강도 유산소 운동(수영, 실내 자전거, 걷기) 위주의 세션 구성이 필요합니다.\n\n"
                         f"**💊 처방 프로그램:** '건강 체력 교실' 1순위 배정, 당음료 섭취 제한 등 가정통신문 연계 영양 상담 병행.")
        elif "🟠" in idx:
            with card_col2:
                st.warning(f"#### {idx}\n"
                           f"**🔻 상태:** 체력 저하가 진행 중이거나 과체중 경계 단계에 있어 사전 예방이 필요한 그룹입니다.\n\n"
                           f"**🏃‍♂️ 운동 방향:** 흥미를 유발할 수 있는 뉴스포츠나 그룹 활동을 통해 일상적인 신체 활동량 증가를 유도해야 합니다.\n\n"
                           f"**💊 처방 프로그램:** 방과 후 스포츠 클럽 참여 적극 권장, 교내 걷기 챌린지(만보걷기 등) 프로그램 도입.")
        elif "🟢" in idx:
            with card_col1:
                st.success(f"#### {idx}\n"
                           f"**🔻 상태:** 표준적인 체력과 체격을 유지하고 있는 가장 안정적인 밸런스의 그룹입니다.\n\n"
                           f"**🏃‍♂️ 운동 방향:** 현재의 신체 활동 수준을 유지하며, 근력과 유연성을 고르게 발달시키는 전신 운동을 권장합니다.\n\n"
                           f"**💊 처방 프로그램:** 정규 체육 수업의 적극적 참여 독려, 1일 1시간 이상 일상적 신체활동 습관화 지속.")
        elif "🔵" in idx:
            with card_col2:
                st.info(f"#### {idx}\n"
                        f"**🔻 상태:** 비만도가 낮고 체력 지표가 매우 우수하며 균형 잡힌 뛰어난 신체 능력을 보유하고 있습니다.\n\n"
                        f"**🏃‍♂️ 운동 방향:** 전문적인 스포츠 기술 습득 및 고강도 인터벌 트레이닝(HIIT) 등 심화 과정을 소화할 수 있습니다.\n\n"
                        f"**💊 처방 프로그램:** 학교 대표 스포츠 선수단 선발, 체육 동아리 멘토/리더 역할 부여, 지역 엘리트 체육 연계 지원.")

    with st.expander("🔍 상세 데이터 테이블 보기"):
        st.dataframe(plot_df.sort_values(['유형', '학교(연도)']), use_container_width=True)

# ─── 5. 메인 실행 ─────────────────────────────────────────────────────────────
raw_df, meta = load_raw_data()
if raw_df is not None:
    valid_cols = meta['valid_cols']
    
    st.markdown("### 🔍 분석 뷰(View) 선택")
    view_option = st.radio("보기 방식", ["📅 연도별 비교", "📍 시·군별 비교"], horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    if view_option == "📅 연도별 비교":
        years = sorted([y for y in raw_df['연도'].unique() if y > 0])
        tab_labels = ["🌐 강원 전체보기"] + [f"📅 {y}년" for y in years]
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            render_tab(raw_df, "강원특별자치도 전체", valid_cols, "y_all")
        
        for i, y in enumerate(years):
            with tabs[i+1]:
                render_tab(raw_df[raw_df['연도'] == y], f"{y}년도", valid_cols, f"y_{y}")
                
    elif view_option == "📍 시·군별 비교":
        sigungus = sorted(raw_df['시군'].astype(str).unique().tolist())
        if '강원' in sigungus: sigungus.remove('강원')

        tab_labels = ["🌐 강원 전체보기"] + [f"📍 {sg}" for sg in sigungus]
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            render_tab(raw_df, "강원특별자치도 전체", valid_cols, "sg_all")
            
        for i, sg in enumerate(sigungus):
            with tabs[i+1]:
                render_tab(raw_df[raw_df['시군'].astype(str) == sg], f"{sg} 지역", valid_cols, f"sg_{sg}")
