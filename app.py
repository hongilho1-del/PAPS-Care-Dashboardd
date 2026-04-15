import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── 1. 웹페이지 설정 ───────────────────────────────────────────────────────
st.set_page_config(page_title="PAPS Care+ Real-time Dashboard", layout="wide")
# ★ 수정: 강원도 -> 강원특별자치도
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

        def find_col(keywords):
            for c in df.columns:
                for kw in keywords:
                    if kw in str(c): return c
            return None

        bmi_col      = find_col(['BMI', '비만', '체질량'])
        run_col      = find_col(['왕복', '오래달리기', '심폐'])
        strength_col = find_col(['악력', '근력'])
        flex_col     = find_col(['앉아윗몸', '유연성'])
        jump_col     = find_col(['제자리멀리', '순발력'])

        target_cols = {
            'BMI': bmi_col, '심폐지구력': run_col, '근력': strength_col, '유연성': flex_col, '순발력': jump_col
        }
        valid_cols = {k: v for k, v in target_cols.items() if v}

        for k, col in valid_cols.items():
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

        school_col = '추출학교명' if '추출학교명' in df.columns else df.columns[0]
        df['연도'] = df[school_col].astype(str).str.extract(r'(20\d{2}|19\d{2})')[0]
        df['연도'] = pd.to_numeric(df['연도'], errors='coerce')
        df['순수학교명'] = df[school_col].astype(str).str.replace(r'\s*(20\d{2}|19\d{2})\s*', '', regex=True).str.strip()

        return df, {'school_col': school_col, 'valid_cols': valid_cols}
    except Exception as e:
        st.error(f"데이터를 읽는 중 오류가 발생했습니다: {e}")
        return None, {}

# ─── 3. 학교별 집계 및 군집 명칭 자동 할당 로직 ──────────────────────────────
def get_clustered_df(tab_df, valid_cols, x_axis, y_axis, n_clusters):
    agg_dict = {v: 'mean' for v in valid_cols.values()}
    df = tab_df.groupby('순수학교명').agg(agg_dict).dropna(subset=[valid_cols[x_axis], valid_cols[y_axis]]).reset_index().round(1)
    df.columns = ['학교명'] + list(valid_cols.keys())

    # 스케일링 및 군집화
    X = df[[x_axis, y_axis]]
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # 💡 [핵심 로직] BMI가 높고 체력(Y축)이 낮은 그룹을 찾아 '고위험군'으로 자동 정렬
    sort_metric = 'BMI' if 'BMI' in df.columns else x_axis
    cluster_means = df.groupby('Cluster')[sort_metric].mean().sort_values(ascending=False)
    
    # 순위별 명칭 맵핑 (BMI가 가장 높은 그룹이 고위험군)
    rank_map = {cluster_idx: i for i, cluster_idx in enumerate(cluster_means.index)}
    name_list = ["🔴 고위험군", "🟠 중점 관리군", "🔵 건강 우수군", "🟢 일반군", "⚪ 기타"]
    
    df['유형'] = df['Cluster'].map(rank_map).apply(lambda x: name_list[min(x, 4)])
    return df

# ─── 4. 탭별 렌더링 함수 ──────────────────────────────────────────────────────
def render_tab(tab_df, tab_label, valid_cols):
    st.subheader(f"📍 {tab_label} 분석 리포트")
    
    col_set1, col_set2 = st.columns([1, 3])
    metrics = list(valid_cols.keys())

    with col_set1:
        st.write("### ⚙️ 분석 지표")
        x_axis = st.selectbox("X축 (주로 BMI)", metrics, index=0, key=f"x_{tab_label}")
        y_axis = st.selectbox("Y축 (주로 심폐지구력)", metrics, index=min(1, len(metrics)-1), key=f"y_{tab_label}")
        n_clusters = st.slider("군집 세분화 (개)", 2, 4, 3, key=f"n_{tab_label}")
        
    plot_df = get_clustered_df(tab_df, valid_cols, x_axis, y_axis, n_clusters)

    with col_set2:
        color_discrete_map = {"🔴 고위험군": "#EF5350", "🟠 중점 관리군": "#FFB74D", "🔵 건강 우수군": "#42A5F5", "🟢 일반군": "#66BB6A"}
        fig = px.scatter(
            plot_df, x=x_axis, y=y_axis, color='유형', text='학교명',
            hover_name='학교명', color_discrete_map=color_discrete_map,
            # ★ 수정: 그래프 타이틀에 강원특별자치도 반영
            title=f"🏫 강원특별자치도 {tab_label} 학교별 건강 등급 분포"
        )
        fig.update_traces(textposition='top center', marker=dict(size=18, line=dict(width=2, color='white')))
        fig.update_layout(height=550, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    # ── 유형별 분석 및 처방 ───────────────────────────────────────────────────
    st.markdown("---")
    st.write("### 📋 유형별 특성 및 정책 제언")
    
    sum_df = plot_df.groupby('유형')[metrics].mean().round(1)
    counts = plot_df['유형'].value_counts()
    
    cols = st.columns(len(sum_df))
    for i, (idx, row) in enumerate(sum_df.iterrows()):
        with cols[i]:
            st.metric(label=idx, value=f"{counts[idx]}개교")
            if "🔴" in idx:
                st.error("🚨 **관리 비상**\n\n비만도 해소 및 기초 체력 증진을 위한 긴급 예산 및 방과 후 스포츠 프로그램 집중 지원이 필요합니다.")
            elif "🟠" in idx:
                st.warning("⚠️ **예방 필요**\n\n현재 수준 유지 시 고위험군 진입 우려. 규칙적인 체력 측정과 균형 잡힌 식단 교육 병행 권장.")
            else:
                st.info("✅ **우수 사례**\n\n해당 학교의 체육 활동 우수 사례를 수집하여 타 학교에 전파할 수 있는 벤치마킹 대상으로 선정.")
            st.write(row)

    with st.expander("🔍 상세 데이터 테이블 보기"):
        st.dataframe(plot_df.sort_values('유형'), use_container_width=True)

# ─── 5. 메인 실행 ─────────────────────────────────────────────────────────────
raw_df, meta = load_raw_data()
if raw_df is not None:
    valid_cols = meta['valid_cols']
    years = sorted(raw_df['연도'].dropna().unique().astype(int).tolist())
    
    tab_labels = ["🌐 전체보기"] + [f"📅 {y}년" for y in years]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        # ★ 수정: 강원도 전체 -> 강원특별자치도 전체
        render_tab(raw_df, "강원특별자치도 전체", valid_cols)
    
    for i, y in enumerate(years):
        with tabs[i+1]:
            render_tab(raw_df[raw_df['연도'] == y], f"{y}년도", valid_cols)
