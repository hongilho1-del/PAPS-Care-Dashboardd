import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── 1. 웹페이지 설정 ───────────────────────────────────────────────────────
st.set_page_config(page_title="PAPS Care+ Real-time Dashboard", layout="wide")
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
                    if kw in str(c):
                        return c
            return None

        bmi_col  = find_col(['BMI', '비만', '체질량'])
        run_col  = find_col(['왕복', '오래달리기', '심폐'])
        grip_col = find_col(['악력'])
        push_col = find_col(['팔굽혀', '말아올리기'])
        flex_col = find_col(['앉아윗몸', '유연성'])
        jump_col = find_col(['제자리멀리', '순발력'])

        target_cols = {
            'BMI':                    bmi_col,
            '심폐지구력 (왕복오래달리기)':   run_col,
            '근력 (악력)':               grip_col,
            '근력 (팔굽혀/말아올리기)':     push_col,
            '유연성 (앉아윗몸)':           flex_col,
            '순발력 (제자리멀리뛰기)':       jump_col,
        }
        valid_cols = {k: v for k, v in target_cols.items() if v}

        for k, col in valid_cols.items():
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True),
                errors='coerce'
            )

        # ── [수정] school_col 결정 ──────────────────────────────────────────
        school_col = '추출학교명' if '추출학교명' in df.columns else df.columns[0]

        # ── [수정] 연도 / 순수학교명 생성 보강 ─────────────────────────────
        df['연도'] = df[school_col].astype(str).str.extract(r'(20\d{2}|19\d{2})')[0]
        df['연도'] = pd.to_numeric(df['연도'], errors='coerce')

        df['순수학교명'] = (
            df[school_col].astype(str)
            .str.replace(r'\s*(20\d{2}|19\d{2})\s*', '', regex=True)
            .str.strip()
        )

        # 순수학교명이 비어있는 행 제거
        df = df[df['순수학교명'].notna() & (df['순수학교명'] != '')]

        return df, {'school_col': school_col, 'valid_cols': valid_cols}

    except Exception as e:
        st.error(f"데이터를 읽는 중 오류가 발생했습니다: {e}")
        return None, {}


# ─── 3. 학교별 집계 및 군집 명칭 자동 할당 ──────────────────────────────────
def get_clustered_df(tab_df, valid_cols, x_axis, y_axis, n_clusters):

    # ── [수정] '순수학교명' 컬럼 존재 여부 확인 ─────────────────────────────
    if '순수학교명' not in tab_df.columns:
        return pd.DataFrame()

    agg_dict = {v: 'mean' for v in valid_cols.values()}
    df = tab_df.groupby('순수학교명').agg(agg_dict).reset_index().round(1)

    # 컬럼 이름 한국어로 변경
    df.columns = ['학교명'] + list(valid_cols.keys())

    # ── [수정] x_axis / y_axis 컬럼 존재 여부 확인 ──────────────────────────
    missing = [c for c in [x_axis, y_axis] if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df.dropna(subset=[x_axis, y_axis])

    if len(df) < n_clusters:
        return df

    X = df[[x_axis, y_axis]]
    X_scaled = StandardScaler().fit_transform(X)

    # ── [수정] n_init='auto' — sklearn 1.2+ DeprecationWarning 방지 ─────────
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    sort_metric = 'BMI' if 'BMI' in df.columns else x_axis
    cluster_means = df.groupby('Cluster')[sort_metric].mean().sort_values(ascending=False)
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
        y_axis = st.selectbox("Y축 (주로 체력지표)", metrics, index=min(1, len(metrics) - 1), key=f"y_{tab_label}")
        n_clusters = st.slider("군집 세분화 (개)", 2, 4, 3, key=f"n_{tab_label}")

    plot_df = get_clustered_df(tab_df, valid_cols, x_axis, y_axis, n_clusters)

    # ── [수정] '유형' 컬럼 존재 여부까지 확인 ───────────────────────────────
    if plot_df.empty or len(plot_df) < n_clusters or '유형' not in plot_df.columns:
        st.warning("⚠️ 선택하신 지표를 동시에 측정한 학교 수가 부족하여 AI 분석을 수행할 수 없습니다. 다른 지표를 선택해 주세요.")
        return

    with col_set2:
        # ── [수정] "⚪ 기타" 색상 추가 ──────────────────────────────────────
        color_discrete_map = {
            "🔴 고위험군":   "#EF5350",
            "🟠 중점 관리군": "#FFB74D",
            "🔵 건강 우수군": "#42A5F5",
            "🟢 일반군":     "#66BB6A",
            "⚪ 기타":       "#BDBDBD",
        }
        fig = px.scatter(
            plot_df, x=x_axis, y=y_axis, color='유형', text='학교명',
            hover_name='학교명', color_discrete_map=color_discrete_map,
            title=f"🏫 강원특별자치도 {tab_label} 학교별 건강 등급 분포"
        )
        fig.update_traces(
            textposition='top center',
            marker=dict(size=18, line=dict(width=2, color='white'))
        )
        fig.update_layout(
            height=550, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.write("### 📋 유형별 특성 및 정책 제언")

    sum_df = plot_df.groupby('유형')[[x_axis, y_axis]].mean().round(1)
    # ── [수정] .get()으로 KeyError 방지 ─────────────────────────────────────
    counts = plot_df['유형'].value_counts()

    cols = st.columns(len(sum_df))
    for i, (idx, row) in enumerate(sum_df.iterrows()):
        with cols[i]:
            st.metric(label=idx, value=f"{counts.get(idx, 0)}개교")
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
        render_tab(raw_df, "전체", valid_cols)

    for i, y in enumerate(years):
        with tabs[i + 1]:
            render_tab(raw_df[raw_df['연도'] == y], f"{y}년도", valid_cols)


# 기존 코드의 fig.update_layout 부분에 폰트 사이즈 추가
fig.update_layout(
    height=400, # 모바일에서는 높이를 550 -> 400 정도로 줄이는 게 한눈에 들어옵니다.
    font=dict(size=10), # 글씨 크기 축소
    showlegend=True, 
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
