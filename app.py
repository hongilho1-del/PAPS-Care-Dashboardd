import streamlit as st
import pandas as pd
import os
import re
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─── 1. 웹페이지 설정 ───────────────────────────────────────────────────────
st.set_page_config(page_title="PAPS Care+ Dashboard", layout="wide")
st.title("📊 PAPS Care+ : 강원도 학교 실데이터 AI 분석 시스템")


# ─── 2. 데이터 로드 ──────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', 'PAPS_Combined_Data.xlsx')

    if not os.path.exists(file_path):
        return None, {}

    try:
        df = pd.read_excel(file_path)

        # ── 컬럼 자동 탐지 ──────────────────────────────────────────────────
        def find_col(keywords):
            for c in df.columns:
                for kw in keywords:
                    if kw in str(c):
                        return c
            return None

        bmi_col      = find_col(['BMI', '비만', '체질량'])
        run_col      = find_col(['왕복', '오래달리기', '심폐'])
        strength_col = find_col(['악력', '근력'])
        flex_col     = find_col(['앉아윗몸', '유연성'])
        jump_col     = find_col(['제자리멀리', '순발력'])

        target_cols = {
            'BMI':      bmi_col,
            '심폐지구력': run_col,
            '근력':      strength_col,
            '유연성':    flex_col,
            '순발력':    jump_col,
        }
        valid_cols = {k: v for k, v in target_cols.items() if v}

        # ── 숫자 정제 ────────────────────────────────────────────────────────
        for k, col in valid_cols.items():
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True),
                errors='coerce'
            )

        # ── 학교 컬럼 탐지 ───────────────────────────────────────────────────
        school_col = '추출학교명' if '추출학교명' in df.columns else df.columns[0]

        # ──────────────────────────────────────────────────────────────────────
        # 핵심: 학교명에서 연도 추출
        # 예시) "강릉중2023" -> 순수학교명: "강릉중", 연도: 2023
        # 예시) "솔올중2025" -> 순수학교명: "솔올중", 연도: 2025
        # ──────────────────────────────────────────────────────────────────────
        
        # 1) 연도 추출 (4자리 숫자)
        df['연도'] = df[school_col].astype(str).str.extract(r'(20\d{2}|19\d{2})')[0]
        df['연도'] = pd.to_numeric(df['연도'], errors='coerce')

        # 2) 순수 학교명 (연도 제거)
        df['순수학교명'] = (
            df[school_col].astype(str)
            .str.replace(r'\s*(20\d{2}|19\d{2})\s*', '', regex=True)  # 연도 제거
            .str.strip()  # 양쪽 공백 제거
        )

        return df, {
            'school_col':   school_col,
            'valid_cols':   valid_cols,
        }

    except Exception as e:
        st.error(f"데이터를 읽는 중 오류가 발생했습니다: {e}")
        return None, {}


# ─── 3. 학교별 집계 함수 ──────────────────────────────────────────────────────
def aggregate_schools(tab_df, valid_cols):
    """순수학교명 기준으로 집계"""
    agg_dict = {v: 'mean' for v in valid_cols.values()}
    school_avg = (
        tab_df.groupby('순수학교명')
        .agg(agg_dict)
        .dropna(how='all')   # 모든 지표가 NaN인 학교만 제거
        .reset_index()
        .round(1)
    )
    school_avg.columns = ['학교명'] + list(valid_cols.keys())
    return school_avg


# ─── 4. 탭별 렌더링 함수 ──────────────────────────────────────────────────────
def render_tab(tab_df, tab_label, valid_cols):
    df = aggregate_schools(tab_df, valid_cols)

    if df.empty:
        st.warning(f"{tab_label} 데이터가 없습니다.")
        return

    # 📊 학교 수 표시
    unique_schools = tab_df['순수학교명'].nunique()
    total_records = len(tab_df)
    st.success(f"✅ {tab_label} - 총 **{unique_schools}개** 학교, **{total_records}개** 레코드 분석 중")

    metrics = [c for c in df.columns if c != '학교명']

    # ── 컬럼 레이아웃 ──────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("🔧 분석 설정")
        x_axis = st.selectbox("X축 지표", metrics, index=0, key=f"x_{tab_label}")
        y_axis = st.selectbox(
            "Y축 지표", metrics,
            index=min(1, len(metrics) - 1),
            key=f"y_{tab_label}"
        )
        n_clusters = st.slider(
            "AI 군집 수", min_value=2, max_value=5, value=3,
            key=f"k_{tab_label}"
        )
        
        # 📋 디버그 정보 표시
        st.markdown("---")
        st.markdown("**학교명 파싱 샘플:**")
        sample = tab_df[['순수학교명', '연도']].drop_duplicates().head(8)
        st.dataframe(sample, use_container_width=True, height=200)

    with col2:
        # ── 선택한 두 지표 기준으로만 NaN 제거 ───────────────────────────
        plot_df = df.dropna(subset=[x_axis, y_axis]).copy()

        if len(plot_df) < n_clusters:
            st.warning(
                f"⚠️ 유효한 학교 수({len(plot_df)}개)가 군집 수({n_clusters})보다 적습니다."
            )
            return

        # ── AI 군집화 ──────────────────────────────────────────────────
        X_scaled = StandardScaler().fit_transform(plot_df[[x_axis, y_axis]])
        kmeans   = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        plot_df['Cluster'] = kmeans.fit_predict(X_scaled)

        cluster_names = {i: f"유형 {i+1}" for i in range(n_clusters)}
        cluster_colors = {
            0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1', 
            3: '#96CEB4', 4: '#FFEAA7'
        }
        plot_df['유형'] = plot_df['Cluster'].map(cluster_names)
        color_map = {v: cluster_colors[k] for k, v in cluster_names.items()}

        # ── 산점도 ───────────────────────────────────────────────────────
        fig = px.scatter(
            plot_df,
            x=x_axis, y=y_axis,
            color='유형',
            hover_name='학교명',
            hover_data={c: ':.2f' for c in metrics if c not in [x_axis, y_axis]},
            text='학교명',
            color_discrete_map=color_map,
            title=f"🏫 강원도 중학교 {x_axis} vs {y_axis} AI 군집 분석 ({tab_label})",
        )
        fig.update_traces(
            textposition='top center',
            textfont_size=10,
            marker=dict(size=16, opacity=0.8, line=dict(width=2, color='white'))
        )
        fig.update_layout(
            height=600,
            font=dict(size=11),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── 결과 테이블들 ──────────────────────────────────────────────────────
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📈 군집별 평균 요약")
        summary = plot_df.groupby('유형')[metrics].mean().round(2).reset_index()
        st.dataframe(summary, use_container_width=True)
    
    with col4:
        st.subheader("📋 각 군집 학교 수")
        count_summary = plot_df['유형'].value_counts().reset_index()
        count_summary.columns = ['유형', '학교 수']
        st.dataframe(count_summary, use_container_width=True)

    # ── 전체 리포트 ──────────────────────────────────────────────────────
    st.subheader("📊 전체 분석 결과")
    display_df = plot_df.drop(columns=['Cluster']).sort_values(['유형', '학교명'])
    st.dataframe(display_df, use_container_width=True, height=350)

    # ── CSV 다운로드 ──────────────────────────────────────────────────────
    csv = display_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label=f"💾 {tab_label} 결과 CSV 다운로드",
        data=csv,
        file_name=f"PAPS_분석결과_{tab_label}.csv",
        mime='text/csv',
        key=f"dl_{tab_label}",
    )


# ─── 5. 메인 실행 ─────────────────────────────────────────────────────────────
raw_df, meta = load_raw_data()

if raw_df is None or raw_df.empty:
    st.error("❌ 'data' 폴더 안에 'PAPS_Combined_Data.xlsx' 파일이 없습니다.")
    st.info("💡 깃허브의 data 폴더 안에 코랩에서 합친 파일을 업로드해주세요.")
    st.stop()

valid_cols = meta['valid_cols']

# ── 연도 탭 구성 ───────────────────────────────────────────────────────────────
if '연도' in raw_df.columns and raw_df['연도'].notna().any():
    years = sorted(raw_df['연도'].dropna().unique().astype(int).tolist())
    st.info(f"🎯 발견된 연도: **{', '.join(map(str, years))}**")
    
    tab_labels = ["🌐 전체"] + [f"📅 {y}년" for y in years]
    tabs = st.tabs(tab_labels)

    # 탭 데이터 매핑
    tab_data_map = {}
    tab_data_map["🌐 전체"] = raw_df
    for y in years:
        tab_data_map[f"📅 {y}년"] = raw_df[raw_df['연도'] == y]

    # 각 탭 렌더링
    for tab_widget, label in zip(tabs, tab_labels):
        with tab_widget:
            render_tab(tab_data_map[label], label.split(' ')[1], valid_cols)  # 이모지 제거
else:
    st.warning("⚠️ 학교명에서 연도(4자리)를 찾지 못했습니다.")
    st.info("💡 학교명 형식 예시: '강릉중2023', '솔올중2025'")
    tabs = st.tabs(["🌐 전체"])
    with tabs[0]:
        render_tab(raw_df, "전체", valid_cols)
