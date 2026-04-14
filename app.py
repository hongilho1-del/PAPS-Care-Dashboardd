import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 웹페이지 설정
st.set_page_config(page_title="PAPS Care+ Dashboard", layout="wide")
st.title("📊 PAPS Care+ : 강원도 학교 실데이터 AI 분석 시스템")

# 2. 합쳐진 데이터 로드 함수 (위치 추적 강화 버전)
@st.cache_data
def load_combined_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = 'PAPS_Combined_Data.xlsx'
    
    # 후보 1: data 폴더 안 / 후보 2: 현재 폴더 바로 아래
    path_options = [
        os.path.join(base_dir, 'data', file_name),
        os.path.join(base_dir, file_name)
    ]
    
    target_path = None
    for p in path_options:
        if os.path.exists(p):
            target_path = p
            break
            
    if target_path is None:
        return None

    try:
        # engine='openpyxl'을 명시하여 진짜 엑셀을 읽습니다.
        df = pd.read_excel(target_path, engine='openpyxl')
        
        # 지표 자동 찾기 로직
        def find_col(keywords):
            for c in df.columns:
                for kw in keywords:
                    if kw in str(c): return c
            return None

        bmi_col = find_col(['BMI', '비만', '체질량'])
        run_col = find_col(['왕복', '오래달리기', '심폐'])
        strength_col = find_col(['악력', '근력'])
        flex_col = find_col(['앉아윗몸', '유연성'])
        jump_col = find_col(['제자리멀리', '순발력'])

        target_cols = {'BMI': bmi_col, '심폐지구력': run_col, '근력': strength_col, '유연성': flex_col, '순발력': jump_col}
        valid_cols = {k: v for k, v in target_cols.items() if v}
        
        for k, col in valid_cols.items():
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

        school_col = '추출학교명' if '추출학교명' in df.columns else df.columns[0]
        school_avg = df.groupby(school_col).agg({v: 'mean' for v in valid_cols.values()}).dropna().reset_index()
        school_avg = school_avg.round(1)
        school_avg.columns = ['학교명'] + list(valid_cols.keys())
        
        return school_avg
    except Exception as e:
        st.error(f"데이터를 읽는 중 오류가 발생했습니다: {e}")
        return None

df = load_combined_data()

# 3. 화면 UI 렌더링
if df is None or df.empty:
    st.error("❌ 'PAPS_Combined_Data.xlsx' 파일을 찾을 수 없습니다.")
    st.info("💡 해결 방법: 깃허브에 합친 엑셀 파일을 업로드했는지 확인해주세요.")
else:
    st.success(f"✅ 총 {len(df)}개 학교 데이터를 성공적으로 분석 중입니다!")
    
    metrics = [c for c in df.columns if c != '학교명']
    col1, col2 = st.columns([1, 1])
    with col1: x_axis = st.selectbox("X축 지표", metrics, index=0)
    with col2: y_axis = st.selectbox("Y축 지표", metrics, index=min(1, len(metrics)-1))
    
    X = df[[x_axis, y_axis]]
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['유형'] = df['Cluster'].map({0: '유형 1 (파랑)', 1: '유형 2 (주황)', 2: '유형 3 (초록)'})

    fig = px.scatter(df, x=x_axis, y=y_axis, color='유형', 
                     hover_name='학교명', text='학교명',
                     color_discrete_map={'유형 1 (파랑)': 'blue', '유형 2 (주황)': 'orange', '유형 3 (초록)': 'green'})
    fig.update_traces(textposition='top center', marker=dict(size=15, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(height=600, title=f"🏫 강원도 중학교 {x_axis} vs {y_axis} AI 분석")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 상세 리포트 확인"):
        st.dataframe(df.drop(columns=['Cluster']).style.highlight_max(axis=0, color='lightgreen'))
