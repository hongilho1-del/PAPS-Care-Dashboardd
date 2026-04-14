import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 웹페이지 설정
st.set_page_config(page_title="PAPS Care+ Dashboard", layout="wide")
st.title("📊 PAPS Care+ : 강원도 학교 실데이터 AI 분석 시스템")

# 2. 합쳐진 데이터 로드 함수
@st.cache_data
def load_combined_data():
    # app.py와 같은 위치의 data 폴더 내 파일을 찾습니다.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', 'PAPS_Combined_Data.xlsx')
    
    if not os.path.exists(file_path):
        return None

    try:
        # 코랩에서 만든 진짜 엑셀 파일이므로 바로 읽습니다.
        df = pd.read_excel(file_path)
        
        # 지표 자동 찾기
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
        
        # 숫자 정제 (코랩에서 이미 되었겠지만 한 번 더 안전하게)
        for k, col in valid_cols.items():
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

        # 학교별 평균 집계 (추출학교명 기준)
        school_col = '추출학교명' if '추출학교명' in df.columns else df.columns[0]
        school_avg = df.groupby(school_col).agg({v: 'mean' for v in valid_cols.values()}).dropna().reset_index()
        school_avg = school_avg.round(1)
        school_avg.columns = ['학교명'] + list(valid_cols.keys())
        
        return school_avg
    except Exception as e:
        st.error(f"데이터를 읽는 중 오류가 발생했습니다: {e}")
        return None

# 데이터 실행
df = load_combined_data()

# 3. 화면 구성
if df is None or df.empty:
    st.error("❌ 'data' 폴더 안에 'PAPS_Combined_Data.xlsx' 파일이 없습니다.")
    st.info("💡 해결 방법: 깃허브의 data 폴더 안에 코랩에서 합친 파일을 업로드해주세요.")
else:
    st.success(f"✅ 총 {len(df)}개 학교 데이터를 성공적으로 분석 중입니다.")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 분석 지표 설정")
    metrics = [c for c in df.columns if c != '학교명']
    x_axis = st.sidebar.selectbox("X축 지표", metrics, index=0)
    y_axis = st.sidebar.selectbox("Y축 지표", metrics, index=min(1, len(metrics)-1))
    
    # AI 군집화
    X = df[[x_axis, y_axis]]
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['유형'] = df['Cluster'].map({0: '유형 1 (파랑)', 1: '유형 2 (주황)', 2: '유형 3 (초록)'})

    # 시각화
    fig = px.scatter(df, x=x_axis, y=y_axis, color='유형', 
                     hover_name='학교명', text='학교명',
                     color_discrete_map={'유형 1 (파랑)': 'blue', '유형 2 (주황)': 'orange', '유형 3 (초록)': 'green'})
    fig.update_traces(textposition='top center', marker=dict(size=15, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(height=650, title=f"🏫 강원도 중학교 {x_axis} vs {y_axis} AI 분석")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 전체 리포트")
    st.dataframe(df.drop(columns=['Cluster']))
