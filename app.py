import streamlit as st
import pandas as pd
import glob
import os
import io
from bs4 import BeautifulSoup
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PAPS Care+ Dashboard", layout="wide")
st.title("📊 PAPS Care+ : 강원도 학교 실데이터 AI 분석 시스템")

@st.cache_data
def load_ms_html_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, 'data')
    
    if not os.path.exists(folder_path):
        return None, "❌ 'data' 폴더가 없습니다."

    excel_files = glob.glob(os.path.join(folder_path, '*.xls'))
    if not excel_files:
        return None, "❌ 'data' 폴더 안에 파일이 없습니다."

    df_list = []
    error_logs = []
    
    for file in excel_files:
        school_name = os.path.basename(file).split('.')[0]
        df_target = None
        
        try:
            # 1. 파일 열기 (박사님이 찾아주신 단서대로 utf-8 고정)
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()

            # 2. 수술 도구(BeautifulSoup) 투입: MS 오피스 찌꺼기 무시하고 표만 찾기
            soup = BeautifulSoup(html_content, 'html.parser')
            tables = soup.find_all('table')
            
            if tables:
                # 3. 여러 표 중에서 <tr>(줄)이 가장 많은 진짜 데이터 표 도려내기
                main_table = max(tables, key=lambda t: len(t.find_all('tr')))
                
                # 4. 도려낸 순수 표를 판다스로 읽기 (100% 성공 보장)
                df_target = pd.read_html(io.StringIO(str(main_table)))[0]
                
        except Exception as e:
            pass # 실패하면 아래에서 에러 로그로 처리
        
        # 데이터 정제 및 저장
        if df_target is not None and not df_target.empty:
            # 다중 헤더(2줄짜리 이름) 1줄로 압축
            if isinstance(df_target.columns, pd.MultiIndex):
                df_target.columns = ['_'.join(str(c) for c in col if pd.notna(c) and 'Unnamed' not in str(c)).strip() for col in df_target.columns]
            
            df_target['추출학교명'] = school_name
            df_list.append(df_target)
        else:
            error_logs.append(school_name)

    if not df_list:
        return None, f"❌ 파일을 뜯었으나 표를 추출하지 못했습니다. (실패: {', '.join(error_logs[:5])})"
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    # ==========================================
    # 스마트 컬럼 추적 및 정제
    # ==========================================
    def find_col(keywords):
        for c in full_df.columns:
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
    
    if len(valid_cols) < 2:
        return None, f"❌ 표는 찾았으나 체력 지표를 찾지 못했습니다. (현재 발견된 데이터 기둥: {list(full_df.columns)[:15]})"

    # 글자("회", "kg" 등) 지우고 순수 숫자만 남기기
    for k, col in valid_cols.items():
        full_df[col] = pd.to_numeric(full_df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

    school_avg = full_df.groupby('추출학교명').agg({v: 'mean' for v in valid_cols.values()}).dropna().reset_index()
    school_avg = school_avg.round(1)
    school_avg.columns = ['학교명'] + list(valid_cols.keys())
    
    return school_avg, f"✅ 총 {len(school_avg)}개 학교 데이터 수술 및 로드 성공!"

# ==========================================
# 화면 UI 렌더링
# ==========================================
df, msg = load_ms_html_data()

if df is None:
    st.error(msg)
else:
    st.success(msg)
    st.sidebar.header("⚙️ 분석 지표 설정")
    available_metrics = [c for c in df.columns if c != '학교명']
    x_axis = st.sidebar.selectbox("X축 지표", available_metrics, index=0)
    y_axis = st.sidebar.selectbox("Y축 지표", available_metrics, index=min(1, len(available_metrics)-1))
    
    X = df[[x_axis, y_axis]]
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=min(3, len(df)), random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['유형'] = df['Cluster'].map({0: '유형 1 (파랑)', 1: '유형 2 (주황)', 2: '유형 3 (초록)'})

    fig = px.scatter(df, x=x_axis, y=y_axis, color='유형', 
                     hover_name='학교명', text='학교명',
                     color_discrete_map={'유형 1 (파랑)': 'blue', '유형 2 (주황)': 'orange', '유형 3 (초록)': 'green'})
    fig.update_traces(textposition='top center', marker=dict(size=15, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(height=650, title=f"🏫 강원도 중학교 {x_axis} vs {y_axis} 실시간 AI 분석 대시보드")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 전체 체력 지표 리포트")
    st.dataframe(df.drop(columns=['Cluster']).style.highlight_max(axis=0, color='lightgreen'))