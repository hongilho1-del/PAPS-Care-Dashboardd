import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

# ────────────────────────────────────────────────────────────────
# 페이지 기본 설정
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PAPS CARE+",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────
# 전체 CSS (흰색 배경 + 사이드바 스타일)
# ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

/* ── 전체 배경 & 기본 폰트 ── */
html, body, [class*="css"], .stApp {
    font-family: 'Noto Sans KR', sans-serif;
    background-color: #f5f6fa !important;
    color: #1a2233;
}

/* ── 사이드바 ── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e8eaf0;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0;
}

/* ── 사이드바 상단 로고 영역 ── */
.sidebar-logo {
    background: #1a2233;
    color: #ffffff;
    padding: 18px 20px;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 8px;
}

/* ── 사이드바 섹션 헤더 ── */
.sidebar-section {
    font-size: 10px;
    font-weight: 600;
    color: #9ca3b0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 14px 20px 4px 20px;
}

/* ── 사이드바 메뉴 아이템 ── */
.sidebar-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 20px;
    font-size: 13px;
    color: #4a5568;
    cursor: pointer;
    border-radius: 0;
    transition: background 0.15s;
    text-decoration: none;
}
.sidebar-item:hover {
    background: #f0f2f7;
    color: #1a2233;
}
.sidebar-item.active {
    background: #eef2ff;
    color: #3b5bdb;
    font-weight: 600;
    border-right: 3px solid #3b5bdb;
}
.sidebar-item .icon {
    font-size: 15px;
    width: 18px;
    text-align: center;
}

/* ── 메인 영역 패딩 ── */
.main-content {
    padding: 0 8px;
}

/* ── 브레드크럼 ── */
.breadcrumb {
    font-size: 12px;
    color: #9ca3b0;
    margin-bottom: 6px;
}
.breadcrumb span {
    color: #4a5568;
    font-weight: 500;
}

/* ── 페이지 타이틀 ── */
.page-title {
    font-size: 22px;
    font-weight: 700;
    color: #1a2233;
    margin-bottom: 20px;
    letter-spacing: -0.01em;
}

/* ── 탭 스타일 오버라이드 ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid #e8eaf0;
    gap: 0;
}
[data-testid="stTabs"] [role="tab"] {
    font-size: 13px;
    font-weight: 500;
    color: #6b7280;
    padding: 10px 18px;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    background: none;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #1a2233;
    font-weight: 700;
    border-bottom: 2px solid #1a2233;
}

/* ── 필터 바 ── */
.filter-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 16px 0 12px 0;
    flex-wrap: wrap;
}
.filter-label {
    font-size: 12px;
    color: #6b7280;
    font-weight: 500;
    white-space: nowrap;
}
.filter-btn {
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    border: 1.5px solid #d1d5db;
    background: white;
    color: #374151;
}
.filter-btn.active {
    background: #1a2233;
    color: white;
    border-color: #1a2233;
}

/* ── 지도 카드 ── */
.map-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 4px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    overflow: hidden;
}

/* ── 상단 헤더 바 ── */
.top-header {
    background: #ffffff;
    border-bottom: 1px solid #e8eaf0;
    padding: 12px 24px;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 20px;
    border-radius: 10px;
}
.top-header .breadcrumb-nav {
    font-size: 13px;
    color: #9ca3b0;
}
.top-header .breadcrumb-nav b {
    color: #374151;
}

/* ── selectbox 스타일 ── */
[data-testid="stSelectbox"] > div {
    border-radius: 8px;
    border: 1.5px solid #e2e5ed !important;
    font-size: 13px;
}

/* ── Streamlit 기본 여백 줄이기 ── */
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}

/* ── 구분선 ── */
.divider {
    border: none;
    border-top: 1px solid #f0f2f5;
    margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# 사이드바
# ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🏃 체육행정 시스템</div>', unsafe_allow_html=True)

    # 대시보드 섹션
    st.markdown('<div class="sidebar-section">대시보드</div>', unsafe_allow_html=True)
    st.markdown('''
        <div class="sidebar-item active">
            <span class="icon">📊</span> 체육 행정
        </div>
        <div class="sidebar-item">
            <span class="icon">📋</span> 체력 랭킹
        </div>
        <div class="sidebar-item">
            <span class="icon">🏫</span> 체육 행정
        </div>
        <div class="sidebar-item">
            <span class="icon">📁</span> 체력 체육
        </div>
        <div class="sidebar-item">
            <span class="icon">👨‍🏫</span> 참여 교사
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # 분석 섹션
    st.markdown('<div class="sidebar-section">분석</div>', unsafe_allow_html=True)
    st.markdown('''
        <div class="sidebar-item">
            <span class="icon">🏠</span> 세움 부형원
        </div>
        <div class="sidebar-item">
            <span class="icon">🏠</span> 원단 광형
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # 불류 섹션
    st.markdown('<div class="sidebar-section">불류</div>', unsafe_allow_html=True)
    st.markdown('''
        <div class="sidebar-item">
            <span class="icon">📄</span> 자담 정보
        </div>
        <div class="sidebar-item">
            <span class="icon">📝</span> 자담 체육
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # 시지 섹션
    st.markdown('<div class="sidebar-section">시지</div>', unsafe_allow_html=True)
    st.markdown('''
        <div class="sidebar-item">
            <span class="icon">⚙️</span> 산면 지역
        </div>
    ''', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# 샘플 데이터 (실제 데이터로 교체하세요)
# ────────────────────────────────────────────────────────────────
kangwon_cities = [
    "춘천", "원주", "강릉", "동해", "태백", "속초", "삼척",
    "홍천", "횡성", "영월", "평창", "정선", "철원", "화천",
    "양구", "인제", "고성", "양양",
]
유형_list = ["고위험군", "관리 필요군", "중점관리군", "일반군", "건강 양호군", "우수군"]

np.random.seed(42)
n = 200
sample_data = pd.DataFrame({
    "시군": np.random.choice(kangwon_cities, n),
    "유형": np.random.choice(유형_list, n, p=[0.1, 0.15, 0.2, 0.25, 0.2, 0.1]),
    "순수학교명": [f"학교{i}" for i in range(n)],
    "연도": np.random.choice([2022, 2023, 2024], n),
    "학년": np.random.choice([1, 2, 3], n),
    "성별": np.random.choice(["남", "여"], n),
    "체력점수": np.random.normal(65, 15, n).clip(20, 100),
    "근력": np.random.normal(60, 12, n).clip(20, 100),
})

# ── 실제 프로젝트에서는 위 sample_data 대신 cluster_source 사용 ──
cluster_source = sample_data


# ────────────────────────────────────────────────────────────────
# 메인 컨텐츠
# ────────────────────────────────────────────────────────────────

# 브레드크럼 + 타이틀
st.markdown('''
<div class="breadcrumb">교육행정 &gt; <span>광력행 가름</span></div>
<div class="page-title">체육행정</div>
''', unsafe_allow_html=True)

dominant_group = cluster_source["유형"].value_counts().idxmax()
dominant_share = round(
    (cluster_source["유형"].value_counts().max() / len(cluster_source)) * 100, 1
)

# ── 탭 ──
sub_tabs = st.tabs(["종합 현황", "취약 체력 지역 히트맵", "맞춤형 처방"])

# ────────────────────────────────────────────────────────────────
# 탭 0 : 종합 현황
# ────────────────────────────────────────────────────────────────
with sub_tabs[0]:
    st.markdown("#### 종합 현황")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("전체 학교 수", f"{len(cluster_source['순수학교명'].unique())}개교")
    with col2:
        st.metric("주요 집단", dominant_group)
    with col3:
        st.metric("주요 집단 비율", f"{dominant_share}%")
    with col4:
        high_risk = len(cluster_source[cluster_source["유형"].isin(["고위험군", "관리 필요군"])])
        st.metric("관리 필요 학교", f"{high_risk}개교", delta="주의")

    st.markdown("---")

    유형_counts = cluster_source["유형"].value_counts().reset_index()
    유형_counts.columns = ["유형", "수"]
    color_map = {
        "고위험군": "#d44b57",
        "관리 필요군": "#e8734a",
        "중점관리군": "#ef8b2c",
        "일반군": "#1c9d74",
        "건강 양호군": "#2574ea",
        "우수군": "#1a56db",
    }
    fig_bar = px.bar(
        유형_counts,
        x="유형", y="수",
        color="유형",
        color_discrete_map=color_map,
        title="집단별 학교 분포",
    )
    fig_bar.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(t=40, b=20, l=20, r=20),
        font=dict(family="Noto Sans KR"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# 탭 1 : 취약 체력 지역 히트맵
# ────────────────────────────────────────────────────────────────
with sub_tabs[1]:

    # ── 필터 바 ──
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        selected_scope = st.selectbox(
            "교재 칠간",
            ["자부 성장", "전체", "고위험군만", "관리 필요군 이상"],
            label_visibility="collapsed",
        )
    with col_f2:
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.markdown('<div class="filter-btn active" style="text-align:center">권역별</div>', unsafe_allow_html=True)
        with btn_col2:
            st.markdown('<div class="filter-btn" style="text-align:center">이여닝닷</div>', unsafe_allow_html=True)
    with col_f3:
        st.markdown('<div style="display:flex;gap:6px;justify-content:flex-end">'
                    '<div class="filter-btn">관리화 날림</div>'
                    '<div class="filter-btn" style="padding:6px 10px">⊞</div>'
                    '<div class="filter-btn" style="padding:6px 10px">≡</div>'
                    '</div>', unsafe_allow_html=True)

    # ── 데이터 준비 ──
    kangwon_coords = {
        "춘천": (37.8813, 127.7298),
        "원주": (37.3422, 127.9202),
        "강릉": (37.7518, 128.8760),
        "동해": (37.5245, 129.1143),
        "태백": (37.1640, 128.9856),
        "속초": (38.2070, 128.5918),
        "삼척": (37.4498, 129.1653),
        "홍천": (37.6970, 127.8887),
        "횡성": (37.4913, 127.9850),
        "영월": (37.1833, 128.4619),
        "평창": (37.3705, 128.3902),
        "정선": (37.3797, 128.6608),
        "철원": (38.1466, 127.3134),
        "화천": (38.1060, 127.7082),
        "양구": (38.1065, 127.9897),
        "인제": (38.0694, 128.1707),
        "고성": (38.3806, 128.4678),
        "양양": (38.0754, 128.6190),
    }

    map_df = cluster_source.copy()

    def get_coords(city_name):
        clean_name = str(city_name).replace("시", "").replace("군", "").strip()
        return kangwon_coords.get(clean_name, (37.8813, 127.7298))

    map_df["lat"] = map_df["시군"].apply(lambda v: get_coords(v)[0])
    map_df["lon"] = map_df["시군"].apply(lambda v: get_coords(v)[1])

    weight_map = {
        "고위험군": 10,
        "관리 필요군": 8,
        "중점관리군": 5,
        "일반군": 1,
        "건강 양호군": 0.5,
        "우수군": 0.1,
    }
    map_df["weight"] = map_df["유형"].map(weight_map).fillna(1)

    # ── 히트맵 ──
    st.markdown('<div class="map-card">', unsafe_allow_html=True)

    fig = px.density_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        z="weight",
        radius=60,
        center=dict(lat=37.75, lon=128.3),
        zoom=6.8,
        mapbox_style="carto-darkmatter",
        hover_name="시군",
        hover_data={
            "순수학교명": True,
            "유형": True,
            "lat": False,
            "lon": False,
            "weight": False,
        },
        color_continuous_scale=[
            [0.0,  "rgba(10, 15, 60, 0.0)"],
            [0.2,  "rgba(20, 30, 120, 0.6)"],
            [0.45, "rgba(40, 60, 180, 0.8)"],
            [0.65, "rgba(200, 80, 30, 0.9)"],
            [0.85, "rgba(240, 120, 30, 0.95)"],
            [1.0,  "rgba(255, 160, 50, 1.0)"],
        ],
    )

    fig.update_traces(opacity=0.85)

    # 우측 상단 범례 카드
    fig.add_annotation(
        x=0.98, y=0.96,
        xref="paper", yref="paper",
        xanchor="right", yanchor="top",
        text=(
            "<b>취약 체력 증목</b><br>"
            "<span style='color:#f07030'>━━━━</span>"
            "<span style='color:#3050b0'>━━━━</span>"
        ),
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.93)",
        bordercolor="rgba(210,215,225,0.8)",
        borderwidth=1,
        borderpad=12,
        font=dict(size=12, color="#1a2233", family="Noto Sans KR"),
    )

    fig.update_layout(
        height=580,
        margin=dict(t=0, b=0, l=0, r=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# 탭 2 : 맞춤형 처방
# ────────────────────────────────────────────────────────────────
with sub_tabs[2]:
    st.markdown("#### 맞춤형 처방")

    st.info("선택한 집단의 체력 수준에 맞는 운동 프로그램을 확인할 수 있습니다.")

    selected_type = st.selectbox("집단 선택", 유형_list)
    filtered = cluster_source[cluster_source["유형"] == selected_type]

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("해당 집단 학교 수", f"{len(filtered)}개교")
    with col_b:
        st.metric("평균 체력점수", f"{filtered['체력점수'].mean():.1f}점")

    prescription_map = {
        "고위험군":   "⚠️ 즉각적인 의료 상담 및 맞춤 재활 운동 프로그램 권장",
        "관리 필요군": "🔶 주 3회 이상 유산소 + 근력 복합 운동 권장",
        "중점관리군": "🟡 주 2~3회 체력 향상 운동 프로그램 적용",
        "일반군":    "🟢 기본 체육 수업 유지 및 자율 활동 장려",
        "건강 양호군": "🔵 현재 수준 유지 및 스포츠 참여 확대 권장",
        "우수군":    "🏆 심화 스포츠 프로그램 및 대회 참가 권장",
    }
    st.markdown(f"""
    <div style='background:#f8faff;border-left:4px solid #3b5bdb;
                padding:16px 20px;border-radius:8px;margin-top:16px;
                font-size:14px;color:#1a2233;line-height:1.8'>
        <b>{selected_type} 처방 가이드</b><br>
        {prescription_map.get(selected_type, "")}
    </div>
    """, unsafe_allow_html=True)
