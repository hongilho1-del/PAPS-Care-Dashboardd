# PAPS Care+ Intelligence

학교 체력 데이터를 기반으로 군집 분석과 맞춤형 처방 방향을 시각화하는 Streamlit 대시보드입니다.

## 주요 기능

- 상단 컨트롤 센터에서 연도, 지역, 학년, 성별, 학교 필터 적용
- 선택한 두 지표 기준의 AI 군집 분석
- 기관 보고용 스타일의 KPI 요약 화면
- 학교 분포를 보여주는 클러스터 맵
- 집단별 운동 처방 및 교육 프로그램 카드 리포트

## 프로젝트 구조

```text
.
├─ app.py
├─ requirements.txt
├─ README.md
└─ data/
   └─ PAPS_Combined_Data.xlsx
```

## 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 파일 준비

아래 위치에 엑셀 파일을 넣어야 합니다.

```text
data/PAPS_Combined_Data.xlsx
```

### 3. 앱 실행

```bash
streamlit run app.py
```

## 사용 기술

- Streamlit
- Pandas
- Plotly
- Scikit-learn
- OpenPyXL

## 배포 참고

- Streamlit Community Cloud 또는 기타 Python 지원 환경에서 배포할 수 있습니다.
- 배포 시 `requirements.txt`와 `app.py`가 저장소 루트에 있어야 합니다.
- 데이터 파일이 저장소에 없으면 앱에서 파일을 찾을 수 없다는 안내가 표시됩니다.

## 주의 사항

- 군집 결과는 선택된 필터 조건을 기준으로 다시 계산됩니다.
- BMI는 높은 값이 항상 우수하다고 해석되지 않도록 별도 방향성을 적용했습니다.
- 엑셀 컬럼명이 크게 다르면 지표 인식이 달라질 수 있습니다.
