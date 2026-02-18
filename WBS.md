# 센서 데이터 기반 기계 불량 예측

## Phase 1 프로젝트 WBS (4주)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 목표

센서 데이터를 기반으로 기계 고장을 사전에 예측하는 ML/DL 모델을 개발하고,
MLOps 파이프라인을 구축하여 지속 가능한 운영 체계를 확립합니다.

---

### 1.2 Phase 1 범위

* 센서 데이터 기반 불량 예측 모델 개발 (Vision 모델 제외)
* 데이터 파이프라인 및 DB 구축
* MLOps 기본 인프라 구축 (실험 관리, 모델 버전 관리, 배포)
* 모니터링 대시보드 구축

---

### 1.3 데이터셋

#### Main Dataset

Machine Failure Prediction using Sensor Data

* 온도, 회전 속도, 토크, 공구 마모 등 센서 데이터
* 다중 고장 유형 분류

  * Heat Dissipation
  * Power
  * Overstrain
  * Tool Wear
  * Random

#### Sub Dataset

NASA Bearing Dataset

* 베어링 진동 센서 시계열 데이터
* 추가 Feature Engineering 및 앙상블 모델 검증용

---

## 2. 팀 구성 및 역할

| 역할       | 인원 | 주요 업무                                                                     |
| -------- | -- | ------------------------------------------------------------------------- |
| 모델링 팀    | 2명 | 데이터 전처리 및 EDA<br>Feature Engineering<br>ML/DL 모델 개발 및 튜닝<br>모델 성능 평가 및 분석 |
| DB 담당    | 1명 | DB 스키마 설계<br>데이터 파이프라인 구축<br>데이터 품질 관리                                    |
| MLOps 담당 | 1명 | 실험 관리 시스템 구축<br>모델 배포 파이프라인<br>모니터링 대시보드<br>API 서버 구축                     |

---

## 3. 주차별 WBS (Work Breakdown Structure)

---

# Week 1: 프로젝트 셋업 & 데이터 이해

## 전체

* 프로젝트 킥오프 미팅
* 역할 및 책임 정의
* Git Repository 구조 설계
* 개발 환경 구축 (Python, 라이브러리)

**산출물**

* 프로젝트 계획서
* Repository

---

## 모델링 팀

* 데이터셋 다운로드 및 이해
* EDA (Exploratory Data Analysis)

  * 데이터 분포 분석
  * 결측치, 이상치 분석
  * 클래스 불균형 확인
  * 변수 간 상관관계 분석
* Baseline 모델 구조 설계

**산출물**

* EDA 보고서
* 데이터 분석 노트북

---

## DB 담당

* 데이터 요구사항 정의
* DB 스키마 설계 (ERD 작성)

  * Raw Data Layer
  * Processed Data Layer
  * Model Metadata Layer
* PostgreSQL 설치 및 설정

**산출물**

* DB 스키마 문서
* ERD 다이어그램

---

## MLOps 담당

* MLOps 아키텍처 설계
* MLflow 설치 및 설정
* 실험 추적 프레임워크 구축
* Docker 환경 구성

**산출물**

* MLOps 아키텍처 문서
* MLflow 설정

---

# Week 2: 데이터 파이프라인 & 모델 개발 시작

## 모델링 팀

* Feature Engineering

  * 통계적 특성 추출 (평균, 분산, 기울기)
  * 시간 윈도우 기반 집계
  * NASA Bearing 데이터 통합 전략 수립
* Baseline ML 모델 구현

  * Logistic Regression
  * Random Forest
  * XGBoost
* 클래스 불균형 처리 (SMOTE, Class Weight)

**산출물**

* Feature 정의서
* Baseline 모델 코드
* 성능 비교표

---

## DB 담당

* DB 테이블 생성 (DDL 작성)
* 데이터 적재 파이프라인 구현

  * Kaggle → PostgreSQL ETL
  * 데이터 검증 로직 추가
* 데이터 버전 관리 체계 구축
* 초기 데이터 적재 및 검증

**산출물**

* DDL 스크립트
* ETL 파이프라인 코드
* 데이터 품질 리포트

---

## MLOps 담당

* MLflow 실험 추적 통합

  * 파라미터 로깅
  * 메트릭 로깅
  * 아티팩트 저장
* 모델 레지스트리 구축
* CI/CD 파이프라인 초기 설계

**산출물**

* MLflow 통합 코드
* Model Registry 설정

---

# Week 3: 고도화 모델 개발 & 배포 준비

## 모델링 팀

* 딥러닝 모델 개발

  * LSTM (시계열 패턴 학습)
  * GRU (경량 시계열 모델)
  * 1D CNN (센서 신호 패턴 인식)
* NASA Bearing 데이터 통합

  * 진동 데이터 Feature 추출
  * 앙상블 모델 검증
* 하이퍼파라미터 튜닝 (Optuna)
* 모델 앙상블 전략 구현

**산출물**

* DL 모델 코드
* 튜닝 결과 리포트
* 최종 모델 선정

---

## DB 담당

* Feature Store 구축

  * 전처리된 Feature 저장
  * Feature 버전 관리
* 데이터 모니터링 쿼리 작성
* 성능 최적화 (인덱싱, 파티셔닝)
* 백업 및 복구 전략 수립

**산출물**

* Feature Store 설계
* 모니터링 쿼리
* 백업 스크립트

---

## MLOps 담당

* FastAPI 서버 구축

  * 모델 추론 엔드포인트
  * Health Check API
  * 모델 버전 관리 API
* Docker 컨테이너화
* 모델 배포 자동화 스크립트
* 로깅 시스템 구축

**산출물**

* FastAPI 서버
* Dockerfile
* 배포 스크립트

---

# Week 4: 통합 테스트 & 운영 준비

## 모델링 팀

* 최종 모델 검증

  * Test Set 성능 평가
  * 오분류 케이스 분석
  * Feature Importance 분석
* 모델 설명 가능성 (SHAP, LIME)
* 모델 성능 리포트 작성
* 프로덕션 모델 등록

**산출물**

* 모델 성능 리포트
* 해석 가능성 문서
* Production 모델

---

## DB 담당

* 전체 파이프라인 통합 테스트
* 데이터 일관성 검증
* 부하 테스트 (대량 데이터 처리)
* 장애 복구 시나리오 테스트
* 운영 매뉴얼 작성

**산출물**

* 테스트 결과 리포트
* 운영 매뉴얼

---

## MLOps 담당

* 모니터링 대시보드 구축 (Grafana)

  * 모델 성능 메트릭
  * API 응답 시간
  * 시스템 리소스 사용량
* 알림 시스템 구축 (Slack 연동)
* CI/CD 파이프라인 완성
* 문서화 (README, API Docs)
* 최종 배포 및 안정화

**산출물**

* Grafana 대시보드
* 알림 시스템
* CI/CD 파이프라인
* 프로젝트 문서

---

## 전체

* 프로젝트 회고 미팅
* 최종 발표 자료 준비
* Phase 2 계획 수립

**산출물**

* 회고 문서
* 발표 자료
* Phase 2 계획

---

## 4. 성공 지표 (KPI)

| 카테고리   | 지표               | 목표                  |
| ------ | ---------------- | ------------------- |
| 모델 성능  | F1-Score (Macro) | ≥ 0.85              |
| 모델 성능  | Recall (불량 클래스)  | ≥ 0.90              |
| API 성능 | 추론 응답 시간         | < 100ms (P95)       |
| 데이터 품질 | 데이터 적재 성공률       | ≥ 99.5%             |
| MLOps  | 모델 배포 자동화        | CI/CD 파이프라인 구축 완료   |
| 문서화    | 기술 문서 완성도        | README, API Docs 완료 |

---

## 5. 리스크 관리

| 리스크          | 영향도    | 대응 방안                              |
| ------------ | ------ | ---------------------------------- |
| 클래스 불균형 심화   | High   | SMOTE, Class Weight, Focal Loss 적용 |
| 모델 성능 목표 미달  | High   | 추가 Feature Engineering, 앙상블 기법     |
| 일정 지연        | Medium | 주간 스프린트 리뷰, 우선순위 조정                |
| DB 성능 이슈     | Medium | 인덱싱 최적화, 쿼리 튜닝, 캐싱                 |
| MLOps 인프라 장애 | Low    | Health Check, 자동 복구, 백업 전략         |

---

## 6. 기술 스택

| 카테고리   | 기술/도구                                   |
| ------ | --------------------------------------- |
| 언어     | Python 3.9+                             |
| 머신러닝   | Scikit-learn, XGBoost, LightGBM, Optuna |
| 딥러닝    | PyTorch, PyTorch Lightning              |
| 데이터 처리 | Pandas, NumPy, Matplotlib, Seaborn      |
| MLOps  | MLflow, DVC                             |
| 데이터베이스 | PostgreSQL                              |
| API    | FastAPI, Pydantic, Uvicorn              |
| 모니터링   | Prometheus, Grafana                     |
| 컨테이너   | Docker, Docker Compose                  |
| CI/CD  | GitHub Actions                          |
| 협업     | Git, GitHub, Slack, Notion              |

---

## 7. Phase 2 로드맵 (예정)

* Vision 모델 통합
* Pseudo Labeling
* 멀티모달 통합 (Sensor + Image)
* Closed-loop Control
* 자동 재학습 (Drift 감지 기반)

---

## 8. 마무리

본 Phase 1 프로젝트는 센서 데이터 기반 기계 불량 예측을 위한 MLOps 시스템의 기본 인프라를 구축하는 것을 목표로 합니다. 4주간의 집중적인 개발을 통해 모델링, 데이터 파이프라인, MLOps 인프라를 완성하고, 이를 기반으로 Phase 2에서 Vision 모델 및 Closed-loop 제어 시스템으로 확장할 예정입니다.

각 팀원의 역할과 책임이 명확히 정의되어 있으며, 주차별 산출물과 성공 지표를 통해 프로젝트 진행 상황을 지속적으로 추적할 것입니다.

