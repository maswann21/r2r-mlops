# R2R Coating 불량 감지 및 자동 최적화 MLOps 프로젝트

---

## 1. 프로젝트 개요

### 1.1 프로젝트 목적

본 프로젝트는 R2R(Roll-to-Roll) 코팅 공정에서 발생하는 불량을 효과적으로 감지하고 예방하기 위한 MLOps 시스템 구축을 목표로 합니다.

이미지 기반 CNN 모델을 활용하여 불량을 정확히 판별하고, 해당 결과를 기반으로 센서 데이터만으로 불량을 사전 예측함으로써 공정을 자동으로 최적화하는 **Closed-loop 시스템**을 구현합니다.

---

### 1.2 핵심 목표

* **Vision 모델**: 이미지 기반 CNN을 통한 고정밀 불량 판별 시스템 구축
* **Pseudo Labeling**: Vision 모델의 예측 결과를 운영 라벨로 활용하여 대량의 학습 데이터 확보
* **Sensor 모델**: 기계 컨디션 데이터만으로 불량을 사전 예측하는 ML/DL 모델 개발
* **자동 최적화**: 불량 확률 기반 공정 파라미터 자동 조정 Closed-loop 시스템 구현
* **MLOps 운영**: 지속적인 모델 성능 모니터링 및 재학습 자동화 체계 구축

---

### 1.3 기술 키워드

CNN, Multi-label Classification, Pseudo Labeling, Time-series Analysis,
LSTM/Transformer, Closed-loop Control, MLOps, Model Monitoring, Auto Retraining

---

# 2. 시스템 아키텍처

---

## 2.1 전체 아키텍처 개요

본 시스템은 크게 4개의 핵심 모듈로 구성되며, 데이터 수집부터 공정 제어까지의 전체 프로세스를 자동화합니다.

---

### 2.1.1 데이터 플로우

1. 이미지 수집

   * R2R 공정에서 실시간 코팅 이미지 캡처

2. Vision 모델 추론

   * CNN 기반 불량 판별
   * (Surface_Crack, Delamination, Pinhole)

3. 라벨 저장

   * 예측 결과를 Pseudo Label로 DB 저장
   * 라벨, 확률값, 타임스탬프

4. 센서 데이터 수집

   * 동일 시점의 기계 컨디션 데이터 수집
   * (온도, 속도, 압력, 전류, 진동 등)

5. Sensor 모델 학습

   * 센서 데이터 기반 불량 사전 예측 모델 학습

6. 공정 제어

   * 불량 확률 기반 공정 파라미터 자동 조정

7. 피드백 루프

   * 조정 결과 재수집 및 모델 재학습 (Closed-loop)

---

## 2.2 불량 유형 정의

Multi-label Classification 방식을 채택하여, 하나의 이미지에서 여러 불량이 동시에 발생할 수 있는 실제 공정 상황을 반영합니다.

| 불량 유형         | 설명            | 우선순위   |
| ------------- | ------------- | ------ |
| Surface_Crack | 코팅 표면 균열 및 크랙 | High   |
| Delamination  | 코팅층 분리 및 박리   | High   |
| Pinhole       | 미세 구멍 및 기공    | Medium |
| Unclassified  | 정상 또는 미분류 상태  | -      |

---

# 3. 모델 개발 전략

---

## 3.1 Vision 모델 (CNN)

### 3.1.1 모델 구조

* Base Model: ResNet18 (pretrained on ImageNet)
* Task: Multi-label Classification
* Output: Sigmoid activation (각 클래스별 확률값 0~1)
* Loss: Binary Cross-Entropy

---

### 3.1.2 성능 평가 지표

* Primary: F1-macro
* Secondary: 클래스별 Recall
* Critical Analysis: False Negative(FN) 중심 분석
* Threshold Tuning: Recall 우선 전략

---

### 3.1.3 데이터 검증 전략

* 롤/배치/시간 단위 분할로 데이터 누수 방지
* 오분류 샘플 분석
* 실제 공정 환경 추론 속도 및 안정성 검증

---

## 3.2 Sensor 모델 (ML/DL)

### 3.2.1 입력 데이터

시계열 센서 데이터 기반 사전 예측 모델

| 센서 유형 | 측정 항목           | 수집 주기 |
| ----- | --------------- | ----- |
| 온도    | 코팅 챔버 온도, 건조 온도 | 1초    |
| 속도    | 롤 이송 속도, 코팅 속도  | 0.5초  |
| 압력    | 코팅 롤 압력, 텐션     | 1초    |
| 전류    | 모터 전류, 히터 전류    | 1초    |
| 진동    | 롤러 진동           | 0.1초  |

---

### 3.2.2 모델 접근법

* Baseline ML: XGBoost, LightGBM
* Feature Engineering: 평균, 분산, 기울기 등
* Window 기반 집계
* LSTM, GRU, Transformer
* Attention 기반 중요 시점 포착

---

### 3.2.3 핵심 평가 요소

* 사전 예측 정확도
* Lead Time (예: 5~10분 전 예측)

---

# 4. MLOps 파이프라인

---

## 4.1 데이터 수집 및 관리

### 4.1.1 데이터 수집

* Vision 예측 결과 자동 저장
* 센서 데이터 실시간 수집
* 데이터 품질 검증 (누락, 이상치)

---

### 4.1.2 DB 스키마 설계

* Raw Data Layer
* Processed Layer
* Label Layer (Pseudo Label)
* Metadata Layer

---

## 4.2 실험 관리

### MLflow 기반

* 실험 추적
* 모델 버전 관리
* 재현성 보장
* 실험 비교 분석

---

### 모델 검증

* Offline Evaluation
* A/B Testing
* Shadow Mode
* Gradual Rollout

---

## 4.3 모델 배포

### Vision 모델

* Edge 또는 On-prem GPU
* TensorRT 최적화

### Sensor 모델

* FastAPI 기반 REST 또는 gRPC

---

## 4.4 모니터링

* 실시간 F1, Recall, Lead Time 추적
* Data Drift 감지
* Model Drift 감지
* Slack / Email 알림

---

## 4.5 자동 재학습

### 트리거 조건

* 성능 저하
* 데이터 누적
* 주기적 재학습
* 수동 트리거

### 재학습 프로세스

* 데이터 준비
* 모델 학습
* 성능 검증
* 자동 배포
* 롤백 가능 구조

---

# 5. Closed-loop 자동 최적화

---

## 5.1 개념

불량 예측 결과를 공정 제어 시스템에 피드백하여
불량 발생 이전에 공정 파라미터 자동 조정

---

## 5.2 제어 전략

### 임계값 기반 제어

* 예: Surface_Crack > 0.7
* Warning / Critical 단계 운영

---

### 제어 파라미터

| 불량 유형         | 조정 파라미터   | 방향 |
| ------------- | --------- | -- |
| Surface_Crack | 건조 온도, 속도 | ↓  |
| Delamination  | 압력, 텐션    | ↑  |
| Pinhole       | 코팅 두께     | ↑  |

---

### 제어 로직

* Rule-based
* ML-based (강화학습)
* Hybrid 전략

---

## 5.3 효과 측정

* 불량률 감소
* 수율 향상
* Lead Time 단축
* 비용 절감

---

# 6. 팀 구성

| 팀원       | 담당 영역    | 주요 업무                          |
| -------- | -------- | ------------------------------ |
| 팀원1 (본인) | MLOps 총괄 | 아키텍처 설계, 운영 전략, Closed-loop 설계 |
| 팀원2      | Vision   | CNN 모델 개발 및 개선                 |
| 팀원3      | Sensor   | 시계열 모델 개발 및 평가                 |
| 팀원4      | 시스템      | 데이터 파이프라인, 서빙, 제어 로직           |

---

# 7. 기술 스택

| 카테고리  | 기술                     |
| ----- | ---------------------- |
| 언어    | Python, Node.js        |
| 딥러닝   | PyTorch, TorchVision   |
| ML    | XGBoost, LightGBM      |
| DB    | PostgreSQL, InfluxDB   |
| MLOps | MLflow                 |
| API   | FastAPI, gRPC          |
| 모니터링  | Prometheus, Grafana    |
| CI/CD | GitHub Actions, Docker |

---

# 8. 기대 효과

## 8.1 정량적

* 불량률 30~50% 감소
* Lead Time 5~10분 단축
* 비용 절감

## 8.2 정성적

* 데이터 기반 공정 관리
* 자동화 품질 시스템
* 지속적 개선 체계 구축

---

# 9. 프로젝트 일정 (예시)

| Phase      | 주요 작업       | 기간 |
| ---------- | ----------- | -- |
| Week 1-2   | 기획 및 정의     | 2주 |
| Week 3-5   | Vision 개발   | 3주 |
| Week 6-7   | 데이터 파이프라인   | 2주 |
| Week 8-10  | Sensor 개발   | 3주 |
| Week 11-12 | Closed-loop | 2주 |
| Week 13-14 | MLOps 구축    | 2주 |
| Week 15-16 | 통합 테스트      | 2주 |

---

# 10. 결론

본 프로젝트는 단순 모델 개발을 넘어
**Vision + Sensor + Closed-loop + MLOps 전체 생명주기 시스템 구축 프로젝트**입니다.

제조 현장에서 지속 가능하게 운영될 수 있는
실전 MLOps 기반 스마트 공정 자동화 시스템 구현을 목표로 합니다.


