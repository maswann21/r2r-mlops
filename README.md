# R2R Coating 불량 감지 및 자동 최적화 MLOps 프로젝트

R2R(Roll-to-Roll) 코팅 공정의 불량을 **Vision 모델 (Classification/Detection/Segmentation)** 과 **센서 데이터** 로 감지하고, 자동 최적화하는 통합 MLOps 시스템입니다.

---

## 📋 프로젝트 개요

### 핵심 목표

#### Vision 모델 (이미지 기반 불량 감지)
1. **Classification**: 불량 유형 분류 (Surface_Crack, Delamination, Pinhole, Unclassified)
   - ResNet18 기반 Multi-label Classification
   - 데이터: `data/classification/` (images + labels.csv)

2. **Detection**: 불량 위치 탐지 
   - YOLO/Faster R-CNN 기반 Object Detection
   - 데이터: `data/detection/` (images + labels/*.txt)

3. **Segmentation**: 불량 영역 분할
   - UNet/DeepLab 기반 Semantic/Instance Segmentation
   - 데이터: `data/segmentation/` (images + masks)

#### Sensor 모델 (시계열 기반 사전 예측)
- 센서 데이터 기반 불량 사전 예측
- ML/DL 모델 (XGBoost, LSTM, Transformer)

#### MLOps & Closed-loop 제어
- **Pseudo Labeling**: Vision 예측 결과를 학습 라벨로 활용
- **MLOps 운영**: 실험 관리, 모델 배포, 성능 모니터링, 자동 재학습
- **Closed-loop 제어**: 불량 확률 기반 공정 파라미터 자동 조정

### Phase 1 범위 
Vision 모델 3가지 (Classification/Detection/Segmentation) + MLOps 인프라

---

## 📁 프로젝트 구조

```
R2R/
├── README.md                          # 프로젝트 설명서
├── WBS.md                             # 상세 작업 분해 구조
├── R2R Coating 불량감지...md         # 전체 프로젝트 설명
│
├── data/                              # 데이터 관리
│   ├── classification/                # Classification 데이터
│   │   ├── images/                    # 코팅 표면 이미지
│   │   └── labels.csv                 # Multi-label (Surface_Crack, Delamination, Pinhole)
│   │
│   ├── detection/                     # Detection 데이터
│   │   ├── images/                    # 코팅 표면 이미지
│   │   └── labels/                    # Bounding box 어노테이션 (.txt)
│   │
│   ├── segmentation/                  # Segmentation 데이터
│   │   ├── images/                    # 코팅 표면 이미지
│   │   └── masks/                     # 불량 영역 마스크
│   │
│   ├── raw/                           # 센서 데이터 (Kaggle, NASA)
│   ├── processed/                     # 전처리된 센서 데이터
│   └── features/                      # Feature Store (센서 Feature)
│
├── src/                               # 소스 코드
│   ├── data/                          # 데이터 처리 모듈
│   │   ├── __init__.py
│   │   │
│   │   ├── vision/                    # Vision 모델
│   │   │   ├── classification.py      # ResNet, EfficientNet 등
│   │   │   ├── detection.py           # YOLO, Faster R-CNN 등
│   │   │   └── segmentation.py        # UNet, DeepLab 등
│   │   │
│   │   └── sensor/                    # 센서 데이터 모델
│   │       ├── baseline.py            # ML (RF, XGBoost, LightGBM)
│   │       ├── lstm.py                # LSTM 시계열
│   │       ├── gru.py                 # GRU 모델
│   │       ├── cnn1d.py               # 1D CNN
│   │       └── ensemble.py            # 앙상블e Engineering
│   │   ├── __init__.py
│   │   ├── engineer.py                # Feature 생성 (평균, 분산, 기울기)
│   │   ├── nasa_features.py           # NASA 베어링 데이터 특화 Feature
│   │   └── scaler.py                  # 정규화/표준화
│   │
│   ├── models/                        # 모델 정의
│   │   ├── __init__.py
│   │   ├── baseline.py                # ML 모델 (RF, XGBoost, LightGBM)
│   │   ├── lstm.py                    # LSTM 시계열 모델
│   │   ├── gru.py                     # GRU 모델
│   │   ├── cnn1d.py                   # 1D CNN
│   │   └── ensemble.py                # 앙상블 모델
│   │
│   ├── training/                      # 학습 파이프라인
│   │   ├── __init__.py
│   │   ├── train.py                   # 학습 엔트리 포인트
│   │   ├── evaluate.py                # 성능 평가
│   │   ├── callbacks.py               # Learning rate scheduler, EarlyStopping
│   │   └── logger.py                  # MLflow 통합
│   │
│   │
│   ├── vision/                        # Vision 모델 분석
│   │   ├── 01_classification_eda.ipynb       # Classification EDA
│   │   ├── 02_classification_train.ipynb     # Classification 학습
│   │   ├── 03_detection_eda.ipynb            # Detection EDA
│   │   ├── 04_detection_train.ipynb          # Detection 학습
│   │   ├── 05_segmentation_eda.ipynb         # Segmentation EDA
│   │   └── 06_segmentation_train.ipynb       # Segmentation 학습
│   │
│   └── sensor/                        # 센서 데이터 분석
│       ├── 01_eda.ipynb               # 탐색적 데이터 분석
│       ├── vision/
│       │   ├── classification/        # Classification 모델 (.pth, .pkl)
│       │   ├── detection/             # Detection 모델 (.pth, .pkl)
│       │   ├── segmentation/          # Segmentation 모델 (.pth, .pkl)
│       │   └── production/            # Production 배포 모델
│       │
│       └── sensor/                    # 센서 모델
│           ├── baseline/              # ML 모델
│           ├── lstm/
│           ├── ensemble/
│           └── production/            # Production 배포 모델egration.ipynb  # 성능 지표 (F1, Recall, 등)
│
├── notebooks/                         # Jupyter 분석 노트북
│   ├── 01_eda.ipynb                   # 탐색적 데이터 분석
│   ├── 02_feature_engineering.ipynb   # Feature 생성 및 검증
│   ├── 03_baseline_models.ipynb       # Baseline ML 모델
│   ├── 04_dl_models.ipynb             # 딥러닝 모델
│   └── 05_nasa_integration.ipynb      # NASA 데이터 통합
│
├── ml/                                # MLOps 관련
│   ├── experiments/                   # MLflow 실험 추적
│   │   └── mlruns/
│   │
│   └── models/                        # 학습된 모델 저장
│       ├── baseline/
│       ├── lstm/
│       ├── ensemble/
│       └── production/
│
├── api/                               # API 서버
│   ├── __init__.py
│   ├── main.py                        # FastAPI 메인
│   ├── routes.py                      # 엔드포인트
│   ├── schemas.py                     # 요청/응답 스키마 (Pydantic)
│   └── models.py                      # 모델 로딩 및 추론
│
├── db/                                # 데이터베이스
│   ├── schema.sql                     # DB 테이블 정의
│   ├── init.py                        # DB 초기화
│   └── connection.py                  # DB 연결 관리
│
├── config/                            # 설정 파일
│   ├── config.yaml                    # 프로젝트 설정
│   ├── params.yaml                    # 모델 하이퍼파라미터
│   └── logging.config                 # 로깅 설정
│
├── docker/                            # Docker 컨테이너
│   ├── Dockerfile.api                 # API 서버 이미지
│   ├── Dockerfile.training            # 학습 이미지
│   └── docker-compose.yml             # 멀티 컨테이너 오케스트레이션
│
├── pipeline/                          # 데이터/학습 파이프라인
│   ├── etl.py                         # ETL 파이프라인 (Kaggle → DB)
│   ├── feature_pipeline.py            # Feature 생성 파이프라인
│   ├── training_pipeline.py           # 모델 학습 파이프라인
│   └── inference_pipeline.py          # 추론 파이프라인
│
├── tests/                             # 테스트
│   ├── test_data.py                   # 데이터 로딩 테스트
│   ├── test_features.py               # Feature Engineering 테스트
│   ├── test_models.py                 # 모델 추론 테스트
│   └── test_api.py                    # API 엔드포인트 테스트
│
├── docs/                              # 문서
│   ├── db_schema.md                   # DB 설계 문서
│   ├── api_docs.md                    # API 문서
│   ├── model_evaluation.md            # 모델 성능 평가
│   └── deployment.md                  # 배포 가이드
│
├── requirements.txt                   # Python 의존성
├── .gitignore                         # Git 무시 파일
├── .env.example                       # 환경변수 템플릿
└── setup.py                           # 패키지 설정
```

---

## 🗂️ 데이터 폴더 구조

### Vision 데이터

#### `data/classification/` - Multi-label Classification
```
classification/
├── images/                            # 코팅 표면 이미지들
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── labels.csv                         # 이미지별 불량 라벨
    # 구조: file_name, Surface_Crack, Delamination, Pinhole, unclassified
    # 값: 0 또는 1 (다중 라벨)
```

#### `data/detection/` - Object Detection (Bounding Box)
```
detection/
├── images/                            # 코팅 표면 이미지들
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── labels/                            # YOLO 형식 어노테이션
    ├── image_001.txt                  # 형식: <class> <x> <y> <w> <h> (정규화)
    ├── image_002.txt
    └── ...
```

#### `data/segmentation/` - Semantic/Instance Segmentation
```
segmentation/
├── images/                            # 원본 이미지
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── masks/                             # 불량 영역 마스크 (PNG)
    ├── img_001_mask.png               # 픽셀 값: 클래스 ID
    ├── img_002_mask.png
    └── ...
```

### 센서 데이터

#### `data/raw/` - 원본 센서 데이터
```
raw/
├── machine_failure_prediction.csv     # Kaggle 데이터
├── nasa_bearing/
│   ├── bearing1.csv
│   ├── bearing2.csv
│   └── bearing4.csv
└── data_info.md
```

#### `data/processed/` - 전처리된 센서 데이터
```
processed/
├── train.csv
├── val.csv
├── test.csv
└── preprocessing_log.json
```

#### `data/features/` - Feature Store (센서 Feature)
```
features/
├── features_v1.pkl
├── features_v2.pkl
├── feature_names.json
└── feature_importance.csv
```

---

## 🔧 기술 스택

| 카테고리  | 기술                     |
|---------|--------------------------|
| 언어     | Python 3.9+              |
| ML      | Scikit-learn, XGBoost, LightGBM, Optuna |
| DL      | PyTorch, PyTorch Lightning |
| 데이터   | Pandas, NumPy, Matplotlib |
| MLOps   | MLflow, DVC              |
| DB      | PostgreSQL               |
| API     | FastAPI, Uvicorn         |
| 모니터링 | Prometheus, Grafana      |
| 컨테이너 | Docker, Docker Compose   |
| CI/CD   | GitHub Actions           |

---

## 🚀 시작하기

### 1. 환경 설정
```bash
# Python 가상환경 생성
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 설정 파일 준비
```bash
cp .env.example .env
# .env 파일에서 DB, API 정보 수정
```

### 3. DB 초기화
```bash
python db/init.py
```

### 4. 데이터 다운로드 및 전처리
```bash
# Kaggle Machine Failure 데이터 다운로드
# → data/raw/machine_failure_prediction.csv

# ETL 파이프라인 실행
python pipeline/etl.py
```

### 5. 모델 학습
```bash
python src/training/train.py --model baseline
python src/training/train.py --model lstm
```

### 6. API 서버 실행
```bash
uvicorn api.main:app --reload
```

---

## 📊 주요 산출물

### Vision 모델 개발
- ✅ Classification EDA ([notebooks/vision/01_classification_eda.ipynb](notebooks/vision/01_classification_eda.ipynb))
- ✅ Classification 학습 ([notebooks/vision/02_classification_train.ipynb](notebooks/vision/02_classification_train.ipynb))
- 🔄 Detection EDA ([notebooks/vision/03_detection_eda.ipynb](notebooks/vision/03_detection_eda.ipynb))
- 🔄 Detection 학습 ([notebooks/vision/04_detection_train.ipynb](notebooks/vision/04_detection_train.ipynb))
- 🔄 Segmentation EDA ([notebooks/vision/05_segmentation_eda.ipynb](notebooks/vision/05_segmentation_eda.ipynb))
- 🔄 Segmentation 학습 ([notebooks/vision/06_segmentation_train.ipynb](notebooks/vision/06_segmentation_train.ipynb))

### 센서 모델 개발
- 🔄 EDA 분석 리포트 ([notebooks/sensor/01_eda.ipynb](notebooks/sensor/01_eda.ipynb))
- 🔄 Feature 정의서 ([notebooks/sensor/02_feature_engineering.ipynb](notebooks/sensor/02_feature_engineering.ipynb))
- 🔄 Baseline 모델 ([notebooks/sensor/03_baseline_models.ipynb](notebooks/sensor/03_baseline_models.ipynb))
- 🔄 딥러닝 모델 ([notebooks/sensor/04_dl_models.ipynb](notebooks/sensor/04_dl_models.ipynb))

### MLOps 인프라
- 🔄 DB 스키마 ([db/schema.sql](db/schema.sql))
- 🔄 FastAPI 서버 ([api/main.py](api/main.py))
- 🔄 모니터링 대시보드 (Grafana)
- 🔄 CI/CD 파이프라인 (GitHub Actions)

## 📈 성능 지표 (KPI)

### Vision 모델
| 모델 | 메트릭 | 목표 |
|------|--------|------|
| Classification | F1-Score (Macro) | ≥ 0.85 |
| Classification | Recall (불량 클래스) | ≥ 0.90 |
| Detection | mAP (Mean Average Precision) | ≥ 0.80 |
| Segmentation | mIoU (Mean Intersection over Union) | ≥ 0.75 |

### Sensor 모델
| 메트릭 | 목표 |
|--------|------|
| F1-Score (Macro) | ≥ 0.85 |
| Recall (불량 클래스) | ≥ 0.90 |

### MLOps
| 메트릭 | 목표 |
|--------|------|
| API 응답시간 (P95) | < 100ms |
| 데이터 적재 성공률 | ≥ 99.5% |

---

## 👥 팀 구성

| 역할 | 담당 | 주요 책임 |
|------|------|---------|
| 모델링 팀 | 2명 | EDA, Feature Engineering, 모델 개발 |
| DB 담당 | 1명 | DB 설계, ETL 파이프라인, 데이터 관리 |
| MLOps | 1명 | 실험 관리, 배포, 모니터링 |

---

## 📚 참고 문서

- [전체 프로젝트 계획](R2R%20Coating%20불량감지%20및%20자동%20최적화%20MLOps%20프로젝트설명.md)
- [상세 WBS](WBS.md)
- [DB 설계 가이드](docs/db_schema.md)
- [API 문서](docs/api_docs.md)
- [배포 가이드](docs/deployment.md)

---

## 🔗 데이터셋

- **Machine Failure Prediction**: [Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
- **NASA Bearing Dataset**: [NASA Prognostics](https://www.nasa.gov/intelligent-systems-division/)

---

## 📝 라이센스

프로젝트 내부용 (비공개)

---

## 💬 문의

프로젝트 관련 질문은 팀 Slack 채널 `#r2r-mlops`에서 논의합니다.
