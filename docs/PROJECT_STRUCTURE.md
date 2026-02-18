# 프로젝트 구조 개요 및 저장 위치

## 📍 전체 프로젝트 위치

```
C:\Users\mash8\sideproject\
├── R2R/                             ← 메인 프로젝트 (MLOps 인프라)
│   ├── data/
│   │   ├── classification/          ← Classification 데이터
│   │   ├── detection/               ← Detection 데이터
│   │   ├── segmentation/            ← Segmentation 데이터
│   │   ├── raw/                     ← 센서 Raw 데이터
│   │   ├── processed/               ← 처리된 센서 데이터
│   │   └── features/                ← 센서 Feature
│   │
│   ├── src/                         ← 소스 코드
│   │   └── models/
│   │       ├── vision/              ← Vision 모델 정의
│   │       └── sensor/              ← 센서 모델 정의
│   │
│   ├── notebooks/                   ← Jupyter 분석 노트북
│   │   ├── vision/                  ← Vision 모델 분석
│   │   └── sensor/                  ← 센서 모델 분석
│   │
│   ├── ml/
│   │   ├── experiments/             ← MLflow 실험 추적
│   │   └── models/                  ← 저장된 모델 (여기에 저장!)
│   │       ├── vision/
│   │       │   ├── classification/
│   │       │   ├── detection/
│   │       │   ├── segmentation/
│   │       │   └── production/
│   │       └── sensor/
│   │           ├── baseline/
│   │           ├── lstm/
│   │           └── production/
│   │
│   ├── config/                      ← 설정 파일
│   ├── docker/                      ← Docker 설정
│   ├── docs/                        ← 기술 문서
│   ├── README.md                    ← 프로젝트 설명 (여기서 읽으세요!)
│   └── ...
│
└── R2Rmachine/                      ← 기존 코드 (참고용)
    └── CoatingVision/               ← ResNet.ipynb가 있는 곳
        ├── ResNet.ipynb             ← 원본 코드 (이미 포팅됨)
        ├── classification/
        ├── detection/
        └── segmentation/
```

---

## 💾 모델 저장 장소

### 최종 모델 저장 위치

**모든 모델은 다음 위치에 저장하세요:**

```
C:\Users\mash8\sideproject\R2R\ml\models\
```

### 구체적인 저장 경로

#### Vision 모델
```
ml/models/vision/
├── classification/v1/          ← Classification 모델 버전 1
├── classification/v2/          ← Classification 모델 버전 2
├── detection/v1/               ← Detection 모델
├── segmentation/v1/            ← Segmentation 모델
└── production/                 ← 프로덕션 최종 모델
    ├── classification/
    ├── detection/
    └── segmentation/
```

#### 센서 모델
```
ml/models/sensor/
├── baseline/v1/                ← XGBoost 모델
├── baseline/v2/
├── lstm/v1/                    ← LSTM 모델
├── lstm/v2/
├── ensemble/v1/                ← 앙상블 모델
└── production/                 ← 프로덕션 최종 모델
    ├── baseline/
    ├── lstm/
    └── ensemble/
```

---

## 📝 저장 시 명명 규칙

### 파일명 규칙
```
{모델명}_{버전}.{확장자}

예시:
- resnet18_v1.pth         (PyTorch 모델)
- xgboost_v1_model.pkl    (XGBoost 모델)
- xgboost_v1_scaler.pkl   (전처리 스케일러)
- unet_v1.pth             (UNet 모델)
```

### 디렉토리 구조 (각 버전별)
```
ml/models/vision/classification/v1/
├── resnet18.pth                  ← 모델 가중치
├── config.yaml                   ← 모델 설정
├── metrics.json                  ← 성능 지표
└── training_log.txt              ← 학습 로그
```

---

## 🔄 데이터 흐름

```
1. 데이터 준비
   ↓
   data/classification/  (images + labels.csv)
   data/detection/       (images + labels/*.txt)
   data/segmentation/    (images + masks/)
   
2. 모델 학습
   ↓
   notebooks/vision/ 에서 노트북으로 실험
   
3. 모델 저장
   ↓
   ml/models/vision/classification/v1/
   ml/models/vision/detection/v1/
   ml/models/vision/segmentation/v1/
   
4. 모델 평가 & 선택
   ↓
   최고 성능 모델을 production으로 이동
   
5. API 배포
   ↓
   api/main.py 에서 production 모델 로딩
```

---

## 📌 중요 포인트

### ✅ 하기
```python
# 1. 모델마다 버전 관리
torch.save(model.state_dict(), "ml/models/vision/classification/v1/resnet18.pth")

# 2. 메타데이터도 저장
import json
with open("ml/models/vision/classification/v1/metrics.json", "w") as f:
    json.dump({
        "f1_macro": 0.87,
        "recall": 0.89
    }, f)

# 3. MLflow에 기록
mlflow.pytorch.log_model(model, "resnet18")
mlflow.log_metrics({"val_f1": 0.87})
```

### ❌ 금지
```python
# 같은 파일을 계속 덮어쓰기 금지
torch.save(model, "ml/models/vision/classification/resnet18.pth")  # ✗

# 버전 없이 저장 금지
torch.save(model, "ml/models/model.pth")  # ✗
```

---

## 🎯 현재 상황 요약

| 항목 | 위치 | 상태 |
|------|------|------|
| **Classification 데이터** | `R2R/data/classification/` | ✅ 준비됨 |
| **Detection 데이터** | `R2R/data/detection/` | ✅ 준비됨 |
| **Segmentation 데이터** | `R2R/data/segmentation/` | ✅ 준비됨 |
| **Vision 모델 코드** | `R2R/src/models/vision/` | ✅ 생성됨 |
| **센서 모델 코드** | `R2R/src/models/sensor/` | ✅ 생성됨 |
| **모델 저장소** | `R2R/ml/models/` | ✅ 폴더 생성됨 |
| **노트북** | `R2R/notebooks/vision/` | 🔄 작성 필요 |
| **MLOps 인프라** | `R2R/docker/` | 🔄 구성 필요 |

---

## 🚀 다음 단계

1. **Classification 모델 학습**
   ```bash
   cd C:\Users\mash8\sideproject\R2R
   jupyter notebook notebooks/vision/01_classification_eda.ipynb
   ```

2. **훈련된 모델 저장**
   ```
   → ml/models/vision/classification/v1/
   ```

3. **Detection 모델 학습**
   ```bash
   jupyter notebook notebooks/vision/03_detection_eda.ipynb
   ```

4. **Segmentation 모델 학습**
   ```bash
   jupyter notebook notebooks/vision/05_segmentation_eda.ipynb
   ```

5. **센서 모델 학습**
   ```bash
   jupyter notebook notebooks/sensor/01_eda.ipynb
   ```

