# 모델 저장 및 사용 가이드

## 📁 모델 저장 위치 구조

프로젝트의 모든 모델은 다음 위치에 저장됩니다:

```
ml/models/
├── vision/                          # Vision 모델들
│   └── classification/              # Classification 모델 저장소
│       ├── v1/                      # 버전별 저장
│       │   ├── resnet18.pth         # 모델 가중치
│       │   ├── config.yaml          # 모델 설정
│       │   └── metrics.json         # 성능 지표
│       ├── v2/
│       └── latest -> v2/            # 최신 버전 심볼릭 링크
│
└── sensor/                          # 센서 모델들
    ├── baseline/
    │   ├── v1/
    │   │   ├── xgboost.pkl
    │   │   ├── scaler.pkl
    │   │   └── metrics.json
    │   └── latest -> v1/
    │
    ├── lstm/
    │   ├── v1/
    │   │   ├── lstm.pth
    │   │   ├── config.yaml
    │   │   └── metrics.json
    │   └── latest -> v1/
    │
    ├── ensemble/
    │   ├── v1/
    │   └── latest -> v1/
    │
    └── production/
        ├── baseline/
        ├── lstm/
        └── ensemble/
```

---

## 🔧 모델 저장 방법

### 1. Vision 모델 (PyTorch)

#### Classification
```python
from src.models.vision import ResNetClassifier
import pytorch_lightning as pl

# 모델 학습
model = ResNetClassifier(num_classes=4)
trainer = pl.Trainer(max_epochs=50)
trainer.fit(model, train_loader, val_loader)

# 모델 저장
trainer.save_checkpoint("ml/models/vision/classification/v1/resnet18.pth")
# 또는
torch.save(model.model.state_dict(), "ml/models/vision/classification/v1/resnet18.pth")
```

### 2. 센서 모델

#### Baseline (XGBoost/Random Forest)
```python
from src.models.sensor import BaselineModel

model = BaselineModel(model_type="xgboost")
model.fit(X_train, y_train, X_val, y_val)

# 저장
model.save("ml/models/sensor/baseline/v1/xgboost")
# → ml/models/sensor/baseline/v1/xgboost_model.pkl
# → ml/models/sensor/baseline/v1/xgboost_scaler.pkl
```

#### LSTM
```python
from src.models.sensor import LSTMModel
import pytorch_lightning as pl

model = LSTMModel(input_size=7, hidden_size=64, num_classes=5)
trainer = pl.Trainer(max_epochs=50)
trainer.fit(model, train_loader, val_loader)

torch.save(model.state_dict(), "ml/models/sensor/lstm/v1/lstm.pth")
```

---

## 📊 메타데이터 저장 (config.yaml & metrics.json)

### config.yaml 예시
```yaml
# ml/models/vision/classification/v1/config.yaml
model:
  name: resnet18
  type: classification
  version: v1
  framework: pytorch
  
training:
  epochs: 50
  batch_size: 16
  learning_rate: 1e-4
  
data:
  train_size: 0.8
  val_size: 0.1
  test_size: 0.1
  
augmentation:
  - RandomHorizontalFlip
  - RandomRotation
```

### metrics.json 예시
```json
{
  "train_loss": 0.1234,
  "val_loss": 0.1567,
  "val_f1_macro": 0.87,
  "val_recall_macro": 0.89,
  "val_precision_macro": 0.85,
  "inference_time_ms": 45
}
```

---

## 🚀 모델 로딩 및 사용

### 1. Classification 모델 로딩
```python
import torch
from src.models.vision import ResNetClassifier

# 모델 로드
model = ResNetClassifier.load_from_checkpoint(
    "ml/models/vision/classification/latest/resnet18.pth"
)
model.eval()

# 추론
with torch.no_grad():
    predictions = model(images)
    probabilities = torch.sigmoid(predictions)
```

### 2. Baseline 모델 로딩
```python
from src.models.sensor import BaselineModel

model = BaselineModel(model_type="xgboost")
model.load("ml/models/sensor/baseline/v1/xgboost")

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

---

## 🔄 MLflow 통합

모든 모델 학습 시 MLflow에서 자동 추적:

```python
import mlflow

with mlflow.start_run():
    # 파라미터 로깅
    mlflow.log_params({
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.001
    })
    
    # 메트릭 로깅
    mlflow.log_metrics({
        "val_f1": 0.87,
        "val_recall": 0.89,
        "inference_time_ms": 45
    })
    
    # 모델 저장
    mlflow.pytorch.log_model(model, "resnet18")
    # 또는
    mlflow.sklearn.log_model(sklearn_model, "xgboost")
    
    # 아티팩트 저장
    mlflow.log_artifact("ml/models/vision/classification/v1/config.yaml")
```

MLflow UI에서 확인:
```bash
mlflow ui --host localhost --port 5000
```
→ http://localhost:5000

---

## 📌 버전 관리 전략

### 개발 과정
1. **v1, v2, v3...** - 개발 버전들
2. **latest** - 가장 최신 개발 버전 (심볼릭 링크)
3. **production** - 프로덕션 배포 모델 (검증완료)

### 배포
1. 개발 완료 후 성능 검증
2. `ml/models/{task}/production/` 에 복사
3. 기존 production 모델 백업
4. API 서버에서 production 모델만 사용

---

## 📈 체크포인트 및 Early Stopping

PyTorch Lightning에서 자동 저장:

```python
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_callback = ModelCheckpoint(
    dirpath="ml/models/vision/classification/v1",
    filename="{epoch:02d}-{val_f1_macro:.2f}",
    monitor="val_f1_macro",
    mode="max",
    save_top_k=3
)

early_stopping = EarlyStopping(
    monitor="val_f1_macro",
    patience=10,
    mode="max"
)

trainer = pl.Trainer(
    callbacks=[checkpoint_callback, early_stopping],
    max_epochs=50
)
```

---

## 🔐 권장 사항

✅ **해야 할 것**
- 각 모델마다 config.yaml과 metrics.json 저장
- 버전 관리 (v1, v2, ...)로 실험 추적
- MLflow에서 실험 메타데이터 기록
- Production 모델은 별도 위치에 관리
- 정기적인 백업

❌ **하지 말아야 할 것**
- 모델을 git에 직접 커밋하지 않기
- 버전 없이 같은 파일명으로 덮어쓰기
- Production 모델을 임의로 변경하기

---

## 📝 저장 경로 단축 명령어

프로젝트 루트의 `config/config.yaml`에서:

```yaml
model_paths:
  vision:
    classification: "./ml/models/vision/classification"

  sensor:
    baseline: "./ml/models/sensor/baseline"
    lstm: "./ml/models/sensor/lstm"
    production: "./ml/models/sensor/production"
```

Python에서 사용:
```python
from src.utils.config import get_config

config = get_config()
classification_path = config["model_paths"]["vision"]["classification"]
```

