# Phase 2: MLOps Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** DB 연결 → ResNet18 학습(MLflow 추적) → FastAPI 추론 서버까지 동작하는 완전한 MLOps 파이프라인 구축

**Architecture:** `data/classification/` 이미지 데이터 → Jupyter 노트북에서 ResNet18 학습 + MLflow 기록 → `ml/models/vision/classification/` 저장 → FastAPI `/predict` 엔드포인트로 서빙. PostgreSQL에 예측 결과 저장. Prometheus로 API 메트릭 수집.

**Tech Stack:** Python 3.11, PyTorch 2.0, PyTorch Lightning 2.0, MLflow 2.7, FastAPI 0.103, SQLAlchemy 2.0, PostgreSQL 15, Docker Compose, pytest

---

## 현재 상태

**이미 존재하는 파일:**
- `src/models/vision/classification.py` — `CSVImageDataset`, `ResNetClassifier`, `get_transforms()`, `create_dataloaders()`
- `src/models/vision/__init__.py` — `ResNetClassifier` export
- `db/schema.sql` — DB 스키마 (7개 테이블)
- `docker/docker-compose.yml` — PostgreSQL + MLflow + API + Prometheus + Grafana
- `docker/Dockerfile.api` — FastAPI 컨테이너
- `config/config.yaml` — 프로젝트 설정
- `data/classification/images/` — 2227개 이미지
- `data/classification/labels.csv` — 멀티라벨 CSV

**누락된 파일 (이 플랜이 만드는 것):**
- `db/connection.py` — SQLAlchemy 연결 관리
- `db/init.py` — DB 초기화 스크립트
- `api/main.py` — FastAPI 앱
- `api/schemas.py` — Pydantic 요청/응답 모델
- `api/model_loader.py` — 모델 로딩 유틸리티
- `api/routes.py` — /predict, /health 라우트
- `docker/prometheus.yml` — Prometheus 설정
- `notebooks/vision/01_classification_train.ipynb` — 학습 노트북
- `tests/test_api.py` — API 테스트
- `tests/__init__.py` — pytest 패키지

---

## Task 1: DB 연결 레이어

**Files:**
- Create: `db/connection.py`
- Create: `db/init.py`
- Create: `tests/__init__.py`
- Create: `tests/test_db.py`

**Step 1: tests/__init__.py 생성**

```bash
touch /mnt/c/Users/user/r2r/r2r-mlops/tests/__init__.py
```

**Step 2: test_db.py 작성 (failing test 먼저)**

`tests/test_db.py`:

```python
import pytest
from unittest.mock import patch, MagicMock


def test_get_engine_returns_engine():
    """get_engine()이 SQLAlchemy engine을 반환해야 함"""
    with patch("db.connection.create_engine") as mock_create:
        mock_engine = MagicMock()
        mock_create.return_value = mock_engine

        from db.connection import get_engine
        engine = get_engine()

        assert engine is mock_engine
        mock_create.assert_called_once()


def test_get_engine_uses_env_vars():
    """get_engine()이 환경변수로 DB URL을 구성해야 함"""
    import os
    env = {
        "DB_HOST": "testhost",
        "DB_PORT": "5432",
        "DB_USER": "testuser",
        "DB_PASSWORD": "testpass",
        "DB_NAME": "testdb"
    }
    with patch.dict(os.environ, env):
        with patch("db.connection.create_engine") as mock_create:
            mock_create.return_value = MagicMock()
            # Re-import to pick up env vars
            import importlib
            import db.connection
            importlib.reload(db.connection)
            db.connection.get_engine()

            call_args = mock_create.call_args[0][0]
            assert "testhost" in call_args
            assert "testuser" in call_args
            assert "testdb" in call_args
```

**Step 3: 테스트 실패 확인**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
python3 -m pytest tests/test_db.py -v 2>&1 | tail -20
```

Expected: `ImportError: No module named 'db.connection'`

**Step 4: db/connection.py 작성**

```python
"""Database connection management using SQLAlchemy"""
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool


def _build_url() -> str:
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "password")
    name = os.getenv("DB_NAME", "r2r_coating")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


def get_engine():
    """Create SQLAlchemy engine from environment variables"""
    return create_engine(_build_url(), poolclass=NullPool)


def get_session():
    """Create a database session"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def check_connection() -> bool:
    """Return True if DB is reachable"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
```

**Step 5: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_db.py -v 2>&1 | tail -10
```

Expected: `2 passed`

**Step 6: db/init.py 작성**

```python
"""Initialize database schema"""
import os
from pathlib import Path
from db.connection import get_engine
from sqlalchemy import text


def init_db(schema_path: str = None) -> None:
    """Execute schema.sql against the connected database"""
    if schema_path is None:
        schema_path = Path(__file__).parent / "schema.sql"

    with open(schema_path, "r") as f:
        sql = f.read()

    engine = get_engine()
    with engine.connect() as conn:
        # Execute each statement (skip empty lines)
        for statement in sql.split(";"):
            stmt = statement.strip()
            if stmt and not stmt.startswith("--"):
                conn.execute(text(stmt))
        conn.commit()
    print("Database initialized successfully.")


if __name__ == "__main__":
    init_db()
```

**Step 7: Commit**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
git add db/connection.py db/init.py tests/__init__.py tests/test_db.py
git commit -m "feat: add DB connection layer and init script"
```

---

## Task 2: FastAPI 스켈레톤 + Health 엔드포인트

**Files:**
- Create: `api/main.py`
- Create: `api/schemas.py`
- Create: `tests/test_api.py`

**Step 1: test_api.py 작성 (failing test 먼저)**

`tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_structure(client):
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_docs_accessible(client):
    response = client.get("/docs")
    assert response.status_code == 200
```

**Step 2: 테스트 실패 확인**

```bash
python3 -m pytest tests/test_api.py -v 2>&1 | tail -10
```

Expected: `ImportError: No module named 'api.main'` 또는 `ModuleNotFoundError`

**Step 3: api/schemas.py 작성**

```python
"""Pydantic schemas for request/response"""
from pydantic import BaseModel
from typing import Dict, Optional
import datetime


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class PredictRequest(BaseModel):
    image_path: str  # 서버 내 절대 경로 또는 data/ 하위 상대 경로


class DefectPrediction(BaseModel):
    Surface_Crack: float
    Delamination: float
    Pinhole: float
    unclassified: float


class PredictResponse(BaseModel):
    image_path: str
    predictions: DefectPrediction
    predicted_labels: list[str]  # 0.5 이상인 클래스
    inference_time_ms: float
    model_version: str
```

**Step 4: api/main.py 작성 (라우터 없이 health만)**

```python
"""FastAPI application entry point"""
from fastapi import FastAPI
from api.schemas import HealthResponse
import datetime

app = FastAPI(
    title="R2R Coating Defect Detection API",
    description="Vision model inference server for coating defect detection",
    version="1.0.0"
)

# Model state (populated on startup)
_model_loaded: bool = False


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model_loaded,
        timestamp=datetime.datetime.utcnow().isoformat()
    )
```

**Step 5: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_api.py -v 2>&1 | tail -10
```

Expected: `3 passed`

**Step 6: Commit**

```bash
git add api/main.py api/schemas.py tests/test_api.py
git commit -m "feat: add FastAPI skeleton with /health endpoint"
```

---

## Task 3: 모델 로더 + /predict 엔드포인트

**Files:**
- Create: `api/model_loader.py`
- Create: `api/routes.py`
- Modify: `api/main.py`
- Modify: `tests/test_api.py`

**Step 1: test_api.py에 /predict 테스트 추가**

`tests/test_api.py` 끝에 추가:

```python
def test_predict_missing_file_returns_404(client):
    """존재하지 않는 이미지 경로는 404 반환"""
    response = client.post("/predict", json={"image_path": "/nonexistent/image.jpg"})
    assert response.status_code == 404


def test_predict_response_structure(client, tmp_path):
    """predict 응답이 올바른 구조를 가져야 함 (모킹)"""
    from unittest.mock import patch
    import numpy as np

    fake_scores = {"Surface_Crack": 0.8, "Delamination": 0.1, "Pinhole": 0.05, "unclassified": 0.05}

    # Create a dummy image file
    dummy_img = tmp_path / "test.jpg"
    from PIL import Image
    Image.new("RGB", (100, 100)).save(str(dummy_img))

    with patch("api.routes.run_inference") as mock_infer:
        mock_infer.return_value = (fake_scores, 12.3)
        response = client.post("/predict", json={"image_path": str(dummy_img)})

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "predicted_labels" in data
    assert "inference_time_ms" in data
```

**Step 2: 테스트 실패 확인**

```bash
python3 -m pytest tests/test_api.py::test_predict_missing_file_returns_404 -v 2>&1 | tail -10
```

Expected: `ImportError` 또는 `404 != 422` — 라우트 없음

**Step 3: api/model_loader.py 작성**

```python
"""Model loading utilities"""
import os
import torch
from pathlib import Path
from typing import Optional
from src.models.vision import ResNetClassifier

_model: Optional[ResNetClassifier] = None
_model_version: str = "unknown"

MODEL_DIR = Path(os.getenv("MODEL_DIR", "ml/models/vision/classification"))


def load_model() -> ResNetClassifier:
    """Load model from MODEL_DIR. Caches in module-level variable."""
    global _model, _model_version

    # Find latest .pth file
    pth_files = sorted(MODEL_DIR.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth model found in {MODEL_DIR}")

    model_path = pth_files[-1]
    _model_version = model_path.stem

    model = ResNetClassifier(num_classes=4, pretrained=False)
    state = torch.load(str(model_path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    _model = model
    return model


def get_model() -> ResNetClassifier:
    """Return cached model, loading if necessary."""
    global _model
    if _model is None:
        load_model()
    return _model


def get_model_version() -> str:
    return _model_version
```

**Step 4: api/routes.py 작성**

```python
"""API route definitions"""
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from fastapi import APIRouter, HTTPException
from torchvision import transforms
from PIL import Image

from api.schemas import PredictRequest, PredictResponse, DefectPrediction
from api.model_loader import get_model, get_model_version

router = APIRouter()

LABEL_COLS = ["Surface_Crack", "Delamination", "Pinhole", "unclassified"]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def run_inference(image_path: str) -> Tuple[Dict[str, float], float]:
    """Run model inference on a single image. Returns (scores_dict, inference_ms)."""
    model = get_model()
    image = Image.open(image_path).convert("RGB")
    tensor = _transform(image).unsqueeze(0)  # [1, 3, 224, 224]

    start = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().tolist()
    elapsed_ms = (time.time() - start) * 1000

    scores = {col: float(p) for col, p in zip(LABEL_COLS, probs)}
    return scores, elapsed_ms


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not Path(request.image_path).exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")

    scores, elapsed_ms = run_inference(request.image_path)
    predicted = [label for label, score in scores.items() if score >= 0.5]

    return PredictResponse(
        image_path=request.image_path,
        predictions=DefectPrediction(**scores),
        predicted_labels=predicted,
        inference_time_ms=round(elapsed_ms, 2),
        model_version=get_model_version()
    )
```

**Step 5: api/main.py 업데이트 (라우터 등록)**

`api/main.py`를 아래로 교체:

```python
"""FastAPI application entry point"""
from fastapi import FastAPI
from api.schemas import HealthResponse
from api.routes import router
import datetime

app = FastAPI(
    title="R2R Coating Defect Detection API",
    description="Vision model inference server for coating defect detection",
    version="1.0.0"
)

app.include_router(router)

_model_loaded: bool = False


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model_loaded,
        timestamp=datetime.datetime.utcnow().isoformat()
    )
```

**Step 6: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_api.py -v 2>&1 | tail -15
```

Expected: `5 passed` (3 기존 + 2 신규)

**Step 7: Commit**

```bash
git add api/model_loader.py api/routes.py api/main.py tests/test_api.py
git commit -m "feat: add /predict endpoint with model loader"
```

---

## Task 4: Prometheus 설정 + API 메트릭

**Files:**
- Create: `docker/prometheus.yml`
- Modify: `api/main.py` (메트릭 미들웨어 추가)

**Step 1: docker/prometheus.yml 작성**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "r2r-api"
    static_configs:
      - targets: ["api:8000"]
    metrics_path: "/metrics"
```

**Step 2: api/main.py에 /metrics 엔드포인트 추가**

`api/main.py`의 import 섹션과 app 정의 사이에 추가:

```python
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Prometheus metrics
PREDICT_COUNT = Counter("r2r_predict_total", "Total prediction requests")
PREDICT_LATENCY = Histogram("r2r_predict_latency_seconds", "Prediction latency")
```

`health()` 함수 아래에 `/metrics` 엔드포인트 추가:

```python
@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

그리고 `routes.py`의 `predict()` 함수 시작 부분에 추가:

```python
from api.main import PREDICT_COUNT, PREDICT_LATENCY
# predict() 함수 내:
PREDICT_COUNT.inc()
with PREDICT_LATENCY.time():
    scores, elapsed_ms = run_inference(request.image_path)
```

> **주의:** 실제로는 순환 import를 피하기 위해 metrics를 별도 `api/metrics.py`로 분리하는 것이 좋음. 아래처럼 작성:

`api/metrics.py`:

```python
from prometheus_client import Counter, Histogram

PREDICT_COUNT = Counter("r2r_predict_total", "Total prediction requests")
PREDICT_LATENCY = Histogram("r2r_predict_latency_seconds", "Prediction latency in seconds")
```

`api/main.py`와 `api/routes.py` 모두 `from api.metrics import PREDICT_COUNT, PREDICT_LATENCY`로 import.

**Step 3: requirements.txt에 prometheus_client 추가 확인**

```bash
grep "prometheus" /mnt/c/Users/user/r2r/r2r-mlops/requirements.txt
```

없으면:
```bash
echo "prometheus-client>=0.17.0" >> /mnt/c/Users/user/r2r/r2r-mlops/requirements.txt
```

**Step 4: 테스트 실행**

```bash
python3 -m pytest tests/test_api.py -v 2>&1 | tail -10
```

Expected: `5 passed`

**Step 5: Commit**

```bash
git add docker/prometheus.yml api/metrics.py api/main.py api/routes.py requirements.txt
git commit -m "feat: add Prometheus metrics endpoint and config"
```

---

## Task 5: Vision 학습 노트북

**Files:**
- Create: `notebooks/vision/01_classification_train.ipynb`

이 Task는 Jupyter 노트북 작성이므로 셀 단위로 구성. 노트북 실행 후 모델이 `ml/models/vision/classification/resnet18_defect.pth`에 저장되어야 함.

**Step 1: 노트북 생성**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
jupyter notebook notebooks/vision/01_classification_train.ipynb
```

**Step 2: 노트북 셀 구성**

**Cell 1 — Setup & Imports:**
```python
import sys
sys.path.insert(0, "/mnt/c/Users/user/r2r/r2r-mlops")

import os
import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from src.models.vision import ResNetClassifier
from src.models.vision.classification import create_dataloaders

# Config
DATA_DIR = "data/classification/images"
CSV_PATH = "data/classification/labels.csv"
MODEL_SAVE_DIR = "ml/models/vision/classification"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

print(f"MLflow URI: {MLFLOW_URI}")
```

**Cell 2 — MLflow 실험 설정:**
```python
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("r2r-coating-classification")
print("MLflow experiment set.")
```

**Cell 3 — DataLoaders 생성:**
```python
train_loader, val_loader = create_dataloaders(
    img_dir=DATA_DIR,
    csv_path=CSV_PATH,
    batch_size=16,
    train_size=0.8,
    num_workers=0  # Windows/WSL 호환
)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
```

**Cell 4 — 모델 및 트레이너 설정:**
```python
model = ResNetClassifier(num_classes=4, learning_rate=1e-4, pretrained=True)

mlflow_logger = MLFlowLogger(
    experiment_name="r2r-coating-classification",
    tracking_uri=MLFLOW_URI
)

checkpoint_cb = ModelCheckpoint(
    dirpath=MODEL_SAVE_DIR,
    filename="resnet18_defect",
    monitor="val_f1_macro",
    mode="max",
    save_top_k=1
)

early_stop_cb = EarlyStopping(
    monitor="val_f1_macro",
    patience=5,
    mode="max"
)

trainer = pl.Trainer(
    max_epochs=30,
    callbacks=[checkpoint_cb, early_stop_cb],
    logger=mlflow_logger,
    log_every_n_steps=10,
    accelerator="auto"
)
```

**Cell 5 — 학습 실행:**
```python
with mlflow.start_run():
    mlflow.log_params({
        "batch_size": 16,
        "learning_rate": 1e-4,
        "max_epochs": 30,
        "pretrained": True
    })
    trainer.fit(model, train_loader, val_loader)

    best_path = checkpoint_cb.best_model_path
    print(f"Best model saved: {best_path}")
```

**Cell 6 — 최종 메트릭 확인:**
```python
import json, torch
from sklearn.metrics import classification_report
import numpy as np

# Load best model
best_model = ResNetClassifier.load_from_checkpoint(checkpoint_cb.best_model_path)
best_model.eval()

# Collect val predictions
all_preds, all_labels = [], []
for images, labels in val_loader:
    with torch.no_grad():
        logits = best_model(images)
        preds = (torch.sigmoid(logits) > 0.5).float()
    all_preds.append(preds.numpy())
    all_labels.append(labels.numpy())

preds_np = np.vstack(all_preds)
labels_np = np.vstack(all_labels)

label_names = ["Surface_Crack", "Delamination", "Pinhole", "unclassified"]
print(classification_report(labels_np, preds_np, target_names=label_names, zero_division=0))
```

**Cell 7 — metrics.json 저장:**
```python
from sklearn.metrics import f1_score, recall_score

metrics = {
    "f1_macro": float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
    "recall_macro": float(recall_score(labels_np, preds_np, average="macro", zero_division=0)),
    "val_samples": len(labels_np)
}

with open(f"{MODEL_SAVE_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics:", metrics)
mlflow.log_metrics(metrics)
```

**Step 3: 노트북 실행 확인**

```bash
ls /mnt/c/Users/user/r2r/r2r-mlops/ml/models/vision/classification/
```

Expected: `resnet18_defect.ckpt` (또는 `.pth`) + `metrics.json` 파일 존재

**Step 4: Commit**

```bash
git add notebooks/vision/01_classification_train.ipynb ml/models/vision/classification/metrics.json
git commit -m "feat: add vision classification training notebook"
```

---

## Task 6: 최종 통합 테스트

**Step 1: 전체 테스트 실행**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
python3 -m pytest tests/ -v 2>&1 | tail -20
```

Expected:
```
tests/test_api.py::test_health_returns_200 PASSED
tests/test_api.py::test_health_response_structure PASSED
tests/test_api.py::test_docs_accessible PASSED
tests/test_api.py::test_predict_missing_file_returns_404 PASSED
tests/test_api.py::test_predict_response_structure PASSED
tests/test_db.py::test_get_engine_returns_engine PASSED
tests/test_db.py::test_get_engine_uses_env_vars PASSED
7 passed
```

**Step 2: Docker 스택 시작 (선택적 — 실제 인프라 확인)**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops/docker
docker-compose up -d postgres mlflow
sleep 10
docker-compose ps
```

Expected: `r2r_postgres`, `r2r_mlflow` 모두 `healthy`

**Step 3: DB 초기화 확인 (선택적)**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
python3 db/init.py
```

Expected: `Database initialized successfully.`

**Step 4: API 로컬 실행 확인 (선택적)**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
sleep 3
curl -s http://localhost:8000/health | python3 -m json.tool
```

Expected:
```json
{
  "status": "ok",
  "model_loaded": false,
  "timestamp": "..."
}
```

**Step 5: Final Commit**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
git add -A
git commit -m "test: verify full pipeline integration"
```

---

## 완료 기준 체크리스트

- [ ] `python3 -m pytest tests/ -v` → 7 passed
- [ ] `python3 db/init.py` → `Database initialized successfully.`
- [ ] `curl http://localhost:8000/health` → `{"status": "ok", ...}`
- [ ] `curl http://localhost:8000/metrics` → Prometheus 형식 텍스트
- [ ] `ml/models/vision/classification/metrics.json` 존재
- [ ] MLflow UI (http://localhost:5000) 에서 실험 결과 확인 가능
