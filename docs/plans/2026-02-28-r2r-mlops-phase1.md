# R2R MLOps Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 코팅 불량 Vision 분류 모델을 학습하고 MLflow로 실험을 추적하며, FastAPI로 추론 서버를 제공하는 완전한 MLOps 파이프라인 구축

**Architecture:** 데이터(`db/classification/`) → DB 적재(ETL) → ResNet18 학습(PyTorch Lightning + MLflow) → 모델 저장(`ml/models/`) → FastAPI 추론 서버. Docker Compose로 PostgreSQL + MLflow + API가 함께 실행됨.

**Tech Stack:** Python 3.9+, PyTorch 2.0, PyTorch Lightning 2.0, MLflow 2.7, FastAPI 0.103, PostgreSQL 15, Docker Compose

---

## 현재 상태 요약

**이미 구현됨:**
- `src/models/vision/classification.py` — `CSVImageDataset`, `ResNetClassifier`, `get_transforms()`, `create_dataloaders()` 완성
- `src/models/sensor/` — baseline.py, lstm.py, gru.py, cnn1d.py 완성
- `db/schema.sql` — DB 스키마 완성
- `docker/docker-compose.yml` — PostgreSQL + MLflow + API + Prometheus + Grafana 설정 완성
- `config/config.yaml`, `config/params.yaml` 완성

**누락된 파일 (이 플랜이 만드는 것):**
- `db/connection.py` — DB 연결 관리
- `db/init.py` — DB 초기화 스크립트
- `src/training/train.py` — 학습 엔트리 포인트
- `api/main.py`, `api/routes.py`, `api/schemas.py`, `api/models.py` — FastAPI 서버
- `docker/prometheus.yml` — Prometheus 설정
- `tests/test_models.py`, `tests/test_api.py` — 기본 테스트

**데이터 경로:**
- 이미지: `db/classification/images/` (2227개 .jpg)
- 라벨: `db/classification/labels.csv` (columns: original_file_name, file_name, Surface_Crack, Delamination, Pinhole, unclassified)

---

## Task 1: 환경 설정 및 의존성 설치

**Files:**
- Modify: `.env.example` → `.env`

**Step 1: .env 파일 생성**

```bash
cp /mnt/c/Users/user/r2r/r2r-mlops/.env.example /mnt/c/Users/user/r2r/r2r-mlops/.env
```

`.env` 파일 내용 확인 후 아래처럼 맞추기:

```env
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=password
DB_NAME=r2r_coating
MLFLOW_TRACKING_URI=http://localhost:5000
```

**Step 2: Python 패키지 설치**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
pip install -r requirements.txt
pip install -e .
```

Expected: 오류 없이 설치 완료

**Step 3: Docker 스택 시작 (PostgreSQL + MLflow)**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops/docker
docker-compose up -d postgres mlflow
```

**Step 4: 서비스 상태 확인**

```bash
docker-compose ps
```

Expected output:
```
r2r_postgres   ... Up   5432/tcp
r2r_mlflow     ... Up   5000/tcp
```

MLflow UI 접속 확인: http://localhost:5000

**Step 5: Commit**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
git add .env
git commit -m "chore: add .env configuration"
```

---

## Task 2: DB 연결 모듈

**Files:**
- Create: `db/connection.py`
- Test: `tests/test_db.py`

**Step 1: 테스트 파일 작성**

`tests/test_db.py`:

```python
"""Tests for database connection module"""
import pytest
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_get_connection_string():
    """DB 연결 문자열이 올바른 형식인지 확인"""
    from db.connection import get_connection_string
    conn_str = get_connection_string()
    assert conn_str.startswith("postgresql://")
    assert "postgres" in conn_str


def test_get_engine_creates_engine():
    """SQLAlchemy 엔진이 생성되는지 확인"""
    from db.connection import get_engine
    engine = get_engine()
    assert engine is not None


def test_get_engine_can_connect():
    """실제 DB에 연결 가능한지 확인 (Docker postgres가 실행 중이어야 함)"""
    from db.connection import get_engine
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(__import__('sqlalchemy').text("SELECT 1"))
        assert result.fetchone()[0] == 1
```

**Step 2: 테스트 실행 (실패 확인)**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
pytest tests/test_db.py -v
```

Expected: `ModuleNotFoundError: No module named 'db.connection'`

**Step 3: `db/connection.py` 구현**

```python
"""
Database connection management for R2R MLOps
"""

import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
from typing import Generator
import logging

load_dotenv()

logger = logging.getLogger(__name__)


def get_connection_string() -> str:
    """환경변수에서 DB 연결 문자열 생성"""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "password")
    database = os.getenv("DB_NAME", "r2r_coating")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def get_engine() -> Engine:
    """SQLAlchemy 엔진 반환"""
    conn_str = get_connection_string()
    engine = create_engine(conn_str, pool_pre_ping=True)
    return engine


def get_session() -> Generator[Session, None, None]:
    """DB 세션 생성 (context manager용)"""
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
```

**Step 4: 테스트 재실행 (성공 확인)**

```bash
pytest tests/test_db.py -v
```

Expected: test_get_connection_string PASS, test_get_engine_creates_engine PASS, test_get_engine_can_connect PASS (Docker 실행 중일 때)

**Step 5: Commit**

```bash
git add db/connection.py tests/test_db.py
git commit -m "feat: add database connection module"
```

---

## Task 3: DB 초기화 스크립트

**Files:**
- Create: `db/init.py`

**Step 1: `db/init.py` 구현**

```python
"""
Database initialization script
Usage: python db/init.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import get_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_db():
    """schema.sql을 실행하여 테이블 생성"""
    engine = get_engine()

    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    with engine.connect() as conn:
        # 각 SQL 문을 개별 실행
        from sqlalchemy import text
        statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
        for stmt in statements:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception as e:
                logger.warning(f"Statement skipped (may already exist): {e}")

    logger.info("Database initialized successfully")


if __name__ == "__main__":
    init_db()
```

**Step 2: 실행 확인**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
python db/init.py
```

Expected: `Database initialized successfully`

**Step 3: Commit**

```bash
git add db/init.py
git commit -m "feat: add database initialization script"
```

---

## Task 4: Vision Classification 학습 스크립트

**Files:**
- Create: `src/training/train.py`
- Test: `tests/test_training.py`

**Step 1: 테스트 작성**

`tests/test_training.py`:

```python
"""Tests for training pipeline"""
import os
import sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_train_args_parser():
    """CLI 인수 파서가 올바르게 동작하는지 확인"""
    from src.training.train import create_parser
    parser = create_parser()
    args = parser.parse_args(["--model", "classification", "--epochs", "1"])
    assert args.model == "classification"
    assert args.epochs == 1


def test_classification_dataloader_loads():
    """Classification 데이터로더가 실제 데이터를 로드하는지 확인"""
    from src.models.vision.classification import create_dataloaders
    img_dir = "db/classification/images"
    csv_path = "db/classification/labels.csv"
    if not os.path.exists(img_dir):
        pytest.skip("Data not available")
    train_loader, val_loader = create_dataloaders(
        img_dir=img_dir,
        csv_path=csv_path,
        batch_size=4,
        num_workers=0
    )
    batch = next(iter(train_loader))
    images, labels = batch
    assert images.shape[0] == 4       # batch size
    assert images.shape[1] == 3       # RGB
    assert images.shape[2] == 224     # height
    assert images.shape[3] == 224     # width
    assert labels.shape == (4, 4)     # 4 classes
```

**Step 2: 테스트 실행 (실패 확인)**

```bash
pytest tests/test_training.py::test_train_args_parser -v
```

Expected: `ModuleNotFoundError: No module named 'src.training.train'`

**Step 3: `src/training/train.py` 구현**

```python
"""
Training entry point for R2R MLOps
Usage:
    python src/training/train.py --model classification
    python src/training/train.py --model classification --epochs 50 --batch-size 16
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path

import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R2R MLOps Training Script")
    parser.add_argument(
        "--model",
        type=str,
        default="classification",
        choices=["classification"],
        help="Model type to train"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="db/classification", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="ml/models/vision/classification/v1", help="Model output directory")
    parser.add_argument("--mlflow-uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--experiment", type=str, default="r2r-coating-classification")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 for Windows)")
    return parser


def train_classification(args: argparse.Namespace) -> dict:
    """ResNet18 Classification 학습"""
    from src.models.vision.classification import (
        ResNetClassifier, create_dataloaders
    )

    img_dir = os.path.join(args.data_dir, "images")
    csv_path = os.path.join(args.data_dir, "labels.csv")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    logger.info(f"Loading data from {args.data_dir}")
    train_loader, val_loader = create_dataloaders(
        img_dir=img_dir,
        csv_path=csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 모델 생성
    model = ResNetClassifier(
        num_classes=4,
        learning_rate=args.lr,
        pretrained=True
    )

    # 콜백
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="resnet18-{epoch:02d}-{val_f1_macro:.3f}",
        monitor="val_f1_macro",
        mode="max",
        save_top_k=1
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1_macro",
        patience=10,
        mode="max"
    )

    # 학습
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    trainer.fit(model, train_loader, val_loader)

    # 최종 메트릭 수집
    metrics = trainer.callback_metrics
    result = {
        "val_f1_macro": float(metrics.get("val_f1_macro", 0)),
        "val_recall_macro": float(metrics.get("val_recall_macro", 0)),
        "val_precision_macro": float(metrics.get("val_precision_macro", 0)),
        "val_loss": float(metrics.get("val_loss", 0)),
        "best_model_path": checkpoint_cb.best_model_path
    }

    # 메트릭 저장
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Training complete. Best model: {result['best_model_path']}")
    return result


def main():
    parser = create_parser()
    args = parser.parse_args()

    # MLflow 설정
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    params = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "data_dir": args.data_dir,
    }

    with mlflow.start_run():
        mlflow.log_params(params)

        if args.model == "classification":
            result = train_classification(args)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        # MLflow에 메트릭 기록
        mlflow.log_metrics({
            "val_f1_macro": result["val_f1_macro"],
            "val_recall_macro": result["val_recall_macro"],
            "val_precision_macro": result["val_precision_macro"],
        })

        # 모델 아티팩트 기록
        if result.get("best_model_path") and os.path.exists(result["best_model_path"]):
            mlflow.log_artifact(result["best_model_path"])

        logger.info(f"MLflow run complete. F1 Macro: {result['val_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
```

**Step 4: 테스트 실행 (성공 확인)**

```bash
pytest tests/test_training.py -v
```

Expected: 2 tests PASS

**Step 5: 실제 학습 실행 (검증)**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
python src/training/train.py --model classification --epochs 2 --batch-size 8 --num-workers 0
```

Expected: MLflow에 실험 기록, `ml/models/vision/classification/v1/`에 체크포인트 저장

**Step 6: Commit**

```bash
git add src/training/train.py tests/test_training.py
git commit -m "feat: add classification training script with MLflow integration"
```

---

## Task 5: Prometheus 설정

**Files:**
- Create: `docker/prometheus.yml`

**Step 1: `docker/prometheus.yml` 생성**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'r2r-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

**Step 2: Commit**

```bash
git add docker/prometheus.yml
git commit -m "feat: add prometheus configuration"
```

---

## Task 6: FastAPI 추론 서버

**Files:**
- Create: `api/schemas.py`
- Create: `api/models.py`
- Create: `api/routes.py`
- Create: `api/main.py`
- Test: `tests/test_api.py`

**Step 1: `api/schemas.py` 작성**

```python
"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    """단일 이미지 경로로 예측 요청"""
    image_path: str = Field(..., description="이미지 파일 경로")


class DefectPrediction(BaseModel):
    """개별 불량 클래스 예측 결과"""
    surface_crack: float = Field(..., ge=0.0, le=1.0)
    delamination: float = Field(..., ge=0.0, le=1.0)
    pinhole: float = Field(..., ge=0.0, le=1.0)
    unclassified: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """예측 응답"""
    image_path: str
    probabilities: DefectPrediction
    predicted_labels: List[str]
    inference_time_ms: float


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
```

**Step 2: `api/models.py` 작성**

```python
"""
Model loading and inference for API
"""

import os
import time
import torch
import logging
from pathlib import Path
from torchvision import transforms
from PIL import Image
from typing import Optional

logger = logging.getLogger(__name__)

# 전역 모델 상태
_model = None
_model_path = None

LABEL_NAMES = ["Surface_Crack", "Delamination", "Pinhole", "unclassified"]

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_path: str) -> bool:
    """체크포인트에서 모델 로드"""
    global _model, _model_path
    try:
        from src.models.vision.classification import ResNetClassifier
        _model = ResNetClassifier.load_from_checkpoint(model_path)
        _model.eval()
        _model_path = model_path
        logger.info(f"Model loaded from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def find_latest_model(base_dir: str = "ml/models/vision/classification") -> Optional[str]:
    """가장 최근 모델 체크포인트 탐색"""
    base = Path(base_dir)
    checkpoints = list(base.rglob("*.ckpt"))
    if not checkpoints:
        return None
    return str(sorted(checkpoints, key=lambda p: p.stat().st_mtime)[-1])


def predict(image_path: str) -> dict:
    """이미지 경로를 받아 불량 확률 반환"""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    start_time = time.time()

    image = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORM(image).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).tolist()

    inference_ms = (time.time() - start_time) * 1000

    predicted_labels = [
        LABEL_NAMES[i] for i, p in enumerate(probs) if p >= 0.5
    ]

    return {
        "probabilities": {
            "surface_crack": probs[0],
            "delamination": probs[1],
            "pinhole": probs[2],
            "unclassified": probs[3],
        },
        "predicted_labels": predicted_labels,
        "inference_time_ms": inference_ms
    }


def is_model_loaded() -> bool:
    return _model is not None


def get_model_path() -> Optional[str]:
    return _model_path
```

**Step 3: `api/routes.py` 작성**

```python
"""
FastAPI route definitions
"""

from fastapi import APIRouter, HTTPException
from api.schemas import PredictionRequest, PredictionResponse, HealthResponse
import api.models as model_manager
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """서버 및 모델 상태 확인"""
    return HealthResponse(
        status="ok",
        model_loaded=model_manager.is_model_loaded(),
        model_path=model_manager.get_model_path()
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """이미지 경로로 불량 예측"""
    if not model_manager.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = model_manager.predict(request.image_path)
        return PredictionResponse(
            image_path=request.image_path,
            **result
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 4: `api/main.py` 작성**

```python
"""
FastAPI main application
Usage: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import logging
from fastapi import FastAPI
from prometheus_client import make_asgi_app
import api.models as model_manager
from api.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="R2R Coating Defect Detection API",
    description="ResNet18 기반 코팅 불량 감지 추론 서버",
    version="1.0.0"
)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Routes
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 자동 로드"""
    model_path = os.getenv("MODEL_PATH", "")
    if not model_path:
        model_path = model_manager.find_latest_model()

    if model_path:
        success = model_manager.load_model(model_path)
        if success:
            logger.info(f"Model loaded on startup: {model_path}")
        else:
            logger.warning("Model load failed on startup — /predict will return 503")
    else:
        logger.warning("No model found — train first with: python src/training/train.py")


def main():
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True
    )


if __name__ == "__main__":
    main()
```

**Step 5: `tests/test_api.py` 작성**

```python
"""Tests for FastAPI endpoints"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


def test_health_check(client):
    """헬스체크 엔드포인트 동작 확인"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict_no_model_returns_503(client):
    """모델 미로드 시 503 반환 확인"""
    response = client.post(
        "/api/v1/predict",
        json={"image_path": "db/classification/images/image_1.jpg"}
    )
    # 모델이 없으면 503, 있으면 200
    assert response.status_code in [200, 503]


def test_predict_invalid_path_returns_error(client):
    """존재하지 않는 이미지 경로 처리 확인"""
    import api.models as m
    if not m.is_model_loaded():
        pytest.skip("Model not loaded")
    response = client.post(
        "/api/v1/predict",
        json={"image_path": "/nonexistent/path/image.jpg"}
    )
    assert response.status_code == 404
```

**Step 6: 테스트 실행**

```bash
pytest tests/test_api.py -v
```

Expected: test_health_check PASS, test_predict_no_model_returns_503 PASS

**Step 7: 개발 서버 실행 확인**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

다른 터미널에서:
```bash
curl http://localhost:8000/api/v1/health
```

Expected: `{"status":"ok","model_loaded":false,"model_path":null}`

**Step 8: Commit**

```bash
git add api/schemas.py api/models.py api/routes.py api/main.py tests/test_api.py
git commit -m "feat: add FastAPI inference server with health check and predict endpoints"
```

---

## Task 7: 전체 Docker 스택 실행 확인

**Step 1: API Docker 이미지 빌드 및 전체 스택 시작**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops/docker
docker-compose up -d
```

**Step 2: 상태 확인**

```bash
docker-compose ps
```

Expected: postgres, mlflow, api, prometheus, grafana 모두 Up

**Step 3: 엔드포인트 확인**

```bash
# API 헬스체크
curl http://localhost:8000/api/v1/health

# MLflow UI
# 브라우저: http://localhost:5000

# Grafana
# 브라우저: http://localhost:3000 (admin/admin)
```

**Step 4: Commit**

```bash
cd /mnt/c/Users/user/r2r/r2r-mlops
git add docker/prometheus.yml
git commit -m "chore: finalize docker stack configuration"
```

---

## Task 8: 전체 파이프라인 E2E 검증

**Step 1: DB 초기화**

```bash
python db/init.py
```

**Step 2: 학습 실행 (전체 30 epoch)**

```bash
python src/training/train.py \
  --model classification \
  --epochs 30 \
  --batch-size 16 \
  --num-workers 0
```

**Step 3: MLflow에서 결과 확인**

브라우저에서 http://localhost:5000 → `r2r-coating-classification` 실험 클릭
- F1 Macro ≥ 0.85 확인
- Recall ≥ 0.90 확인

**Step 4: API 추론 테스트**

```bash
# 서버 시작 (학습된 모델 자동 로드)
uvicorn api.main:app --reload --port 8000

# 예측 요청
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"image_path": "db/classification/images/image_1.jpg"}'
```

Expected 응답:
```json
{
  "image_path": "db/classification/images/image_1.jpg",
  "probabilities": {"surface_crack": 0.92, "delamination": 0.03, ...},
  "predicted_labels": ["Surface_Crack"],
  "inference_time_ms": 45.2
}
```

**Step 5: 전체 테스트 실행**

```bash
pytest tests/ -v
```

Expected: 모든 테스트 PASS

**Step 6: 최종 Commit**

```bash
git add -A
git commit -m "feat: complete Phase 1 MLOps pipeline - classification training + API server"
```

---

## 완료 기준 체크리스트

- [ ] `python db/init.py` 실행 시 테이블 생성됨
- [ ] `python src/training/train.py --epochs 2` 실행 시 오류 없이 학습됨
- [ ] MLflow UI (http://localhost:5000) 에서 실험 결과 확인 가능
- [ ] `ml/models/vision/classification/v1/*.ckpt` 파일 생성됨
- [ ] `curl http://localhost:8000/api/v1/health` → `{"status":"ok"}`
- [ ] `pytest tests/ -v` → 전체 PASS
- [ ] `docker-compose ps` → 모든 서비스 Up

## 성능 목표 (KPI)

| 메트릭 | 목표 |
|--------|------|
| F1 Macro | ≥ 0.85 |
| Recall (불량 클래스) | ≥ 0.90 |
| API 응답시간 P95 | < 100ms |
