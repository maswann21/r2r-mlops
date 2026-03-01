# Project Restructure Design

**Date:** 2026-03-01
**Goal:** Vision Classification에 집중하도록 프로젝트 구조를 단순화. 사용하지 않는 코드/폴더 제거, 모든 이미지 데이터를 `data/`로 통합.

---

## 배경

Detection/Segmentation Vision 모델을 제거하고 Classification만 남기기로 결정. 그러나 이미지 데이터(db/detection, db/segmentation)는 Phase 2를 위해 보존. 현재 프로젝트에는 빈 stub 모듈, 사용하지 않는 sensor 코드, 루트 레벨 잡파일들이 혼재해 구조가 불명확한 상태.

---

## 설계 결정

### 1. 데이터 경로 통합: `db/` → `data/`

모든 이미지 데이터를 `data/`로 이동. `db/`는 스키마 SQL만 보관.

**Before:**
```
db/
├── classification/images/ + labels.csv
├── detection/images/ + labels/
├── segmentation/images/ + masks/
└── schema.sql
```

**After:**
```
data/
├── classification/
│   ├── images/        (1,010개 .jpg)
│   └── labels.csv
├── detection/
│   ├── images/        (2,227개)
│   └── labels/        (581개 YOLO .txt)
└── segmentation/
    ├── images/        (2,227개)
    └── masks/         (2,227개)

db/
└── schema.sql
```

**이유:** `db/`는 데이터베이스 스키마 파일의 위치. 이미지 파일과 혼재하면 혼란. `data/`는 ML 데이터의 표준적인 위치.

---

### 2. 소스코드 정리: `src/`

사용하지 않는 빈 모듈 제거, sensor 코드 제거.

**Before:**
```
src/
├── __init__.py
├── data/__init__.py        ← 빈 stub
├── features/__init__.py    ← 빈 stub
├── models/
│   ├── __init__.py
│   ├── sensor/             ← Phase 2 대상, 현재 미사용
│   │   ├── baseline.py
│   │   ├── lstm.py
│   │   ├── gru.py
│   │   └── cnn1d.py
│   └── vision/
│       ├── __init__.py
│       └── classification.py
├── training/__init__.py    ← 빈 stub
└── utils/__init__.py       ← 빈 stub
```

**After:**
```
src/
└── models/
    └── vision/
        ├── __init__.py          # ResNetClassifier export
        └── classification.py   # ResNet18 구현 (현재 코드 그대로)
```

**이유:** 빈 stub은 존재만으로 구조가 복잡해 보이게 함. Sensor 코드는 Phase 2에서 별도 브랜치로 재추가.

---

### 3. 모델 저장소 정리: `ml/`

빈 sensor 폴더 제거, 루트 노트북 이동.

**Before:**
```
ml/
├── .gitkeep
├── Machine_Failure_Prediction_using_Sensor_data.ipynb
└── models/
    ├── sensor/
    │   ├── baseline/.gitkeep
    │   ├── ensemble/.gitkeep
    │   ├── lstm/.gitkeep
    │   └── production/.gitkeep
    └── vision/
        └── classification/
            └── resnet18_defect.pth  (43MB)
```

**After:**
```
ml/
└── models/
    └── vision/
        └── classification/
            └── resnet18_defect.pth  (43MB, 유지)

notebooks/
└── sensor/
    └── Machine_Failure_Prediction_using_Sensor_data.ipynb  (이동)
```

---

### 4. 루트 레벨 정리

| 항목 | 처리 |
|------|------|
| `data.csv` | `data/raw/data.csv`로 이동 |
| `WBS.md` | `docs/WBS.md`로 이동 |
| `R2R Coating 불량 감지 및 자동 최적화 MLOps 프로젝트설명.md` | `docs/project-overview.md`로 이동 (파일명 정리) |

---

## 최종 구조

```
r2r-mlops/
├── api/                          (미구현, 유지)
├── config/
│   ├── config.yaml
│   └── params.yaml
├── data/
│   ├── classification/
│   │   ├── images/
│   │   └── labels.csv
│   ├── detection/
│   │   ├── images/
│   │   └── labels/
│   ├── raw/
│   │   └── data.csv
│   └── segmentation/
│       ├── images/
│       └── masks/
├── db/
│   └── schema.sql
├── docker/
│   ├── Dockerfile.api
│   └── docker-compose.yml
├── docs/
│   ├── plans/
│   ├── PROJECT_STRUCTURE.md
│   ├── WBS.md
│   ├── db_schema.md
│   ├── model_saving_guide.md
│   └── project-overview.md
├── ml/
│   └── models/
│       └── vision/
│           └── classification/
│               └── resnet18_defect.pth
├── notebooks/
│   ├── sensor/
│   │   └── Machine_Failure_Prediction_using_Sensor_data.ipynb
│   └── vision/
├── src/
│   └── models/
│       └── vision/
│           ├── __init__.py
│           └── classification.py
├── .env.example
├── .gitignore
├── CLAUDE.md
├── README.md
├── requirements.txt
└── setup.py
```

---

## 변경 범위 요약

| 카테고리 | 변경 수 | 내용 |
|----------|---------|------|
| 이동 | 5건 | db/classification → data/, db/detection → data/, db/segmentation → data/, data.csv → data/raw/, ml/*.ipynb → notebooks/sensor/, WBS.md/설명.md → docs/ |
| 삭제 | 8건 | src/data, src/features, src/utils, src/training, src/models/sensor/, ml/models/sensor/, ml/.gitkeep, src/__init__.py |
| 유지 | - | classification.py, resnet18_defect.pth, schema.sql, config/, docker/, api/ |

---

## 영향 범위

- `config/config.yaml`의 데이터 경로 업데이트 필요 (`./db/classification` → `./data/classification`)
- `CLAUDE.md` 데이터 경로 참조 업데이트
- `docs/PROJECT_STRUCTURE.md` 업데이트
- `setup.py` packages 리스트 업데이트 (sensor 패키지 제거)
