# Database Schema 설계서

## 개요
R2R Coating MLOps 시스템의 PostgreSQL 데이터베이스 스키마

---

## 테이블 구조

### 1. raw_sensor_data (원본 센서 데이터)
```
raw_sensor_data
├── id (PK)
├── footfall, temp_mode, aq, uss, cs
├── voc, rp, ip
├── temperature
├── fail
└── created_at
```

**목적**: IoT 센서 데이터 저장 (data/raw/data.csv 기반)

---

### 2. nasa_bearing_sensor (NASA 베어링 데이터)
```
nasa_bearing_sensor
├── id (PK)
├── bearing_id
├── measurement_num
├── timestamp
├── vibration_data (FLOAT[])
├── rms, peak, kurtosis, skewness
└── created_at
```

**목적**: NASA Bearing 진동 데이터 저장

---

### 3. processed_data (전처리 데이터)
```
processed_data
├── id (PK)
├── raw_data_id (FK)
├── feature_vector (FLOAT[])
├── defect_type
├── feature_version
└── processed_at
```

**목적**: Feature Engineering 후 처리된 데이터 저장

---

### 4. models (모델 메타데이터)
```
models
├── id (PK)
├── model_name
├── model_type (RandomForest, XGBoost, LSTM, ...)
├── version
├── model_path
├── framework (sklearn, xgboost, pytorch, ...)
├── metrics_f1_macro, recall, precision
├── status (development, testing, production, deprecated)
└── created_at
```

**목적**: 모델 버전 관리 및 메타데이터

---

### 5. predictions (추론 결과)
```
predictions
├── id (PK)
├── model_id (FK)
├── raw_data_id (FK)
├── prediction_defect_type
├── confidence_score
├── inference_time_ms
├── is_correct (실제값과 비교)
└── created_at
```

**목적**: 모델 추론 결과 로깅

---

### 6. experiments (MLflow 실험 추적)
```
experiments
├── id (PK)
├── experiment_name
├── run_id (UNIQUE)
├── parameters (JSONB)
├── metrics (JSONB)
├── artifacts_path
├── start_time, end_time
├── status
└── 제약조건
```

**목적**: 머신러닝 실험 추적

---

### 7. data_quality_logs (데이터 품질 모니터링)
```
data_quality_logs
├── id (PK)
├── timestamp
├── missing_count
├── outlier_count
├── data_drift_detected (BOOLEAN)
├── data_drift_score
├── notes
└── logged_at
```

**목적**: 데이터 품질 추적 및 data drift 감지

---

### 8. model_performance_logs (모델 성능 모니터링)
```
model_performance_logs
├── id (PK)
├── model_id (FK)
├── metric_date (DATE)
├── f1_score, recall, precision, accuracy
├── false_negative_count, false_positive_count
├── inference_time_p95_ms
└── logged_at
```

**목적**: 모델 성능 모니터링

---

## 데이터 플로우

```
1. Raw Data Loading
   └─→ raw_sensor_data, nasa_bearing_sensor

2. Data Processing
   └─→ processed_data

3. Feature Engineering
   └─→ features stored in processed_data

4. Model Training
   └─→ experiments (MLflow)
   └─→ models

5. Model Inference
   └─→ predictions

6. Monitoring
   ├─→ data_quality_logs
   └─→ model_performance_logs
```

---

## 인덱스 전략

```sql
-- 조회 성능 최적화
idx_predictions_model_id          -- 모델별 예측 조회
idx_predictions_created_at        -- 시간대별 예측 조회
idx_experiments_run_id            -- 실험 조회
idx_model_performance_model_id    -- 모델별 성능 조회
idx_model_performance_date        -- 날짜별 성능 조회
```

---

## 파티셔닝 (대규모 데이터용)

향후 대량의 데이터 처리를 위해 시간 기반 파티셔닝 적용 예정:

```sql
-- raw_sensor_data 파티셔닝 (향후 대량 데이터 시 적용)
-- 현재 timestamp 컬럼 없으므로 id 기반 range 파티셔닝 고려
PARTITION BY RANGE (id)
```

---

## 백업 & 복구 전략

- **자동 백업**: 일일 백업 (pg_dump)
- **복구 테스트**: 주간 복구 테스트
- **보관 정책**: 최근 3개월 데이터

---

## 용량 계획

| 테이블 | 예상 행 수/월 | 크기 |
|-------|-------------|------|
| raw_sensor_data | 1M | ~200MB |
| processed_data | 1M | ~150MB |
| predictions | 5M | ~500MB |
| experiments | 100 | ~10KB |
| others | - | ~50MB |
| **Total** | - | **~900MB** |
