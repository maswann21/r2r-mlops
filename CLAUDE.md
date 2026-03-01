# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

R2R Coating Defect Detection and Auto-Optimization MLOps System - a comprehensive machine learning platform for detecting coating defects in Roll-to-Roll manufacturing processes.

**Core Concept**: Dual-model architecture combining Vision models (image-based defect detection) with Sensor models (time-series prediction), connected through a Pseudo Labeling pipeline for closed-loop optimization.

## Architecture

### Dual-Model System

This project implements two parallel ML pipelines:

1. **Vision Pipeline** (Image → Defect Classification)
   - Classification: ResNet18-based multi-label classifier for Surface_Crack, Delamination, Pinhole, unclassified
   - Output: Defect labels with confidence scores

2. **Sensor Pipeline** (Time-series → Defect Prediction)
   - Baseline: XGBoost/RandomForest/LightGBM on engineered features
   - Deep Learning: LSTM/GRU/1D-CNN for sequential sensor data
   - Ensemble: Voting classifier combining multiple models
   - Output: Pre-failure predictions based on sensor readings

### Key Architectural Pattern: Pseudo Labeling

Vision model predictions are stored as "pseudo labels" in the database, which then train the Sensor model to predict defects from sensor data alone. This enables:
- Closed-loop optimization: Adjust process parameters before defects occur
- Automated data labeling at scale
- Lead time prediction (5-10 minutes before defect occurrence)

### Data Flow

```
R2R Process → Images + Sensor Data
    ↓
Vision Model → Defect Labels + Confidence
    ↓
PostgreSQL (Pseudo Labels)
    ↓
Sensor Model Training → Pre-failure Prediction
    ↓
Closed-loop Control → Process Parameter Adjustment
```

## Development Commands

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env with your database credentials
```

### Database

```bash
# Initialize database schema
python db/init.py

# Start PostgreSQL with Docker
cd docker && docker-compose up -d postgres

# View database logs
docker-compose logs -f postgres
```

### MLOps Infrastructure

```bash
# Start all services (PostgreSQL, MLflow, API, Prometheus, Grafana)
cd docker && docker-compose up -d

# Start only MLflow tracking server
docker-compose up -d mlflow
# Access at http://localhost:5000

# Start API server (development)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# View monitoring dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Model Training

```bash
# Train sensor models
python src/training/train.py --model baseline  # XGBoost/RF/LightGBM
python src/training/train.py --model lstm      # LSTM model
python src/training/train.py --model ensemble  # Ensemble model

# Train vision models (typically in Jupyter notebooks)
jupyter lab notebooks/vision/02_classification_train.ipynb
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_models.py
pytest tests/test_api.py

# Run with coverage
pytest --cov=src tests/
```

## Model Storage Convention

### Directory Structure

Models are versioned and stored in `ml/models/` with the following structure:

```
ml/models/
├── vision/
│   └── classification/v1/    # ResNet18 weights, config, metrics
└── sensor/
    ├── baseline/v1/           # XGBoost/RF (.pkl files)
    ├── lstm/v1/               # PyTorch LSTM (.pth files)
    ├── ensemble/v1/           # Voting ensemble
    └── production/            # Production-deployed models only
```

### Saving Models

**Vision Models (PyTorch)**:
```python
# Save checkpoint with PyTorch Lightning
trainer.save_checkpoint("ml/models/vision/classification/v1/resnet18.pth")

# Save with metadata
torch.save(model.state_dict(), "ml/models/vision/classification/v1/resnet18.pth")
# Also save: config.yaml, metrics.json
```

**Sensor Models (Scikit-learn/XGBoost)**:
```python
from src.models.sensor import BaselineModel

model = BaselineModel(model_type="xgboost")
model.fit(X_train, y_train, X_val, y_val)
model.save("ml/models/sensor/baseline/v1/xgboost")
# Creates: xgboost_model.pkl, xgboost_scaler.pkl
```

**Important**: Always save corresponding `config.yaml` (hyperparameters) and `metrics.json` (performance metrics) alongside model weights for reproducibility.

### Loading Models

```python
# Vision model
from src.models.vision import ResNetClassifier
model = ResNetClassifier.load_from_checkpoint("ml/models/vision/classification/latest/resnet18.pth")

# Sensor model
from src.models.sensor import BaselineModel
model = BaselineModel(model_type="xgboost")
model.load("ml/models/sensor/baseline/v1/xgboost")
```

## Database Schema

### Core Tables

- `raw_sensor_data`: Original sensor readings (temperature, rotation_speed, torque, tool_wear, etc.)
- `nasa_bearing_sensor`: Vibration data from NASA bearing dataset
- `processed_data`: Feature-engineered data ready for model training
- `models`: Model metadata registry (name, version, path, metrics, status)
- `predictions`: Inference results with confidence scores and timing
- `experiments`: MLflow experiment tracking integration
- `data_quality_logs`: Data drift detection and quality monitoring
- `model_performance_logs`: Daily model performance metrics

### Querying Pattern

Vision model predictions are stored in `predictions` table and joined with `raw_sensor_data` by timestamp to create training data for sensor models.

## Data Organization

### Vision Data

```
db/classification/
  ├── images/          # PNG/JPG coating surface images
  └── labels.csv       # Multi-label CSV: file_name, Surface_Crack, Delamination, Pinhole, unclassified
```

### Sensor Data

```
data/raw/
  ├── machine_failure_prediction.csv  # Kaggle dataset
  └── nasa_bearing/*.csv              # NASA bearing vibration data

data/processed/
  ├── train.csv, val.csv, test.csv   # Split datasets
  └── preprocessing_log.json

data/features/
  └── features_v{N}.pkl               # Feature Store versions
```

## Key Technical Decisions

### Multi-label Classification
Vision models use `BCEWithLogitsLoss` (Binary Cross-Entropy) since multiple defect types can occur simultaneously. Predictions use sigmoid activation with 0.5 threshold per class.

### Feature Engineering for Sensors
Time-series sensor data is transformed into features using rolling windows (10, 30, 60 seconds) with statistical aggregations: mean, std, min, max, median. NASA bearing data adds RMS, peak, kurtosis, skewness from vibration signals.

### Class Imbalance Handling
- SMOTE for synthetic minority oversampling
- Class weights in loss functions
- Stratified splits to preserve class distribution

### Model Evaluation Priority
**Recall > Precision** for defect detection - missing a defect (False Negative) is more costly than a false alarm. Target: F1 ≥ 0.85, Recall ≥ 0.90.

### Experiment Tracking
All training runs are logged to MLflow with:
- Hyperparameters (`mlflow.log_params`)
- Metrics (`mlflow.log_metrics`)
- Model artifacts (`mlflow.pytorch.log_model` or `mlflow.sklearn.log_model`)
- Config files (`mlflow.log_artifact`)

Access MLflow UI at http://localhost:5000 after starting docker-compose.

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| F1-Score (Macro) | ≥ 0.85 | Primary model performance metric |
| Recall (Defect Classes) | ≥ 0.90 | Critical: minimize False Negatives |
| API Response Time (P95) | < 100ms | Inference latency |
| Data Pipeline Success Rate | ≥ 99.5% | ETL reliability |

## Common Workflows

### 1. Training a New Vision Model

```bash
# Start Jupyter
jupyter lab notebooks/vision/

# Open classification notebook
# notebooks/vision/02_classification_train.ipynb

# After training, save to versioned directory
# ml/models/vision/classification/v{N}/

# Log to MLflow within notebook
```

### 2. Training a Sensor Model

```bash
# Ensure data is processed
python pipeline/etl.py

# Train baseline model
python src/training/train.py --model xgboost

# Models auto-saved to ml/models/sensor/baseline/v{N}/
# Check MLflow for experiment results
```

### 3. Deploying to Production

```bash
# Copy best model to production directory
cp -r ml/models/vision/classification/v2/ ml/models/vision/production/classification/

# Update API to load from production path
# api/models.py references config.yaml model_paths.production

# Restart API
docker-compose restart api
```

### 4. Monitoring Model Drift

```bash
# Query model_performance_logs table
psql -h localhost -U postgres -d r2r_coating

SELECT model_id, metric_date, f1_score, recall
FROM model_performance_logs
WHERE model_id = 1
ORDER BY metric_date DESC
LIMIT 30;

# Check Grafana dashboards for visualizations
# http://localhost:3000
```

## Dependencies and Tech Stack

- **ML/DL**: PyTorch, PyTorch Lightning, scikit-learn, XGBoost, LightGBM
- **Data**: Pandas, NumPy, Pillow (images)
- **MLOps**: MLflow (experiment tracking), DVC (data versioning)
- **API**: FastAPI, Uvicorn, Pydantic
- **Database**: PostgreSQL (via psycopg2-binary, SQLAlchemy)
- **Monitoring**: Prometheus, Grafana
- **Deployment**: Docker, Docker Compose
- **Hyperparameter Tuning**: Optuna

## Configuration Files

- `config/config.yaml`: Project-wide configuration (DB, paths, training params)
- `config/params.yaml`: Model hyperparameters by model type
- `.env`: Environment variables (DB credentials, API keys) - **never commit**

## Notes for Future Development

### Phase 2 Roadmap (Beyond Current Scope)

- Integration of all three vision models (Classification + Detection + Segmentation)
- Multimodal fusion (Vision + Sensor combined predictions)
- Closed-loop PID control implementation
- Automated retraining triggers based on data/model drift detection
- A/B testing framework for model comparison in production
- Shadow mode deployment for new model validation

### Known Limitations

- Current phase focuses on sensor models; vision models are in notebooks
- Closed-loop control logic is designed but not yet implemented
- Automated retraining pipeline requires manual trigger
- No GPU optimization (TensorRT) for edge deployment yet

## Troubleshooting

**MLflow UI not accessible**: Ensure `docker-compose up -d mlflow` is running and PostgreSQL is healthy.

**Model loading fails**: Check that model version path exists and includes both weights (.pth/.pkl) and scaler (.pkl for ML models).

**Database connection error**: Verify `.env` file has correct credentials and `docker-compose up -d postgres` is running.

**Import errors in notebooks**: Ensure you ran `pip install -e .` to install the project package in editable mode.
