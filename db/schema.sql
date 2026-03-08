-- R2R Coating MLOps Database Schema

-- Raw Data Layer
CREATE TABLE IF NOT EXISTS raw_sensor_data (
    id SERIAL PRIMARY KEY,
    footfall INT,
    temp_mode INT,
    aq INT,
    uss INT,
    cs INT,
    voc INT,
    rp INT,
    ip INT,
    temperature INT,
    fail INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NASA Bearing Data
CREATE TABLE IF NOT EXISTS nasa_bearing_sensor (
    id SERIAL PRIMARY KEY,
    bearing_id INT NOT NULL,
    measurement_num INT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    vibration_data FLOAT[],
    rms FLOAT,
    peak FLOAT,
    kurtosis FLOAT,
    skewness FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(bearing_id, measurement_num)
);

-- Processed Data Layer
CREATE TABLE IF NOT EXISTS processed_data (
    id SERIAL PRIMARY KEY,
    raw_data_id INT REFERENCES raw_sensor_data(id),
    timestamp TIMESTAMP NOT NULL,
    feature_vector FLOAT[],
    defect_type VARCHAR(50) NOT NULL,
    feature_version VARCHAR(20),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Metadata Layer
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_path VARCHAR(255),
    framework VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metrics_f1_macro FLOAT,
    metrics_recall FLOAT,
    metrics_precision FLOAT,
    status VARCHAR(20) DEFAULT 'development',  -- development, testing, production, deprecated
    UNIQUE(model_name, version)
);

-- Model Predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    model_id INT REFERENCES models(id),
    raw_data_id INT REFERENCES raw_sensor_data(id),
    prediction_defect_type VARCHAR(50),
    confidence_score FLOAT,
    inference_time_ms FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_correct BOOLEAN  -- For actual vs predicted
);

-- Experiment Tracking (MLflow integration)
CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255) NOT NULL,
    run_id VARCHAR(100) NOT NULL UNIQUE,
    parameters JSONB,
    metrics JSONB,
    artifacts_path VARCHAR(255),
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(20),  -- RUNNING, FINISHED, FAILED
    UNIQUE(experiment_name, run_id)
);

-- Data Quality Monitoring
CREATE TABLE IF NOT EXISTS data_quality_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    missing_count INT,
    outlier_count INT,
    data_drift_detected BOOLEAN,
    data_drift_score FLOAT,
    notes TEXT,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Performance Monitoring
CREATE TABLE IF NOT EXISTS model_performance_logs (
    id SERIAL PRIMARY KEY,
    model_id INT REFERENCES models(id),
    metric_date DATE NOT NULL,
    f1_score FLOAT,
    recall FLOAT,
    precision FLOAT,
    accuracy FLOAT,
    false_negative_count INT,
    false_positive_count INT,
    inference_time_p95_ms FLOAT,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, metric_date)
);

-- Indexes for performance
CREATE INDEX idx_predictions_model_id ON predictions(model_id);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);
CREATE INDEX idx_experiments_run_id ON experiments(run_id);
CREATE INDEX idx_model_performance_model_id ON model_performance_logs(model_id);
CREATE INDEX idx_model_performance_date ON model_performance_logs(metric_date);
