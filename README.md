# 🚀 End-to-End Fraud Detection Data + ML Pipeline (Dockerized)

## 📌 Overview
This project builds a **production-like fraud detection pipeline** using a modern data engineering + ML workflow.

The system is designed with:
- **Docker (1 image, multiple containers)**
- **Medallion Architecture (Bronze → Silver → Gold)**
- **PySpark for scalable data processing**
- **Modular pipeline (ingestion → transform → feature → train)**


## 🏗️ Architecture

```
Raw CSV
    ↓
[Bronze] ingestion
    ↓
[Silver] join + clean
    ↓
[Gold] feature dataset
    ↓
[Model] training
```


## 🧱 Tech Stack
- Python 3.10
- PySpark
- Docker
- Parquet (data storage)
- YAML config


## 📂 Project Structure

```
configs/        → paths, configs
data/
  raw/          → original CSV
  bronze/       → raw → parquet
  silver/       → joined datasets
  gold/         → feature datasets
  artifacts/    → metadata, model output

ingestion/      → load raw data
pipeline/       → bronze → silver
features/       → feature engineering
models/         → training

Dockerfile
docker-compose.yml
requirements.txt
```


## ⚙️ Pipeline Execution (Docker)

### 1. Build image
```bash
docker build -t fraud-pipeline:latest .
```

### 2. Bronze — Ingestion

```bash
docker run --rm --name fraud-ingestion \
  -v ${PWD}:/app -w /app \
  fraud-pipeline python ingestion/load_ieee_cis.py
```

### 3. Silver — Transform

```bash
docker run --rm --name fraud-transform \
  -v ${PWD}:/app -w /app \
  fraud-pipeline python pipeline/bronze_to_silver.py
```

### 4. Gold — Feature

```bash
docker run --rm --name fraud-feature \
  -v ${PWD}:/app -w /app \
  fraud-pipeline python features/feature_builder.py
```

### 5. Model Training

```bash
docker run --rm --name fraud-train \
  -v ${PWD}:/app -w /app \
  fraud-pipeline python models/train.py
```



## 📊 Current Progress

✅ Dockerized pipeline (single image, multi-container)

✅ Basic Bronze ingestion (CSV → Parquet)

✅ Basic Silver transformation (join transaction + identity)

✅ Basic Feature engineering (baseline numeric features)

⏳ Model training (Logistic Regression baseline)


## 🎯 Next Steps

* Improve EDA, Feature Engineering
* Train advanced models (LightGBM / XGBoost)
* Add model evaluation metrics (AUC-PR, Recall)
* Build API serving (FastAPI)
* Add monitoring & orchestration


## 💡 Key Learnings

* Spark requires proper memory tuning inside Docker
* Separation of pipeline steps improves debugging
* One-image multi-container design = production-like workflow
* Medallion architecture simplifies data lifecycle


## 📎 Dataset

IEEE-CIS Fraud Detection (Kaggle)