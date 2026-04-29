# 🚀 End-to-End Fraud Detection Data + ML Pipeline (Dockerized v1)

## 📌 Overview

This project builds a **production-like batch fraud data platform** using a Data Engineering–first approach.

The system implements a **full pipeline from raw data → final fraud predictions**, designed with:

* Medallion Architecture (Bronze → Silver → Gold → Scoring → Final)
* PySpark for scalable batch processing
* Docker as the official runtime
* Strict phase separation and data contracts


## 🏗️ Architecture

```
Raw CSV
    ↓
[Bronze] ingestion (CSV → Parquet)
    ↓
[Silver] join + clean
    ↓
[Gold] feature table (ML-ready)
    ↓
[Scoring] model + predictions
    ↓
[Final] business-ready dataset
```


## 🎯 What This Project Demonstrates

* Production-style **data pipeline design**
* Clean **phase ownership & separation**
* End-to-end **batch ML workflow**
* **Idempotent pipeline execution**
* Real-world dataset handling (IEEE-CIS)


## 🧱 Tech Stack

* Python 3.10
* PySpark 3.5.1 
* Docker
* Parquet (data storage)
* YAML (config-driven pipeline)


## 📂 Project Structure

```
configs/
  config.yaml

data/
  raw/          → input CSV
  bronze/       → raw → parquet
  silver/       → joined + cleaned
  gold/         → feature datasets
  artifacts/    → model + predictions
  final/        → final output dataset

pipelines/
  bronze/run_bronze.py
  silver/run_silver.py
  gold/run_gold.py
  scoring/run_scoring.py
  final/run_final.py

run_pipeline.py

requirements.txt
Dockerfile
```


## ⚙️ How to Run

### 1. Build Docker image

```bash
docker build -t fraud-pipeline:latest .
```


### 2. Run full pipeline (recommended)

```powershell
docker run --rm --name fraud-pipeline-full `
  -v ${PWD}:/app -w /app `
  fraud-pipeline:latest python run_pipeline.py
```

### 3. Run individual phases (optional)

Example:

```powershell
docker run --rm -v ${PWD}:/app -w /app `
  fraud-pipeline:latest python pipelines/bronze/run_bronze.py
```

## 📊 Output

Final dataset:

```
data/final/scored_transactions/
```

Schema:

```text
TransactionID
fraud_probability
```

This is the **official data product** of the pipeline.


## ✅ Current Status (v1)

* ✅ Bronze ingestion (CSV → Parquet)
* ✅ Silver join + cleaning
* ✅ Gold feature engineering (ML-ready table)
* ✅ Scoring (Logistic Regression baseline)
* ✅ Final output dataset (clean contract)
* ✅ End-to-end pipeline execution (one command)


## ⚠️ Notes

* Spark memory warnings are expected in Docker and do not affect correctness
* Pipeline is strictly batch (no streaming)
* Each phase is independent and idempotent


## ❌ Not Included in v1

* Real-time processing (Kafka, streaming)
* Advanced ML models (XGBoost, LightGBM)
* Model evaluation / monitoring
* Orchestration tools (Airflow)
* Cloud deployment


## 🔜 Next Steps (Post v1)

* Add **model evaluation layer (AUC, Recall, PR-AUC)**
* Improve feature engineering
* Introduce better models
* Add API serving layer (FastAPI)
* Add orchestration (Airflow)


## 📎 Dataset

IEEE-CIS Fraud Detection (Kaggle)

## 💡 Key Design Principles

* One phase = one responsibility
* No logic overlap between phases
* Data contracts between layers
* Final dataset = only external-facing output