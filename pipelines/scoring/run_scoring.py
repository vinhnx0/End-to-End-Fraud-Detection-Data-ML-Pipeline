# pipelines/scoring/run_scoring.py

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_config() -> dict:
    config_path = Path(os.getenv("PIPELINE_CONFIG", DEFAULT_CONFIG_PATH))

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_spark(app_name: str = "fraud_scoring") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing feature metadata: {metadata_path}")

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def main() -> None:
    setup_logging()

    config = load_config()

    gold_root = PROJECT_ROOT / config["data"]["gold_root"]
    artifacts_root = PROJECT_ROOT / config["data"]["artifacts_root"]

    train_features_path = gold_root / "train_features"
    test_features_path = gold_root / "test_features"
    metadata_path = artifacts_root / "feature_metadata.json"

    model_output_path = artifacts_root / config["scoring"]["model_output_dir"]
    predictions_output_path = artifacts_root / config["scoring"]["predictions_output_dir"]

    metadata = read_metadata(metadata_path)

    id_column = metadata["id_column"]
    label_column = metadata["label_column"]
    feature_columns = metadata["feature_columns"]

    spark = build_spark()

    try:
        train_df = spark.read.parquet(str(train_features_path))
        test_df = spark.read.parquet(str(test_features_path))

        logging.info("Loaded train features | rows=%s", train_df.count())
        logging.info("Loaded test features | rows=%s", test_df.count())
        logging.info("Feature count=%s", len(feature_columns))

        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="features",
        )

        logistic_regression = LogisticRegression(
            featuresCol="features",
            labelCol=label_column,
            probabilityCol="probability",
            predictionCol="prediction",
            maxIter=20,
        )

        pipeline = Pipeline(stages=[assembler, logistic_regression])

        logging.info("Training baseline Logistic Regression model")
        model = pipeline.fit(train_df)

        logging.info("Generating fraud probability predictions")
        scored_df = model.transform(test_df)

        predictions_df = scored_df.select(
            F.col(id_column).alias("TransactionID"),
            vector_to_array(F.col("probability"))[1].alias("fraud_probability"),
        )

        model.write().overwrite().save(str(model_output_path))

        (
            predictions_df
            .write
            .mode("overwrite")
            .parquet(str(predictions_output_path))
        )

        logging.info("Model saved to: %s", model_output_path)
        logging.info("Predictions saved to: %s", predictions_output_path)
        logging.info("Fraud scoring completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Fraud scoring failed: %s", exc)
        sys.exit(1)