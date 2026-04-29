# pipelines/final/run.py

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
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


def build_spark(app_name: str = "final_output_layer") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "0.0.0.0")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_parquet(spark: SparkSession, path: Path) -> DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet input: {path}")

    return spark.read.parquet(str(path))


def write_parquet(df: DataFrame, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    df.write.mode("overwrite").parquet(str(path))


def validate_predictions(df: DataFrame) -> None:
    required_columns = {"TransactionID", "fraud_probability"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    null_transaction_count = df.filter(F.col("TransactionID").isNull()).count()
    if null_transaction_count > 0:
        raise ValueError(
            f"Final output contains null TransactionID values: {null_transaction_count}"
        )

    invalid_probability_count = df.filter(
        F.col("fraud_probability").isNull()
        | (F.col("fraud_probability") < 0)
        | (F.col("fraud_probability") > 1)
    ).count()

    if invalid_probability_count > 0:
        raise ValueError(
            f"Final output contains invalid fraud_probability values: "
            f"{invalid_probability_count}"
        )


def build_final_output(predictions_df: DataFrame) -> DataFrame:
    return predictions_df.select(
        F.col("TransactionID"),
        F.col("fraud_probability").cast("double").alias("fraud_probability"),
    )


def main() -> None:
    setup_logging()

    config = load_config()

    artifacts_root = PROJECT_ROOT / config["data"]["artifacts_root"]

    final_root = PROJECT_ROOT / config["data"].get("final_root", "data/final")

    predictions_input_path = (
        artifacts_root / config["scoring"]["predictions_output_dir"]
    )

    final_output_path = final_root / "scored_transactions"

    spark = build_spark()

    try:
        logging.info("Reading scoring predictions from: %s", predictions_input_path)

        predictions_df = read_parquet(spark, predictions_input_path)

        final_df = build_final_output(predictions_df)

        validate_predictions(final_df)

        row_count = final_df.count()

        logging.info("Final output row count: %s", row_count)
        logging.info("Final output schema:")
        final_df.printSchema()

        write_parquet(final_df, final_output_path)

        logging.info("Final scored dataset saved to: %s", final_output_path)
        logging.info("Final output layer completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Final output layer failed: %s", exc)
        sys.exit(1)