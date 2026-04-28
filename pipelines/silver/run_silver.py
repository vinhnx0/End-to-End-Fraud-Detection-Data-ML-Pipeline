from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import time

import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# Get the project root path
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

PROJECT_ROOT = get_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Set up logging for tracking process
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Load the configuration from the config.yaml file
def load_config() -> dict:
    config_path = Path(os.getenv("PIPELINE_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Build Spark session for processing
def build_spark(app_name: str = "silver_join_clean") -> SparkSession:
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

# Ensure the given directory exists
def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

# Read Parquet file into Spark DataFrame
def read_parquet(spark: SparkSession, parquet_path: Path) -> DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet input: {parquet_path}")
    return spark.read.parquet(str(parquet_path))

# Write the DataFrame to Parquet format
def write_parquet(df: DataFrame, output_path: Path) -> None:
    df.write.mode("overwrite").parquet(str(output_path))

# Join transaction and identity datasets on TransactionID
def join_transaction_identity(
    transaction_df: DataFrame,
    identity_df: DataFrame,
) -> DataFrame:
    return transaction_df.join(identity_df, on="TransactionID", how="left")

# Clean the data by handling null values and ensuring proper types
def clean_data(df: DataFrame) -> DataFrame:
    # Handle missing values by filling nulls with zero (or another strategy as needed)
    return df.fillna(0)

# Process the files: Join and clean
def process_files(spark: SparkSession, bronze_root: Path, silver_root: Path) -> None:
    logging.info("Reading bronze data")

    # Read in the bronze layer data (train and test datasets)
    train_transaction = read_parquet(spark, bronze_root / "train_transaction")
    train_identity = read_parquet(spark, bronze_root / "train_identity")
    test_transaction = read_parquet(spark, bronze_root / "test_transaction")
    test_identity = read_parquet(spark, bronze_root / "test_identity")

    # Join the datasets
    logging.info("Joining train_transaction with train_identity")
    train_joined = join_transaction_identity(train_transaction, train_identity)
    logging.info("Joining test_transaction with test_identity")
    test_joined = join_transaction_identity(test_transaction, test_identity)

    # Clean the joined data
    logging.info("Cleaning train and test data")
    train_cleaned = clean_data(train_joined)
    test_cleaned = clean_data(test_joined)

    # Write the cleaned data to the silver layer
    logging.info("Writing cleaned data to silver layer")
    write_parquet(train_cleaned, silver_root / "train_joined")
    write_parquet(test_cleaned, silver_root / "test_joined")

# Main function to execute the join and clean process
def main() -> None:
    setup_logging()

    config = load_config()
    bronze_root = PROJECT_ROOT / config["data"]["bronze_root"]
    silver_root = PROJECT_ROOT / config["data"]["silver_root"]
    ensure_directory(silver_root)

    spark = build_spark()

    try:
        # Process the files (join and clean)
        process_files(spark, bronze_root, silver_root)

        logging.info("Silver layer join and clean completed successfully.")
    finally:
        spark.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Silver layer join and clean failed: %s", exc)
        sys.exit(1)