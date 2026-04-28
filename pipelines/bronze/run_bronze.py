from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import time

import yaml
from pyspark.sql import DataFrame, SparkSession

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
def build_spark(app_name: str = "bronze_ingestion") -> SparkSession:
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

# Ensure the given directory exists
def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

# Read CSV file into Spark DataFrame
def read_csv(spark: SparkSession, csv_path: Path) -> DataFrame:
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(csv_path))
    )

# Write the DataFrame to Parquet format
def write_parquet(df: DataFrame, output_path: Path) -> None:
    df.write.mode("overwrite").parquet(str(output_path))

# Process a file: Read CSV, log, and convert to Parquet
def process_file(spark: SparkSession, source_path: Path, bronze_path: Path) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing input file: {source_path}")

    logging.info("Reading CSV: %s", source_path)
    start = time.time()

    df = read_csv(spark, source_path)
    row_count = df.count()
    col_count = len(df.columns)

    logging.info("Loaded %s | rows=%s | cols=%s", source_path.name, row_count, col_count)
    write_parquet(df, bronze_path)

    elapsed = time.time() - start
    logging.info("Written bronze parquet: %s | elapsed=%.2fs", bronze_path, elapsed)

# Main function to execute the ingestion process
def main() -> None:
    setup_logging()

    config = load_config()
    raw_root = PROJECT_ROOT / config["data"]["raw_root"]
    bronze_root = PROJECT_ROOT / config["data"]["bronze_root"]
    ensure_directory(bronze_root)

    file_map = {
        "train_transaction.csv": bronze_root / "train_transaction",
        "train_identity.csv": bronze_root / "train_identity",
        "test_transaction.csv": bronze_root / "test_transaction",
        "test_identity.csv": bronze_root / "test_identity",
    }

    spark = build_spark()

    try:
        for file_name, output_dir in file_map.items():
            source_file = raw_root / file_name
            process_file(spark, source_file, output_dir)

        logging.info("Bronze ingestion completed successfully.")
    finally:
        spark.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Bronze ingestion failed: %s", exc)
        sys.exit(1)