from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import yaml
from pyspark.sql import DataFrame, SparkSession


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = get_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"


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


def build_spark(app_name: str = "bronze_to_silver_ieee_cis") -> SparkSession:
    driver_memory = os.getenv("SPARK_DRIVER_MEMORY", "4g")
    shuffle_partitions = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "8")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
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


def write_parquet(df: DataFrame, path: Path, num_partitions: int = 4) -> None:
    if path.exists():
        import shutil
        shutil.rmtree(path)

    (
        df.repartition(num_partitions)
        .write
        .mode("overwrite")
        .parquet(str(path))
    )


def join_transaction_identity(
    transaction_df: DataFrame,
    identity_df: DataFrame,
) -> DataFrame:
    return transaction_df.join(identity_df, on="TransactionID", how="left")


def main() -> None:
    setup_logging()

    config = load_config()
    bronze_root = PROJECT_ROOT / config["data"]["bronze_root"]
    silver_root = PROJECT_ROOT / "data" / "silver"
    silver_root.mkdir(parents=True, exist_ok=True)

    spark = build_spark()

    try:
        train_transaction = read_parquet(spark, bronze_root / "train_transaction")
        train_identity = read_parquet(spark, bronze_root / "train_identity")
        test_transaction = read_parquet(spark, bronze_root / "test_transaction")
        test_identity = read_parquet(spark, bronze_root / "test_identity")

        logging.info("Joining train_transaction with train_identity")
        train_joined = join_transaction_identity(train_transaction, train_identity)

        logging.info("Joining test_transaction with test_identity")
        test_joined = join_transaction_identity(test_transaction, test_identity)

        logging.info(
            "Train joined | rows=%s | cols=%s",
            train_joined.count(),
            len(train_joined.columns),
        )
        logging.info(
            "Test joined | rows=%s | cols=%s",
            test_joined.count(),
            len(test_joined.columns),
        )

        write_parquet(train_joined, silver_root / "train_joined")
        write_parquet(test_joined, silver_root / "test_joined")

        logging.info("Silver layer created successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Silver build failed: %s", exc)
        sys.exit(1)