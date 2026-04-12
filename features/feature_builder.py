from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ByteType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
)
import yaml


NUMERIC_TYPES = (
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    DecimalType,
)


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


def build_spark(app_name: str = "feature_builder_ieee_cis") -> SparkSession:
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


def get_numeric_feature_columns(df: DataFrame, exclude_cols: set[str]) -> list[str]:
    numeric_cols: list[str] = []
    for field in df.schema.fields:
        if field.name in exclude_cols:
            continue
        if isinstance(field.dataType, NUMERIC_TYPES):
            numeric_cols.append(field.name)
    return numeric_cols


def prepare_train_features(df: DataFrame) -> tuple[DataFrame, list[str]]:
    if "isFraud" not in df.columns:
        raise ValueError("Train dataset must contain isFraud column.")

    label_col = "isFraud"
    exclude_cols = {"TransactionID", label_col}
    feature_cols = get_numeric_feature_columns(df, exclude_cols=exclude_cols)

    prepared = df.select(
        "TransactionID",
        label_col,
        *[F.coalesce(F.col(c).cast("double"), F.lit(0.0)).alias(c) for c in feature_cols],
    )

    return prepared, feature_cols


def prepare_test_features(df: DataFrame, feature_cols: list[str]) -> DataFrame:
    existing_cols = set(df.columns)

    select_exprs = ["TransactionID"]
    for col_name in feature_cols:
        if col_name in existing_cols:
            select_exprs.append(
                F.coalesce(F.col(col_name).cast("double"), F.lit(0.0)).alias(col_name)
            )
        else:
            select_exprs.append(F.lit(0.0).cast("double").alias(col_name))

    return df.select(*select_exprs)


def main() -> None:
    setup_logging()

    config = load_config()
    silver_root = PROJECT_ROOT / "data" / "silver"
    gold_root = PROJECT_ROOT / "data" / "gold"
    artifacts_root = PROJECT_ROOT / "data" / "artifacts"

    gold_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    spark = build_spark()

    try:
        train_joined = read_parquet(spark, silver_root / "train_joined")
        test_joined = read_parquet(spark, silver_root / "test_joined")

        train_features, feature_cols = prepare_train_features(train_joined)
        test_features = prepare_test_features(test_joined, feature_cols)

        logging.info(
            "Prepared train features | rows=%s | feature_count=%s",
            train_features.count(),
            len(feature_cols),
        )
        logging.info(
            "Prepared test features | rows=%s | feature_count=%s",
            test_features.count(),
            len(feature_cols),
        )

        write_parquet(train_features, gold_root / "train_features")
        write_parquet(test_features, gold_root / "test_features")

        metadata = {
            "label_column": "isFraud",
            "id_column": "TransactionID",
            "feature_columns": feature_cols,
            "feature_count": len(feature_cols),
        }

        metadata_path = artifacts_root / "feature_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logging.info("Feature datasets written successfully.")
        logging.info("Feature metadata saved to %s", metadata_path)
    finally:
        spark.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Feature building failed: %s", exc)
        sys.exit(1)