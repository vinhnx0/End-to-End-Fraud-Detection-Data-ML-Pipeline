from __future__ import annotations

import subprocess
import sys
import time


PHASES = [
    ("Bronze", "python pipelines/bronze/run_bronze.py"),
    ("Silver", "python pipelines/silver/run_silver.py"),
    ("Gold", "python pipelines/gold/run_gold.py"),
    ("Scoring", "python pipelines/scoring/run_scoring.py"),
    ("Final", "python pipelines/final/run_final.py"),
]


def run_phase(name: str, command: str) -> None:
    print(f"\n==============================")
    print(f"Running phase: {name}")
    print(f"Command: {command}")
    print(f"==============================\n")

    start = time.time()

    result = subprocess.run(
        command,
        shell=True,
        text=True,
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"{name} phase failed with exit code {result.returncode}"
        )

    print(f"\n{name} completed successfully in {elapsed:.2f}s")


def main() -> None:
    print("Starting full fraud pipeline run")

    start = time.time()

    for name, command in PHASES:
        run_phase(name, command)

    elapsed = time.time() - start

    print("\n==============================")
    print("Full pipeline completed successfully")
    print(f"Total runtime: {elapsed:.2f}s")
    print("==============================")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        sys.exit(1)