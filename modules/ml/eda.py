"""
Quick EDA script for TV3 report preparation.

Usage:
    python -m modules.ml.eda
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from modules.ml.train import ARTIFACTS_DIR, MOCK_DATASET_PATH, build_mock_dataset


LENGTH_PLOT_PATH = ARTIFACTS_DIR / "eda_length_distribution.png"
TARGET_PLOT_PATH = ARTIFACTS_DIR / "eda_travel_time_distribution.png"
SUMMARY_CSV_PATH = ARTIFACTS_DIR / "eda_summary.csv"


def _load_or_create_dataset() -> pd.DataFrame:
    if MOCK_DATASET_PATH.exists():
        return pd.read_csv(MOCK_DATASET_PATH)

    dataset, _ = build_mock_dataset(prefer_map_edges=True)
    return dataset


def export_eda_artifacts() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = _load_or_create_dataset()

    summary = dataset.describe(include="all").transpose()
    summary.to_csv(SUMMARY_CSV_PATH)

    plt.figure(figsize=(7, 4))
    plt.hist(dataset["length_m"], bins=30, color="#5B8FF9", edgecolor="white")
    plt.xlabel("Length (m)")
    plt.ylabel("Count")
    plt.title("Edge Length Distribution")
    plt.tight_layout()
    plt.savefig(LENGTH_PLOT_PATH, dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(dataset["travel_time_min"], bins=30, color="#61DDAA", edgecolor="white")
    plt.xlabel("Travel time (minutes)")
    plt.ylabel("Count")
    plt.title("Travel Time Distribution")
    plt.tight_layout()
    plt.savefig(TARGET_PLOT_PATH, dpi=180)
    plt.close()

    print(f"Saved EDA summary to: {SUMMARY_CSV_PATH}")
    print(f"Saved plots to: {LENGTH_PLOT_PATH} and {TARGET_PLOT_PATH}")


if __name__ == "__main__":
    export_eda_artifacts()
