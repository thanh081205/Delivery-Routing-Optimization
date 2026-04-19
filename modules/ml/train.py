"""
Train a travel-time prediction model for routing.

Usage:
    python -m modules.ml.train
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from modules.ml.preprocess import cast_feature_types, edge_to_feature_row
from modules.ml.sample_data import describe_edge_source, generate_edge_catalog


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
FEATURES_DIR = ROOT_DIR / "features"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MOCK_DATASET_PATH = DATA_DIR / "mock_edge_travel_time.csv"
MODEL_PATH = FEATURES_DIR / "travel_time_model.pkl"
SUMMARY_PATH = ARTIFACTS_DIR / "training_summary.json"
MODEL_COMPARISON_CSV = ARTIFACTS_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_CSV = ARTIFACTS_DIR / "feature_importance.csv"
MODEL_COMPARISON_PNG = ARTIFACTS_DIR / "model_comparison.png"
FEATURE_IMPORTANCE_PNG = ARTIFACTS_DIR / "feature_importance.png"

FEATURE_COLUMNS = [
    "length_m",
    "maxspeed_kmh",
    "road_type",
    "weather",
    "is_peak_hour",
    "congestion_prob",
]
TARGET_COLUMN = "travel_time_min"


def _simulate_travel_time_minutes(feature_row: Dict[str, float]) -> float:
    """Simulate realistic travel-time labels from edge features."""
    base_speed = max(feature_row["maxspeed_kmh"] * 0.76, 8.0)
    weather_penalty = 1.28 if feature_row["weather"] == "rain" else 1.0
    peak_penalty = 1.22 if feature_row["is_peak_hour"] == 1 else 1.0
    congestion_penalty = 1.0 + 0.95 * float(feature_row["congestion_prob"])

    road_penalty = {
        "motorway": 0.92,
        "primary": 0.98,
        "secondary": 1.02,
        "tertiary": 1.06,
        "residential": 1.12,
        "service": 1.18,
    }.get(str(feature_row["road_type"]), 1.08)

    effective_speed = base_speed / (weather_penalty * peak_penalty * congestion_penalty * road_penalty)
    travel_hours = (feature_row["length_m"] / 1000.0) / max(effective_speed, 5.0)
    noise = random.uniform(-0.25, 0.25)
    return max(travel_hours * 60.0 + noise, 0.2)


def _try_load_edges_from_map(
    latitude: float = 10.7729,
    longitude: float = 106.6578,
    dist: int = 1000,
) -> Tuple[pd.DataFrame | None, str]:
    """
    Lazily try to load map edges from TV1's module.

    This keeps TV3 runnable even when OSMnx or network access is unavailable.
    """
    try:
        from modules.graph.map_loader import load_map

        with contextlib.redirect_stdout(io.StringIO()):
            graph_data = load_map(latitude, longitude, dist)
        edges_df = graph_data["edges"].copy()
        return edges_df, "map_loader"
    except Exception as exc:
        return None, f"synthetic_fallback ({exc.__class__.__name__})"


def build_mock_dataset(
    num_synthetic_edges: int = 300,
    random_state: int = 42,
    prefer_map_edges: bool = True,
) -> tuple[pd.DataFrame, str]:
    """
    Build a mock travel-time dataset from real map edges when possible,
    otherwise fall back to a reproducible synthetic edge catalog.
    """
    random.seed(random_state)
    np.random.seed(random_state)

    edges_df: pd.DataFrame | None = None
    source_name = "synthetic"
    if prefer_map_edges:
        edges_df, source_name = _try_load_edges_from_map()

    if edges_df is None or edges_df.empty:
        edges_df = generate_edge_catalog(num_edges=num_synthetic_edges, random_state=random_state)
        source_name = "synthetic"

    weather_options = ["clear", "rain"]
    rows: List[Dict[str, Any]] = []
    for _, edge in edges_df.iterrows():
        edge_dict = edge.to_dict()
        for weather in weather_options:
            for is_peak_hour in [0, 1]:
                base_prob = 0.14 if weather == "clear" else 0.44
                if is_peak_hour:
                    base_prob += 0.20
                if str(edge_dict.get("highway")) in {"residential", "service"}:
                    base_prob += 0.04

                congestion_prob = float(np.clip(base_prob + random.uniform(-0.08, 0.08), 0.01, 0.99))
                feature_row = edge_to_feature_row(
                    edge=edge_dict,
                    weather=weather,
                    is_peak_hour=is_peak_hour,
                    congestion_prob=congestion_prob,
                )
                feature_row[TARGET_COLUMN] = _simulate_travel_time_minutes(feature_row)
                rows.append(feature_row)

    dataset = pd.DataFrame(rows)
    dataset = cast_feature_types(dataset)
    return dataset, describe_edge_source(source_name, len(edges_df))


def _build_preprocessor() -> ColumnTransformer:
    numeric_cols = ["length_m", "maxspeed_kmh", "is_peak_hour", "congestion_prob"]
    categorical_cols = ["road_type", "weather"]
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def _evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)
    return {"name": name, "mae": float(mae), "rmse": rmse, "r2": float(r2)}


def _extract_feature_importance(best_model: Pipeline) -> pd.DataFrame:
    preprocess = best_model.named_steps["preprocess"]
    model = best_model.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance_values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importance_values = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    else:
        importance_values = np.zeros(len(feature_names), dtype=float)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    return importance_df


def _export_training_artifacts(
    dataset: pd.DataFrame,
    source_name: str,
    comparison_df: pd.DataFrame,
    best_name: str,
    best_metrics: Dict[str, float],
    importance_df: pd.DataFrame,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(MODEL_COMPARISON_CSV, index=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    summary = {
        "dataset_source": source_name,
        "num_samples": int(len(dataset)),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "selected_model": best_name,
        "metrics": best_metrics,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(7, 4))
    plt.bar(comparison_df["name"], comparison_df["rmse"], color=["#5B8FF9", "#61DDAA"])
    plt.ylabel("RMSE (minutes)")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PNG, dpi=180)
    plt.close()

    top_importance = importance_df.head(10).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(top_importance["feature"], top_importance["importance"], color="#F6BD16")
    plt.xlabel("Importance")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PNG, dpi=180)
    plt.close()


def train_and_save(
    random_state: int = 42,
    prefer_map_edges: bool = True,
) -> Dict[str, Any]:
    """
    Train the ML model, persist it for downstream modules, and export report artifacts.
    """
    random.seed(random_state)
    np.random.seed(random_state)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    dataset, source_name = build_mock_dataset(
        num_synthetic_edges=300,
        random_state=random_state,
        prefer_map_edges=prefer_map_edges,
    )
    dataset.to_csv(MOCK_DATASET_PATH, index=False)

    x = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    candidates = [
        (
            "linear_regression",
            Pipeline(
                steps=[
                    ("preprocess", _build_preprocessor()),
                    ("model", LinearRegression()),
                ]
            ),
        ),
        (
            "decision_tree",
            Pipeline(
                steps=[
                    ("preprocess", _build_preprocessor()),
                    ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=4, random_state=random_state)),
                ]
            ),
        ),
    ]

    comparison_rows: List[Dict[str, float]] = []
    best_metrics: Dict[str, float] | None = None
    best_name = ""
    best_model: Pipeline | None = None

    for name, model in candidates:
        model.fit(x_train, y_train)
        metrics = _evaluate_model(name, model, x_test, y_test)
        comparison_rows.append(metrics)
        print(
            f"{name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}"
        )
        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_metrics = metrics
            best_name = name
            best_model = model

    if best_model is None or best_metrics is None:
        raise RuntimeError("No model was trained successfully.")

    comparison_df = pd.DataFrame(comparison_rows).sort_values("rmse", ascending=True, ignore_index=True)
    importance_df = _extract_feature_importance(best_model)
    _export_training_artifacts(
        dataset=dataset,
        source_name=source_name,
        comparison_df=comparison_df,
        best_name=best_name,
        best_metrics=best_metrics,
        importance_df=importance_df,
    )

    artifact = {
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "pipeline": best_model,
        "metrics": best_metrics,
        "dataset_source": source_name,
        "artifacts_dir": str(ARTIFACTS_DIR),
    }
    joblib.dump(artifact, MODEL_PATH)

    print(f"Dataset source: {source_name}")
    print(f"Saved mock dataset to: {MOCK_DATASET_PATH}")
    print(f"Saved model artifact to: {MODEL_PATH}")
    print(f"Saved report artifacts to: {ARTIFACTS_DIR}")
    print(f"Selected model: {best_name} with RMSE={best_metrics['rmse']:.4f}")
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TV3 travel-time predictor.")
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip TV1 map_loader and build the mock dataset from synthetic edges only.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataset generation and model training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_and_save(
        random_state=args.seed,
        prefer_map_edges=not args.synthetic_only,
    )
