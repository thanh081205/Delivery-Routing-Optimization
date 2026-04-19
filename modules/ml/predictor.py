"""
Inference helper for travel-time prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from modules.ml.preprocess import cast_feature_types, edge_to_feature_row


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT_DIR / "features" / "travel_time_model.pkl"


class TravelTimePredictor:
    """Load model artifact once and reuse for low-latency predictions."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        artifact = self._load_or_create_artifact(model_path)
        self.feature_columns = artifact["feature_columns"]
        self.pipeline = artifact["pipeline"]
        self.model_name = artifact.get("model_name", "unknown")
        self.metrics = artifact.get("metrics", {})
        self.dataset_source = artifact.get("dataset_source", "unknown")

    def _load_or_create_artifact(self, model_path: Path) -> Dict[str, Any]:
        if not model_path.exists():
            from modules.ml.train import train_and_save

            train_and_save()

        artifact = joblib.load(model_path)
        required_keys = {"feature_columns", "pipeline"}
        missing = required_keys.difference(artifact.keys())
        if missing:
            raise KeyError(f"Invalid model artifact at {model_path}; missing keys: {sorted(missing)}")
        return artifact

    def predict_travel_time(self, edge_features: Dict[str, Any]) -> float:
        """Predict travel time in minutes from prepared feature dict."""
        frame = pd.DataFrame([edge_features], columns=self.feature_columns)
        frame = cast_feature_types(frame)
        prediction = float(self.pipeline.predict(frame)[0])
        return round(max(prediction, 0.0), 4)


_DEFAULT_PREDICTOR: Optional[TravelTimePredictor] = None


def predict_travel_time(
    edge: Dict[str, Any],
    weather: str = "clear",
    is_peak_hour: int = 0,
    congestion_prob: float = 0.2,
) -> float:
    """
    Convenience API for other modules (TV2/TV1).

    Args:
        edge: edge dict from map loader, e.g. {"length", "maxspeed", "highway"}.
        weather: weather state ("clear" or "rain").
        is_peak_hour: 0/1 flag.
        congestion_prob: probability from Bayes module (TV4).
    """
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = TravelTimePredictor()

    feature_row = edge_to_feature_row(
        edge=edge,
        weather=weather,
        is_peak_hour=is_peak_hour,
        congestion_prob=congestion_prob,
    )
    return _DEFAULT_PREDICTOR.predict_travel_time(feature_row)
