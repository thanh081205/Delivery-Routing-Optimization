"""Machine learning module for travel-time estimation."""

from modules.ml.predictor import TravelTimePredictor, predict_travel_time
from modules.ml.travel_time_predictor import predict_travel_time as predict_travel_time_batch

__all__ = [
    "TravelTimePredictor",
    "predict_travel_time",
    "predict_travel_time_batch",
]
