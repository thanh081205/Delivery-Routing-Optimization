"""
Utilities for feature extraction and preprocessing for travel-time prediction.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd


DEFAULT_MAXSPEED_KMH = 40.0
DEFAULT_WEATHER = "clear"
DEFAULT_TIME_OF_DAY = "normal"


def parse_maxspeed_kmh(raw_value: Any, default: float = DEFAULT_MAXSPEED_KMH) -> float:
    """Parse maxspeed values from OSM-like fields into km/h."""
    if raw_value is None:
        return float(default)

    if isinstance(raw_value, (int, float)):
        return float(raw_value)

    if isinstance(raw_value, list) and raw_value:
        raw_value = raw_value[0]

    text = str(raw_value).strip().lower()
    if not text:
        return float(default)

    text = text.replace("km/h", "").replace("kph", "").strip()
    token = text.split()[0]
    try:
        return float(token)
    except (TypeError, ValueError):
        return float(default)


def normalize_highway(raw_value: Any) -> str:
    """Normalize OSM highway tag into a single string category."""
    if raw_value is None:
        return "unclassified"
    if isinstance(raw_value, list) and raw_value:
        raw_value = raw_value[0]
    value = str(raw_value).strip().lower()
    return value if value else "unclassified"


def normalize_weather(raw_value: Any) -> str:
    """Collapse weather aliases into the labels used by TV3/TV4."""
    aliases = {
        "clear": "clear",
        "sunny": "clear",
        "normal": "clear",
        "dry": "clear",
        "rain": "rain",
        "rainy": "rain",
        "storm": "rain",
        "wet": "rain",
    }
    token = str(raw_value or DEFAULT_WEATHER).strip().lower()
    return aliases.get(token, DEFAULT_WEATHER)


def normalize_time_of_day(raw_value: Any) -> str:
    """Collapse time-of-day aliases into the labels used by TV3/TV4."""
    aliases = {
        "normal": "normal",
        "offpeak": "normal",
        "off_peak": "normal",
        "off-peak": "normal",
        "peak": "peak",
        "rush": "peak",
        "rush_hour": "peak",
        "rush-hour": "peak",
    }
    token = str(raw_value or DEFAULT_TIME_OF_DAY).strip().lower()
    return aliases.get(token, DEFAULT_TIME_OF_DAY)


def to_peak_hour_flag(raw_value: Any) -> int:
    """Convert time-of-day style input into a stable binary peak-hour flag."""
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, (int, float)):
        return 1 if int(raw_value) else 0
    return 1 if normalize_time_of_day(raw_value) == "peak" else 0


def edge_to_feature_row(
    edge: Dict[str, Any],
    weather: str,
    is_peak_hour: int,
    congestion_prob: float,
) -> Dict[str, Any]:
    """Convert a map edge into a model feature row."""
    return {
        "length_m": float(edge.get("length", 0.0) or 0.0),
        "maxspeed_kmh": parse_maxspeed_kmh(edge.get("maxspeed")),
        "road_type": normalize_highway(edge.get("highway")),
        "weather": normalize_weather(weather),
        "is_peak_hour": int(is_peak_hour),
        "congestion_prob": float(congestion_prob),
    }


def require_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> None:
    """Raise a helpful error when an integration DataFrame misses required columns."""
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise KeyError(f"{df_name} is missing required columns: {missing}")


def cast_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce stable dtypes for downstream sklearn pipeline."""
    out = df.copy()
    out["length_m"] = pd.to_numeric(out["length_m"], errors="coerce").fillna(0.0)
    out["maxspeed_kmh"] = pd.to_numeric(out["maxspeed_kmh"], errors="coerce").fillna(
        DEFAULT_MAXSPEED_KMH
    )
    out["is_peak_hour"] = pd.to_numeric(out["is_peak_hour"], errors="coerce").fillna(0).astype(int)
    out["congestion_prob"] = pd.to_numeric(out["congestion_prob"], errors="coerce").fillna(0.0)
    out["road_type"] = out["road_type"].fillna("unclassified").map(normalize_highway).astype(str)
    out["weather"] = out["weather"].fillna(DEFAULT_WEATHER).map(normalize_weather).astype(str)
    return out
