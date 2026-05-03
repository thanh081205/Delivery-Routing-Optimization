"""

Mạng Bayes rút gọn:
    Weather        -> RoadCondition
    TimeOfDay      -> TrafficDemand
    RoadType       -> TrafficDemand, EffectiveCapacity, AccidentRisk
    MaxSpeed       -> EffectiveCapacity
    Length         -> AccidentRisk
    RoadCondition, TrafficDemand, EffectiveCapacity, AccidentRisk -> Congestion
"""

from __future__ import annotations

import re
from itertools import product
from typing import Any

import pandas as pd


ROAD_EXPRESSWAY = "expressway"
ROAD_ARTERIAL = "arterial"
ROAD_COLLECTOR = "collector"
ROAD_LOCAL = "local"
ROAD_UNKNOWN = "unknown"

_HIGHWAY_TO_ROAD_TYPE = {
    "motorway": ROAD_EXPRESSWAY,
    "motorway_link": ROAD_EXPRESSWAY,
    "trunk": ROAD_EXPRESSWAY,
    "trunk_link": ROAD_EXPRESSWAY,
    "primary": ROAD_ARTERIAL,
    "primary_link": ROAD_ARTERIAL,
    "secondary": ROAD_ARTERIAL,
    "secondary_link": ROAD_ARTERIAL,
    "tertiary": ROAD_COLLECTOR,
    "tertiary_link": ROAD_COLLECTOR,
    "unclassified": ROAD_COLLECTOR,
    "residential": ROAD_LOCAL,
    "living_street": ROAD_LOCAL,
    "service": ROAD_LOCAL,
    "road": ROAD_UNKNOWN,
}

# P(RoadCondition = bad | Weather)
_CPT_BAD_ROAD_CONDITION = {
    "clear": 0.08,
    "rain": 0.45,
}

# P(TrafficDemand = high | TimeOfDay, RoadType)
_CPT_HIGH_TRAFFIC_DEMAND = {
    ("normal", ROAD_EXPRESSWAY): 0.24,
    ("normal", ROAD_ARTERIAL): 0.32,
    ("normal", ROAD_COLLECTOR): 0.28,
    ("normal", ROAD_LOCAL): 0.20,
    ("normal", ROAD_UNKNOWN): 0.25,
    ("peak", ROAD_EXPRESSWAY): 0.55,
    ("peak", ROAD_ARTERIAL): 0.70,
    ("peak", ROAD_COLLECTOR): 0.62,
    ("peak", ROAD_LOCAL): 0.48,
    ("peak", ROAD_UNKNOWN): 0.55,
}

# Base P(EffectiveCapacity = low | RoadType), then adjusted by speed/weather.
_CPT_LOW_CAPACITY_BASE = {
    ROAD_EXPRESSWAY: 0.10,
    ROAD_ARTERIAL: 0.18,
    ROAD_COLLECTOR: 0.24,
    ROAD_LOCAL: 0.32,
    ROAD_UNKNOWN: 0.27,
}

# Base P(AccidentRisk = high | Weather, RoadType), then adjusted by length.
_CPT_HIGH_ACCIDENT_RISK_BASE = {
    ("clear", ROAD_EXPRESSWAY): 0.04,
    ("clear", ROAD_ARTERIAL): 0.05,
    ("clear", ROAD_COLLECTOR): 0.05,
    ("clear", ROAD_LOCAL): 0.04,
    ("clear", ROAD_UNKNOWN): 0.05,
    ("rain", ROAD_EXPRESSWAY): 0.13,
    ("rain", ROAD_ARTERIAL): 0.16,
    ("rain", ROAD_COLLECTOR): 0.15,
    ("rain", ROAD_LOCAL): 0.12,
    ("rain", ROAD_UNKNOWN): 0.14,
}

# P(Congestion = true | HighDemand, LowCapacity, BadCondition, HighAccident)
_CPT_CONGESTION = {
    (False, False, False, False): 0.04,
    (True, False, False, False): 0.18,
    (False, True, False, False): 0.16,
    (False, False, True, False): 0.10,
    (False, False, False, True): 0.12,
    (True, True, False, False): 0.42,
    (True, False, True, False): 0.30,
    (True, False, False, True): 0.34,
    (False, True, True, False): 0.28,
    (False, True, False, True): 0.31,
    (False, False, True, True): 0.22,
    (True, True, True, False): 0.58,
    (True, True, False, True): 0.63,
    (True, False, True, True): 0.50,
    (False, True, True, True): 0.47,
    (True, True, True, True): 0.82,
}

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _clip_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _normalize_highway(value: Any) -> str:
    for item in _as_list(value):
        token = str(item).strip().lower()
        if token:
            return token
    return "unclassified"


def _road_type_from_highway(highway: str) -> str:
    return _HIGHWAY_TO_ROAD_TYPE.get(highway, ROAD_UNKNOWN)


def _normalize_weather(weather: str) -> str:
    token = str(weather).strip().lower()
    aliases = {
        "clear": "clear",
        "sunny": "clear",
        "dry": "clear",
        "normal": "clear",
        "nang": "clear",
        "nắng": "clear",
        "troi nang": "clear",
        "trời nắng": "clear",
        "rain": "rain",
        "rainy": "rain",
        "storm": "rain",
        "heavy_rain": "rain",
        "mua": "rain",
        "mưa": "rain",
        "troi mua": "rain",
        "trời mưa": "rain",
    }
    if token not in aliases:
        raise ValueError("weather phải là 'rain' hoặc 'clear'.")
    return aliases[token]


def _normalize_time_of_day(time_of_day: str) -> str:
    token = str(time_of_day).strip().lower()
    aliases = {
        "normal": "normal",
        "offpeak": "normal",
        "off_peak": "normal",
        "binh thuong": "normal",
        "bình thường": "normal",
        "peak": "peak",
        "rush": "peak",
        "rush_hour": "peak",
        "cao_diem": "peak",
        "cao diem": "peak",
        "cao điểm": "peak",
        "gio cao diem": "peak",
        "giờ cao điểm": "peak",
    }
    if token not in aliases:
        raise ValueError("time_of_day phải là 'peak' hoặc 'normal'.")
    return aliases[token]


def _first_number(value: Any) -> float | None:
    for item in _as_list(value):
        if isinstance(item, (int, float)):
            return float(item)

        text = str(item).strip().lower().replace(",", ".")
        match = _NUMBER_RE.search(text)
        if not match:
            continue
        try:
            return float(match.group(0))
        except ValueError:
            continue
    return None


def _safe_maxspeed(value: Any) -> float:
    speed = _first_number(value)
    if speed is None or speed != speed or speed <= 0:
        return 40.0

    text = str(_as_list(value)[0]).lower() if _as_list(value) else ""
    if "mph" in text:
        return speed * 1.60934
    return speed


def _safe_length(value: Any) -> float:
    length = _first_number(value)
    if length is None or length != length or length <= 0:
        return 100.0
    return length


def _low_capacity_probability(road_type: str, maxspeed: float, weather: str) -> float:
    probability = _CPT_LOW_CAPACITY_BASE[road_type]

    if maxspeed >= 70:
        probability -= 0.04
    elif maxspeed < 30:
        probability += 0.13
    elif maxspeed < 45:
        probability += 0.07

    if weather == "rain":
        probability += 0.06

    return _clip_probability(probability)


def _high_accident_probability(road_type: str, weather: str, length_m: float) -> float:
    probability = _CPT_HIGH_ACCIDENT_RISK_BASE[(weather, road_type)]

    if length_m > 2000:
        probability += 0.07
    elif length_m > 1000:
        probability += 0.04
    elif length_m > 500:
        probability += 0.02

    return _clip_probability(probability)


def _state_probability(state_is_true: bool, probability_true: float) -> float:
    return probability_true if state_is_true else 1.0 - probability_true


def _infer_congestion_probability(
    *,
    highway: str,
    maxspeed: float,
    length_m: float,
    weather: str,
    time_of_day: str,
) -> float:
    """
    Marginalize các node ẩn để tính P(Congestion=True | evidence).

    Evidence quan sát được: weather, time_of_day, highway, maxspeed, length.
    Node ẩn: TrafficDemand, EffectiveCapacity, RoadCondition, AccidentRisk.
    """
    road_type = _road_type_from_highway(highway)
    p_bad_condition = _CPT_BAD_ROAD_CONDITION[weather]
    p_high_demand = _CPT_HIGH_TRAFFIC_DEMAND[(time_of_day, road_type)]
    p_low_capacity = _low_capacity_probability(road_type, maxspeed, weather)
    p_high_accident = _high_accident_probability(road_type, weather, length_m)

    p_congestion = 0.0
    for high_demand, low_capacity, bad_condition, high_accident in product((False, True), repeat=4):
        p_hidden_state = (
            _state_probability(high_demand, p_high_demand)
            * _state_probability(low_capacity, p_low_capacity)
            * _state_probability(bad_condition, p_bad_condition)
            * _state_probability(high_accident, p_high_accident)
        )
        p_congestion += p_hidden_state * _CPT_CONGESTION[
            (high_demand, low_capacity, bad_condition, high_accident)
        ]

    return _clip_probability(p_congestion)


def compute_congestion(edges: pd.DataFrame, weather: str, time_of_day: str) -> pd.DataFrame:
    """
    Tính xác suất kẹt xe cho từng cạnh bằng mạng Bayes

    Args:
        edges: DataFrame cần có tối thiểu các cột u, v, key, length, maxspeed, highway.
        weather: "rain" hoặc "clear".
        time_of_day: "peak" hoặc "normal".

    Returns:
        DataFrame gồm đúng các cột u, v, key, p_congestion.
    """
    required_columns = {"u", "v", "key", "length", "maxspeed", "highway"}
    missing = required_columns.difference(edges.columns)
    if missing:
        raise KeyError(f"edges đang thiếu các cột bắt buộc: {sorted(missing)}")

    weather_norm = _normalize_weather(weather)
    time_norm = _normalize_time_of_day(time_of_day)

    records: list[dict[str, Any]] = []
    for _, row in edges.iterrows():
        highway = _normalize_highway(row["highway"])
        probability = _infer_congestion_probability(
            highway=highway,
            maxspeed=_safe_maxspeed(row["maxspeed"]),
            length_m=_safe_length(row["length"]),
            weather=weather_norm,
            time_of_day=time_norm,
        )
        records.append(
            {
                "u": row["u"],
                "v": row["v"],
                "key": row["key"],
                "p_congestion": round(float(probability), 4),
            }
        )

    return pd.DataFrame(records, columns=["u", "v", "key", "p_congestion"])


__all__ = ["compute_congestion"]
