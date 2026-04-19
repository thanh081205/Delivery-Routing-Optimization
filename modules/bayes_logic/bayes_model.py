"""
bayes_model.py — Module ước lượng xác suất kẹt xe cho từng cạnh

Module này bám theo interface trong INTERFACES.md:
    compute_congestion(edges: pd.DataFrame, weather: str, time_of_day: str) -> pd.DataFrame

Lưu ý quan trọng:
- Sơ đồ học thuật của TV4 dùng mạng Bayes 11 node.
- Nhưng interface tích hợp hiện tại chỉ truyền vào:
    + edges: length, maxspeed, highway
    + weather
    + time_of_day
- Vì vậy code v1 này triển khai một phiên bản suy luận rút gọn, trong đó một số node
  ẩn được “marginalize” hoặc xấp xỉ thông qua các feature quan sát được.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# Xác suất nền theo loại đường (proxy cho RoadType -> EffectiveCapacity / TrafficDemand)
_HIGHWAY_BASE_PRIOR = {
    "motorway": 0.16,
    "trunk": 0.20,
    "primary": 0.25,
    "secondary": 0.31,
    "tertiary": 0.36,
    "residential": 0.41,
    "living_street": 0.43,
    "service": 0.46,
    "unclassified": 0.38,
}

# Ảnh hưởng của thời tiết tới chất lượng đường và nguy cơ sự cố
_WEATHER_CONDITION_EFFECT = {
    "clear": 0.05,
    "rain": 0.28,
}

# Ảnh hưởng của khung giờ tới nhu cầu giao thông
_TIME_DEMAND_EFFECT = {
    "normal": 0.08,
    "peak": 0.30,
}


def _clip(value: float, low: float = 0.05, high: float = 0.98) -> float:
    return max(low, min(high, value))


def _normalize_highway(value: Any) -> str:
    """
    Chuẩn hóa trường highway từ OSM.

    highway có thể là:
    - str: "primary"
    - list[str]: ["residential", "unclassified"]
    """
    if isinstance(value, list) and value:
        value = value[0]
    if value is None:
        return "unclassified"
    token = str(value).strip().lower()
    return token if token else "unclassified"


def _normalize_weather(weather: str) -> str:
    token = str(weather).strip().lower()
    aliases = {
        "clear": "clear",
        "sunny": "clear",
        "normal": "clear",
        "rain": "rain",
        "rainy": "rain",
        "storm": "rain",
    }
    if token not in aliases:
        raise ValueError("weather phải là 'rain' hoặc 'clear' (có thể dùng alias như sunny/rainy).")
    return aliases[token]


def _normalize_time_of_day(time_of_day: str) -> str:
    token = str(time_of_day).strip().lower()
    aliases = {
        "normal": "normal",
        "offpeak": "normal",
        "off_peak": "normal",
        "peak": "peak",
        "rush": "peak",
        "rush_hour": "peak",
    }
    if token not in aliases:
        raise ValueError("time_of_day phải là 'peak' hoặc 'normal'.")
    return aliases[token]


def _safe_maxspeed(value: Any) -> float:
    """Chuẩn hóa maxspeed về float. map_loader của TV1 thường đã xử lý trước."""
    if isinstance(value, (int, float)):
        return float(value)

    try:
        if isinstance(value, list) and value:
            value = value[0]
        text = str(value).strip().split()[0]
        return float(text)
    except (TypeError, ValueError, IndexError):
        return 40.0


def _speed_capacity_penalty(maxspeed: float) -> float:
    """
    Xấp xỉ node EffectiveCapacity từ maxspeed.
    Tốc độ càng thấp -> năng lực thực tế càng thấp -> nguy cơ tắc càng cao.
    """
    if maxspeed >= 70:
        return 0.04
    if maxspeed >= 50:
        return 0.09
    if maxspeed >= 35:
        return 0.15
    return 0.22


def _length_exposure_penalty(length_m: float) -> float:
    """
    Đoạn đường dài hơn có xác suất gặp cản trở cao hơn đôi chút.
    Đây là một penalty nhẹ để phản ánh thời gian phơi nhiễm trên cạnh.
    """
    if length_m <= 200:
        return 0.02
    if length_m <= 500:
        return 0.05
    if length_m <= 1000:
        return 0.08
    return 0.11


def _infer_probabilities(highway: str, maxspeed: float, length_m: float, weather: str, time_of_day: str) -> float:
    """
    Suy luận xác suất kẹt xe theo phiên bản rút gọn của mạng Bayes 11 node.

    Mapping với mạng Bayes tổng quát:
    - RoadType        <- highway
    - RoadCondition   <- weather
    - TrafficDemand   <- time_of_day + road prior
    - EffectiveCapacity <- highway + maxspeed + weather proxy
    - AccidentRisk    <- weather + road prior
    - CongestionLevel <- kết hợp các node trung gian
    """
    base_prior = _HIGHWAY_BASE_PRIOR.get(highway, _HIGHWAY_BASE_PRIOR["unclassified"])
    p_bad_condition = _clip(_WEATHER_CONDITION_EFFECT[weather] + 0.10 * (1.0 if highway in {"residential", "service"} else 0.0))
    p_high_demand = _clip(base_prior + _TIME_DEMAND_EFFECT[time_of_day])
    p_low_capacity = _clip(base_prior * 0.55 + _speed_capacity_penalty(maxspeed) + 0.20 * p_bad_condition)
    p_high_accident = _clip(0.08 + 0.35 * p_bad_condition + 0.15 * base_prior + 0.50 * _length_exposure_penalty(length_m))

    # Noisy-OR: nếu một trong các yếu tố demand/capacity/accident cao, nguy cơ tắc sẽ tăng.
    p_congestion = 1.0 - (
        (1.0 - 0.88 * p_high_demand)
        * (1.0 - 0.82 * p_low_capacity)
        * (1.0 - 0.76 * p_high_accident)
    )

    return _clip(p_congestion)


def compute_congestion(edges: pd.DataFrame, weather: str, time_of_day: str) -> pd.DataFrame:
    """
    Tính xác suất kẹt xe cho từng cạnh.

    Args:
        edges (pd.DataFrame): Cần có tối thiểu các cột u, v, length, maxspeed, highway.
        weather (str): "rain" hoặc "clear".
        time_of_day (str): "peak" hoặc "normal".

    Returns:
        pd.DataFrame: DataFrame gồm các cột:
            - u (int)
            - v (int)
            - p_congestion (float, trong [0, 1])
    """
    required_columns = {"u", "v", "length", "maxspeed", "highway"}
    missing = required_columns.difference(edges.columns)
    if missing:
        raise KeyError(f"edges đang thiếu các cột bắt buộc: {sorted(missing)}")

    weather_norm = _normalize_weather(weather)
    time_norm = _normalize_time_of_day(time_of_day)

    working = edges.copy()
    working["highway_norm"] = working["highway"].apply(_normalize_highway)
    working["maxspeed_norm"] = working["maxspeed"].apply(_safe_maxspeed)
    working["length_norm"] = pd.to_numeric(working["length"], errors="coerce").fillna(100.0)

    working["p_congestion"] = working.apply(
        lambda row: _infer_probabilities(
            highway=row["highway_norm"],
            maxspeed=float(row["maxspeed_norm"]),
            length_m=float(row["length_norm"]),
            weather=weather_norm,
            time_of_day=time_norm,
        ),
        axis=1,
    )

    result = working[["u", "v", "p_congestion"]].copy()
    result["p_congestion"] = result["p_congestion"].astype(float).round(4)
    return result


__all__ = ["compute_congestion"]
