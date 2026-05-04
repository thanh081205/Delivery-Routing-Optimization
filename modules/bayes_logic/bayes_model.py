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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from typing import Any

import networkx as nx
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


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _coerce_edge_id(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return value


def _edge_id(u: Any, v: Any, key: Any) -> tuple[Any, Any, Any]:
    return (_coerce_edge_id(u), _coerce_edge_id(v), _coerce_edge_id(key))


def _edge_metadata_lookup(edges: Any) -> dict[tuple[Any, Any, Any], dict[str, Any]]:
    if edges is None or not hasattr(edges, "iterrows") or not hasattr(edges, "columns"):
        return {}

    if not {"u", "v", "key"}.issubset(set(edges.columns)):
        return {}

    lookup: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for _, row in edges.iterrows():
        lookup[_edge_id(row["u"], row["v"], row["key"])] = {
            column: row[column]
            for column in edges.columns
            if column not in {"u", "v", "key"} and not _is_missing_value(row[column])
        }

    return lookup


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


def _edge_attr(data: dict[str, Any], fallback: dict[str, Any], field: str, default: Any) -> Any:
    value = data.get(field)
    if _is_missing_value(value):
        value = fallback.get(field, default)
    if _is_missing_value(value):
        return default
    return value


def _edge_dataframe_from_graph(graph: nx.Graph, supplemental_edges: Any = None) -> pd.DataFrame:
    if not isinstance(graph, (nx.DiGraph, nx.MultiDiGraph)):
        raise TypeError("graph phải là nx.DiGraph hoặc nx.MultiDiGraph.")

    metadata = _edge_metadata_lookup(supplemental_edges)
    records: list[dict[str, Any]] = []

    if isinstance(graph, nx.MultiDiGraph):
        edge_iter = graph.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, 0, data) for u, v, data in graph.edges(data=True))

    for u, v, key, data in edge_iter:
        fallback = metadata.get(_edge_id(u, v, key), {})
        records.append(
            {
                "u": u,
                "v": v,
                "key": key,
                "length": _edge_attr(data, fallback, "length", 100.0),
                "maxspeed": _edge_attr(data, fallback, "maxspeed", 40.0),
                "highway": _edge_attr(data, fallback, "highway", "unclassified"),
            }
        )

    return pd.DataFrame(records, columns=["u", "v", "key", "length", "maxspeed", "highway"])


def _edge_dataframe_from_source(source: Any) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, dict):
        if "G" in source:
            return _edge_dataframe_from_graph(source["G"], supplemental_edges=source.get("edges"))
        if "edges" in source:
            return source["edges"].copy()
        raise KeyError("graph_data phải chứa khóa 'G' hoặc 'edges'.")

    if isinstance(source, (nx.DiGraph, nx.MultiDiGraph)):
        return _edge_dataframe_from_graph(source)

    raise TypeError("source phải là graph_data, nx.Graph hoặc pd.DataFrame edges.")


def _node_ids_from_source(source: Any, edges: pd.DataFrame) -> list[Any]:
    graph = source.get("G") if isinstance(source, dict) else source
    if isinstance(graph, (nx.DiGraph, nx.MultiDiGraph)):
        return list(graph.nodes())

    if edges.empty:
        return []

    return list(pd.concat([edges["u"], edges["v"]]).drop_duplicates())


def _observed_at_iso(observed_at: datetime | str | None) -> str:
    if observed_at is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(observed_at, datetime):
        return observed_at.isoformat()
    return str(observed_at)


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


def compute_graph_congestion(
    source: Any,
    weather: str,
    time_of_day: str,
    observed_at: datetime | str | None = None,
) -> pd.DataFrame:
    """
    Compute congestion probabilities for every edge in a graph-like source.

    Args:
        source: graph_data dict, nx.MultiDiGraph/nx.DiGraph, or edges DataFrame.
        weather: real-time weather state ("rain" or "clear", aliases accepted).
        time_of_day: real-time demand state ("peak" or "normal", aliases accepted).
        observed_at: optional timestamp for the real-time update.

    Returns:
        DataFrame with columns [u, v, key, p_congestion, weather, time_of_day, observed_at].
    """
    weather_norm = _normalize_weather(weather)
    time_norm = _normalize_time_of_day(time_of_day)
    timestamp = _observed_at_iso(observed_at)

    edges = _edge_dataframe_from_source(source)
    congestion_df = compute_congestion(edges, weather=weather_norm, time_of_day=time_norm)
    congestion_df["weather"] = weather_norm
    congestion_df["time_of_day"] = time_norm
    congestion_df["observed_at"] = timestamp
    return congestion_df


def compute_congestion_matrix(
    source: Any,
    weather: str,
    time_of_day: str,
    observed_at: datetime | str | None = None,
    fill_value: float = 0.0,
    parallel_edge_strategy: str = "max",
) -> pd.DataFrame:
    """
    Return a node-by-node congestion probability matrix for the whole graph.

    Parallel edges are collapsed into one matrix cell. The exact edge-level
    probabilities, including edge keys, are stored in matrix.attrs["edge_probabilities"].
    """
    if parallel_edge_strategy not in {"max", "mean", "min"}:
        raise ValueError("parallel_edge_strategy phải là 'max', 'mean' hoặc 'min'.")

    edges = _edge_dataframe_from_source(source)
    node_ids = _node_ids_from_source(source, edges)
    edge_probabilities = compute_graph_congestion(
        edges,
        weather=weather,
        time_of_day=time_of_day,
        observed_at=observed_at,
    )

    matrix = pd.DataFrame(
        fill_value,
        index=pd.Index(node_ids, name="u"),
        columns=pd.Index(node_ids, name="v"),
        dtype=float,
    )

    if not edge_probabilities.empty:
        grouped = edge_probabilities.groupby(["u", "v"])["p_congestion"].agg(parallel_edge_strategy)
        for (u, v), probability in grouped.items():
            if u in matrix.index and v in matrix.columns:
                matrix.loc[u, v] = round(float(probability), 4)

    matrix.attrs["edge_probabilities"] = edge_probabilities
    matrix.attrs["weather"] = _normalize_weather(weather)
    matrix.attrs["time_of_day"] = _normalize_time_of_day(time_of_day)
    matrix.attrs["observed_at"] = edge_probabilities["observed_at"].iloc[0] if not edge_probabilities.empty else _observed_at_iso(observed_at)
    matrix.attrs["parallel_edge_strategy"] = parallel_edge_strategy
    matrix.attrs["fill_value"] = fill_value
    return matrix


@dataclass
class BayesCongestionModel:
    """Reusable real-time wrapper for TV4 Bayes congestion inference."""

    source: Any | None = None
    default_weather: str = "clear"
    default_time_of_day: str = "normal"
    last_matrix: pd.DataFrame | None = field(init=False, default=None)
    last_edge_probabilities: pd.DataFrame | None = field(init=False, default=None)

    def update_realtime(
        self,
        source: Any | None = None,
        weather: str | None = None,
        time_of_day: str | None = None,
        observed_at: datetime | str | None = None,
    ) -> pd.DataFrame:
        graph_source = self.source if source is None else source
        if graph_source is None:
            raise ValueError("Cần truyền source hoặc khởi tạo BayesCongestionModel(source=...).")

        matrix = compute_congestion_matrix(
            graph_source,
            weather=weather or self.default_weather,
            time_of_day=time_of_day or self.default_time_of_day,
            observed_at=observed_at,
        )
        self.source = graph_source
        self.last_matrix = matrix
        self.last_edge_probabilities = matrix.attrs["edge_probabilities"].copy()
        return matrix

    def as_feature_frame(self) -> pd.DataFrame:
        if self.last_edge_probabilities is None:
            raise RuntimeError("Chưa có kết quả. Hãy gọi update_realtime() trước.")
        return self.last_edge_probabilities.copy()


__all__ = [
    "BayesCongestionModel",
    "compute_congestion",
    "compute_congestion_matrix",
    "compute_graph_congestion",
]
