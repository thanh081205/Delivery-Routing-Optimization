from __future__ import annotations

import re
from typing import Any

import networkx as nx


_ACCESS_FIELDS = (
    "access",
    "vehicle",
    "motor_vehicle",
    "motorcar",
    "hgv",
    "goods",
)

_DELIVERY_ALLOW_VALUES = {
    "yes",
    "permissive",
    "designated",
    "delivery",
}

_RESTRICTED_ACCESS_VALUES = {
    "no",
    "private",
    "restricted",
    "forestry",
    "agricultural",
    "customers",
    "permit",
    "emergency",
    "official",
    "bus",
    "psv",
}

_CLOSED_HIGHWAY_VALUES = {
    "construction",
    "proposed",
    "abandoned",
    "planned",
    "razed",
    "demolished",
    "disused",
}

_INACTIVE_VALUES = {
    "yes",
    "true",
    "1",
}

_WEIGHT_LIMIT_FIELDS = (
    "maxweight",
    "maxgcweight",
)

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


def _metadata_from_edges_dataframe(edges: Any) -> dict[tuple[Any, Any, Any], dict[str, Any]]:
    """
    Read normalized edge attributes from graph_data["edges"] when TV1 provides them.
    This keeps the logic rules usable even if the NetworkX edge is missing a field.
    """
    if edges is None or not hasattr(edges, "iterrows") or not hasattr(edges, "columns"):
        return {}

    columns = set(edges.columns)
    if not {"u", "v", "key"}.issubset(columns):
        return {}

    metadata: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for _, row in edges.iterrows():
        edge_key = _edge_id(row["u"], row["v"], row["key"])
        metadata[edge_key] = {
            column: row[column]
            for column in edges.columns
            if column not in {"u", "v", "key"} and not _is_missing_value(row[column])
        }

    return metadata


def _sync_edge_metadata(cleaned_graph: nx.MultiDiGraph, graph_data: dict) -> int:
    dataframe_metadata = _metadata_from_edges_dataframe(graph_data.get("edges"))
    if not dataframe_metadata:
        return 0

    synced_edges = 0
    for u, v, key, data in cleaned_graph.edges(keys=True, data=True):
        row_data = dataframe_metadata.get(_edge_id(u, v, key))
        if not row_data:
            continue

        for field, value in row_data.items():
            if field not in data or _is_missing_value(data[field]):
                data[field] = value
        synced_edges += 1

    return synced_edges


def _as_list(value: Any) -> list[Any]:
    """Chuẩn hóa giá trị OSM thành list để xử lý đồng nhất."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _normalize_token(value: Any) -> str:
    return str(value).strip().lower()


def _tag_tokens(edge_data: dict[str, Any], field: str) -> list[str]:
    return [_normalize_token(value) for value in _as_list(edge_data.get(field))]


def _has_delivery_override(edge_data: dict[str, Any]) -> bool:
    for field in ("delivery", "hgv", "goods"):
        if any(token in _DELIVERY_ALLOW_VALUES for token in _tag_tokens(edge_data, field)):
            return True
    return False


def _has_restricted_access(edge_data: dict[str, Any]) -> bool:
    """
    IF access/vehicle/motor_vehicle/hgv bị cấm THEN loại cạnh.

    Với xe giao hàng, các tag explicit như delivery=yes hoặc access=delivery
    được xem là hợp lệ.
    """
    if _has_delivery_override(edge_data):
        return False

    for field in _ACCESS_FIELDS:
        for token in _tag_tokens(edge_data, field):
            if token in _DELIVERY_ALLOW_VALUES:
                continue
            if token in _RESTRICTED_ACCESS_VALUES:
                return True
    return False


def _parse_weight_tons(value: Any) -> float | None:
    """
    Trích giới hạn tải trọng từ tag OSM và quy đổi về tấn.

    Ví dụ hợp lệ: 7.5, "7.5 t", "7500 kg", "16000 lbs".
    Các giá trị như "none", "default", "unknown" được bỏ qua.
    """
    for item in _as_list(value):
        if isinstance(item, (int, float)):
            if float(item) != float(item):
                continue
            return float(item)

        text = str(item).strip().lower().replace(",", ".")
        if text in {"", "none", "no", "default", "unknown", "unsigned"}:
            continue

        match = _NUMBER_RE.search(text)
        if not match:
            continue

        try:
            number = float(match.group(0))
        except ValueError:
            continue

        if "kg" in text:
            return number / 1000.0
        if "lb" in text:
            return number * 0.00045359237
        return number

    return None


def _violates_weight_limit(edge_data: dict[str, Any], vehicle_weight: float) -> bool:
    """
    IF vehicle_weight > maxweight THEN loại cạnh.
    """
    limits = [
        limit
        for field in _WEIGHT_LIMIT_FIELDS
        if (limit := _parse_weight_tons(edge_data.get(field))) is not None
    ]
    if not limits:
        return False
    return vehicle_weight > min(limits)


def _is_closed_or_under_construction(edge_data: dict[str, Any]) -> bool:
    """
    IF highway/construction/disused/abandoned thể hiện cạnh không hoạt động
    THEN loại cạnh.
    """
    highway_values = _tag_tokens(edge_data, "highway")
    if any(value in _CLOSED_HIGHWAY_VALUES for value in highway_values):
        return True

    construction_values = _tag_tokens(edge_data, "construction")
    if any(value not in {"", "no", "none", "false", "0"} for value in construction_values):
        return True

    for field in ("disused", "abandoned"):
        if any(value in _INACTIVE_VALUES for value in _tag_tokens(edge_data, field)):
            return True

    return False


def _has_invalid_length(edge_data: dict[str, Any]) -> bool:
    """
    IF length thiếu hoặc <= 0 THEN loại cạnh.
    """
    if "length" not in edge_data or _is_missing_value(edge_data.get("length")):
        return True

    try:
        length = float(edge_data["length"])
        return length != length or length <= 0
    except (TypeError, ValueError):
        return True


def filter_graph(graph_data: dict, vehicle_weight: float) -> nx.MultiDiGraph:
    """
    Lọc đồ thị theo các luật logic giao thông 

    Args:
        graph_data: Output từ TV1, tối thiểu phải có khóa "G".
        vehicle_weight: Tải trọng xe, đơn vị tấn.

    Returns:
        nx.MultiDiGraph đã xóa các cạnh vi phạm luật.
    """
    if "G" not in graph_data:
        raise KeyError("graph_data phải chứa khóa 'G'.")

    if not isinstance(vehicle_weight, (int, float)):
        raise TypeError("vehicle_weight phải là số (int hoặc float).")
    if float(vehicle_weight) < 0:
        raise ValueError("vehicle_weight không được âm.")

    original_graph = graph_data["G"]
    if not isinstance(original_graph, nx.MultiDiGraph):
        raise TypeError("graph_data['G'] phải là nx.MultiDiGraph.")

    cleaned_graph = original_graph.copy()
    synced_edges = _sync_edge_metadata(cleaned_graph, graph_data)
    stats = {
        "original_edges": cleaned_graph.number_of_edges(),
        "removed_invalid_length": 0,
        "removed_restricted_access": 0,
        "removed_weight_limit": 0,
        "removed_closed_or_construction": 0,
        "removed_total": 0,
        "remaining_edges": cleaned_graph.number_of_edges(),
        "vehicle_weight_ton": float(vehicle_weight),
        "metadata_edges_synced": synced_edges,
        "oneway_rule": "handled_by_directed_osmnx_graph",
    }

    edges_to_remove: list[tuple[Any, Any, Any]] = []

    for u, v, key, data in cleaned_graph.edges(keys=True, data=True):
        if _has_invalid_length(data):
            edges_to_remove.append((u, v, key))
            stats["removed_invalid_length"] += 1
            continue

        if _has_restricted_access(data):
            edges_to_remove.append((u, v, key))
            stats["removed_restricted_access"] += 1
            continue

        if _violates_weight_limit(data, float(vehicle_weight)):
            edges_to_remove.append((u, v, key))
            stats["removed_weight_limit"] += 1
            continue

        if _is_closed_or_under_construction(data):
            edges_to_remove.append((u, v, key))
            stats["removed_closed_or_construction"] += 1
            continue

    cleaned_graph.remove_edges_from(edges_to_remove)

    stats["removed_total"] = len(edges_to_remove)
    stats["remaining_edges"] = cleaned_graph.number_of_edges()
    cleaned_graph.graph["logic_filter_stats"] = stats

    return cleaned_graph


__all__ = ["filter_graph"]
