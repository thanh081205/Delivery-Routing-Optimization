"""
logic_filter.py — Module lọc đồ thị theo luật logic giao thông

Module này nhận graph_data từ TV1, duyệt qua đồ thị NetworkX gốc và xóa các cạnh
không hợp lệ trước khi chuyển sang A* của TV2.

Thiết kế bám theo interface trong INTERFACES.md:
    filter_graph(graph_data: dict, vehicle_weight: float) -> nx.MultiDiGraph
"""

from __future__ import annotations

import re
from typing import Any

import networkx as nx


_RESTRICTED_VALUES = {
    "no",
    "private",
    "restricted",
    "delivery",
    "forestry",
    "agricultural",
}

_CLOSED_HIGHWAY_VALUES = {
    "construction",
    "proposed",
    "abandoned",
    "planned",
    "razed",
    "demolished",
}


def _as_list(value: Any) -> list[Any]:
    """Chuẩn hóa giá trị bất kỳ thành list để xử lý thống nhất."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _normalize_token(value: Any) -> str:
    """Chuẩn hóa một token về chuỗi lower-case, bỏ khoảng trắng dư."""
    return str(value).strip().lower()


def _extract_first_float(value: Any) -> float | None:
    """
    Trích số thực đầu tiên từ dữ liệu OSM.

    Hỗ trợ các dạng phổ biến như:
    - 7.5
    - "7.5"
    - "7.5 t"
    - "7,5 tons"
    - ["7.5", "10"]
    """
    for item in _as_list(value):
        if isinstance(item, (int, float)):
            return float(item)

        text = str(item).strip().lower().replace(",", ".")
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                continue
    return None


def _has_restricted_access(edge_data: dict[str, Any]) -> bool:
    """
    Kiểm tra các thuộc tính hạn chế truy cập thường gặp trong OSM.

    Các tag thường dùng:
    - access
    - motor_vehicle
    - vehicle
    """
    for field in ("access", "motor_vehicle", "vehicle"):
        for raw_value in _as_list(edge_data.get(field)):
            if _normalize_token(raw_value) in _RESTRICTED_VALUES:
                return True
    return False


def _violates_weight_limit(edge_data: dict[str, Any], vehicle_weight: float) -> bool:
    """Kiểm tra xe có vượt giới hạn tải trọng của cạnh hay không."""
    maxweight = _extract_first_float(edge_data.get("maxweight"))
    if maxweight is None:
        return False
    return vehicle_weight > maxweight


def _is_closed_or_under_construction(edge_data: dict[str, Any]) -> bool:
    """Phát hiện các cạnh đang đóng, bị loại bỏ, hoặc đang xây dựng."""
    highway_values = [_normalize_token(v) for v in _as_list(edge_data.get("highway"))]
    if any(v in _CLOSED_HIGHWAY_VALUES for v in highway_values):
        return True

    if _normalize_token(edge_data.get("construction")) not in {"", "none"}:
        return True

    return False


def _has_invalid_length(edge_data: dict[str, Any]) -> bool:
    """Loại cạnh có độ dài không hợp lệ hoặc thiếu dữ liệu nghiêm trọng."""
    length = edge_data.get("length")
    try:
        return length is None or float(length) <= 0
    except (TypeError, ValueError):
        return True


def filter_graph(graph_data: dict, vehicle_weight: float) -> nx.MultiDiGraph:
    """
    Lọc đồ thị theo các luật logic giao thông.

    Args:
        graph_data (dict): Output từ TV1, tối thiểu phải có khóa "G".
        vehicle_weight (float): Tải trọng xe tính theo tấn.

    Returns:
        nx.MultiDiGraph: Đồ thị đã xóa các cạnh vi phạm luật.

    Luật v1 đang áp dụng:
    1. Cạnh có access / vehicle / motor_vehicle bị cấm -> xóa.
    2. Cạnh có giới hạn tải trọng và xe vượt ngưỡng -> xóa.
    3. Cạnh đóng / construction / proposed -> xóa.
    4. Cạnh có length không hợp lệ -> xóa.

    Ghi chú:
    - Hàm làm việc trực tiếp trên graph_data["G"], không chỉ dựa vào DataFrame edges,
      vì metadata đầy đủ của OSM thường nằm trong edge attributes của NetworkX graph.
    - Hàm lưu thống kê vào cleaned_graph.graph["logic_filter_stats"] để tiện debug.
    """
    if "G" not in graph_data:
        raise KeyError("graph_data phải chứa khóa 'G'.")

    if not isinstance(vehicle_weight, (int, float)):
        raise TypeError("vehicle_weight phải là số (int hoặc float).")

    original_graph = graph_data["G"]
    if not isinstance(original_graph, nx.MultiDiGraph):
        raise TypeError("graph_data['G'] phải là nx.MultiDiGraph.")

    cleaned_graph = original_graph.copy()

    stats = {
        "removed_invalid_length": 0,
        "removed_restricted_access": 0,
        "removed_weight_limit": 0,
        "removed_closed_or_construction": 0,
        "removed_total": 0,
        "remaining_edges": cleaned_graph.number_of_edges(),
        "vehicle_weight_ton": float(vehicle_weight),
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
