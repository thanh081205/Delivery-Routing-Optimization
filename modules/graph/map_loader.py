"""
map_loader.py — Module tải và xử lý bản đồ đường bộ từ OSMnx
TV1 - Hệ thống & Tích hợp
"""

import osmnx as ox
import networkx as nx
import pandas as pd
import folium


def load_map(latitude: float, longitude: float, dist: int = 1000) -> dict:
    """
    Tải bản đồ mạng lưới đường bộ từ OpenStreetMap xung quanh một tọa độ GPS.

    Args:
        latitude  (float): Vĩ độ của điểm trung tâm.
        longitude (float): Kinh độ của điểm trung tâm.
        dist      (int):   Bán kính tải bản đồ tính bằng mét (mặc định: 1000m).

    Returns:
        graph_data (dict): {
            "G"    : nx.MultiDiGraph  — Đồ thị NetworkX gốc từ osmnx,
            "nodes": pd.DataFrame    — Thông tin các nút (osmid, x, y),
            "edges": pd.DataFrame    — Thông tin các cạnh (u, v, length, maxspeed, highway)
        }
    """

    print(f"Đang tải bản đồ tại tọa độ ({latitude}, {longitude}), bán kính {dist}m...")

    # Bước 1: Tải đồ thị từ OSMnx
    G = ox.graph_from_point((latitude, longitude), dist=dist, network_type="drive")
    print(f"✅ Tải xong! Số nút: {len(G.nodes)}, Số cạnh: {len(G.edges)}")

    # Bước 2: Chuyển thành DataFrame
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    # Bước 3: Lọc các cột cần thiết
    nodes = nodes[["x", "y"]].reset_index()
    edges = edges.reset_index()[["u", "v", "key", "length", "maxspeed", "highway"]]

    # Bước 4: Điền giá trị mặc định cho các ô bị thiếu
    edges["maxspeed"] = edges["maxspeed"].fillna("40")
    edges["maxspeed"] = edges["maxspeed"].apply(
        lambda x: float(str(x).split()[0]) if isinstance(x, str) else 40.0
    )
    edges["highway"] = edges["highway"].fillna("unclassified")

    return {
        "G"    : G,
        "nodes": nodes,
        "edges": edges
    }


def visualize_map(graph_data: dict, latitude: float, longitude: float, zoom: int = 15) -> folium.Map:
    """
    Hiển thị bản đồ đường bộ lên bản đồ tương tác Folium.

    Args:
        graph_data (dict):   Output từ hàm load_map().
        latitude   (float):  Vĩ độ tâm bản đồ.
        longitude  (float):  Kinh độ tâm bản đồ.
        zoom       (int):    Mức zoom ban đầu (mặc định: 15).

    Returns:
        m (folium.Map): Bản đồ tương tác có thể hiển thị trong Colab.
    """

    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    # Tạo bản đồ nền
    m = folium.Map(location=[latitude, longitude], zoom_start=zoom)

    # Vẽ các đoạn đường lên bản đồ
    node_coords = nodes.set_index('osmid')[['y', 'x']].apply(tuple, axis=1).to_dict()
    for _, row in edges.iterrows():
        u_coord = node_coords.get(row["u"])
        v_coord = node_coords.get(row["v"])
        if u_coord and v_coord:
            folium.PolyLine([u_coord, v_coord], color="blue", weight=2, opacity=0.5).add_to(m)

    # Đánh dấu vị trí trung tâm
    folium.Marker(
        location=[latitude, longitude],
        popup="ĐH Bách Khoa - Lý Thường Kiệt",
        icon=folium.Icon(color="red", icon="university", prefix="fa")
    ).add_to(m)

    return m


# --- Chạy thử trực tiếp (chỉ dùng khi test, không dùng khi import) ---
if __name__ == "__main__":
    LAT  = 10.7729
    LON  = 106.6578
    DIST = 1000

    graph_data = load_map(LAT, LON, DIST)

    print("\n✅ Nodes (5 dòng đầu):")
    print(graph_data["nodes"].head())

    print("\n✅ Edges (5 dòng đầu):")
    print(graph_data["edges"].head())