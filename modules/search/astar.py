import networkx as nx
import pandas as pd
from itertools import permutations
import math
from modules.graph.core_system import MapGraph

def run_astar(cleaned_graph: nx.MultiDiGraph, weighted_edges: pd.DataFrame, origin: int, destinations: list[int], time_windows: dict, start_time: float) -> dict:
    # Đồng bộ hóa với core_system bằng cách sử dụng MapGraph
    nodes_list = []
    for node_id, data in cleaned_graph.nodes(data=True):
        nodes_list.append({"osmid": node_id, "x": data.get("x", 0.0), "y": data.get("y", 0.0)})
    nodes_df = pd.DataFrame(nodes_list)
    
    graph_data = {
        "G": cleaned_graph,
        "nodes": nodes_df,
        "edges": pd.DataFrame() # Không cần thiết cho heuristic
    }
    
    # Khởi tạo MapGraph
    map_graph = MapGraph(graph_data)
    
    # Cập nhật trọng số thông qua hàm đồng bộ của hệ thống
    map_graph.update_edge_weights(weighted_edges)

    def get_weight(u, v, d):
        # Trong MultiDiGraph, d chứa danh sách các cạnh (theo key). Lấy min travel_time_min.
        if not d:
            return 9999.0
        return min(edge_attr.get('travel_time_min', 9999.0) for edge_attr in d.values())

    # Tính toán ma trận khoảng cách giữa các node quan trọng (Origin + Destinations)
    pois = [origin] + destinations
    dist_matrix = {}
    path_matrix = {}
    
    for u in pois:
        dist_matrix[u] = {}
        path_matrix[u] = {}
        for v in pois:
            if u == v:
                dist_matrix[u][v] = 0.0
                path_matrix[u][v] = [u]
            else:
                try:
                    # Chạy A* với đồ thị của MapGraph, heuristic của hệ thống
                    path = nx.astar_path(map_graph.G, u, v, heuristic=map_graph.get_heuristic_distance, weight=get_weight)
                    # Tính tổng chi phí (chọn cạnh ngắn nhất giữa u và v trong MultiDiGraph)
                    cost = sum(get_weight(path[i], path[i+1], map_graph.G[path[i]][path[i+1]]) for i in range(len(path)-1))
                    dist_matrix[u][v] = cost
                    path_matrix[u][v] = path
                except nx.NetworkXNoPath:
                    dist_matrix[u][v] = float('inf')
                    path_matrix[u][v] = []

    # Tìm hoán vị tối ưu của các điểm đến (CSP Time Windows)
    best_time = float('inf')
    best_route_full = []
    best_visited_order = []
    
    for perm in permutations(destinations):
        current_time = start_time
        current_node = origin
        valid_permutation = True
        full_route = [origin]
        
        for dest in perm:
            travel_t = dist_matrix[current_node][dest]
            if travel_t == float('inf'):
                valid_permutation = False
                break
                
            arrival_time = current_time + travel_t
            
            # --- CSP: Kiểm tra ràng buộc Time Window ---
            if dest in time_windows:
                tw_start, tw_end = time_windows[dest]
                # Nếu đến quá sớm, chờ đến tw_start
                if arrival_time < tw_start:
                    arrival_time = tw_start
                # Nếu đến trễ quá tw_end, hoán vị này không hợp lệ
                elif arrival_time > tw_end:
                    valid_permutation = False
                    break
            # -------------------------------------------
            
            # Kết nối path
            segment_path = path_matrix[current_node][dest]
            # Loại bỏ nút đầu tiên của segment để không bị lặp nút trong full_route
            full_route.extend(segment_path[1:])
            
            current_time = arrival_time
            current_node = dest
            
        if valid_permutation and current_time < best_time:
            best_time = current_time
            best_route_full = full_route
            best_visited_order = list(perm)

    if best_time == float('inf'):
        return {
            "route": [],
            "total_time_min": float('inf'),
            "visited_order": []
        }

    return {
        "route": best_route_full,
        "total_time_min": best_time - start_time, # Chỉ trả về duration 
        "visited_order": best_visited_order
    }
