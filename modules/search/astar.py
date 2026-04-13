import networkx as nx
import pandas as pd
from itertools import permutations
import math

def run_astar(cleaned_graph: nx.MultiDiGraph, weighted_edges: pd.DataFrame, origin: int, destinations: list[int], time_windows: dict, start_time: float) -> dict:
    weight_dict = {}
    for _, row in weighted_edges.iterrows():
        u = int(row['u'])
        v = int(row['v'])
        t = float(row['travel_time_min'])
        if (u, v) not in weight_dict or t < weight_dict[(u, v)]:
            weight_dict[(u, v)] = t
            
    # Tạo một DiGraph đơn giản từ MultiDiGraph để chạy A* dễ hơn
    G_simple = nx.DiGraph()
    for u, v, data in cleaned_graph.edges(data=True):
        if (u, v) in weight_dict:
            w = weight_dict[(u, v)]
            if G_simple.has_edge(u, v):
                G_simple[u][v]['weight'] = min(G_simple[u][v]['weight'], w)
            else:
                G_simple.add_edge(u, v, weight=w)
        else:
            # Nếu không có trong weighted_edges, giả sử một trọng số dự phòng 
            w = data.get('length', 100) / (data.get('maxspeed', 30) * 1000 / 60) if isinstance(data.get('maxspeed'), (int, float)) else 9999
            if not G_simple.has_edge(u, v):
                G_simple.add_edge(u, v, weight=w)

    def heuristic(n1, n2):
        return 0 

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
                    # Chạy A* cho từng cặp
                    path = nx.astar_path(G_simple, u, v, heuristic=heuristic, weight='weight')
                    # Tính tổng thời gian
                    cost = sum(G_simple[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
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
            
            # --- CSP: Kiểm tra ráng buộc Time Window ---
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
