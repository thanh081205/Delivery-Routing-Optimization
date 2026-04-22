from typing import List, Dict, Tuple
import networkx as nx
import pandas as pd
import math

class MapGraph:
    """Quản lý bản đồ và cung cấp các hàm tra cứu cho thuật toán A* và Bayes."""
    
    def __init__(self, graph_data: dict):
        # Lưu trữ nguyên bản
        self.G = graph_data["G"].copy() 
        self.nodes_df = graph_data["nodes"]
        self.edges_df = graph_data["edges"]
        
        # Tối ưu hóa tra cứu tọa độ O(1) cho hàm Heuristic
        # Trả về dict dạng: {node_id: (y_lat, x_lon)}
        self.node_coords = self.nodes_df.set_index('osmid')[['y', 'x']].apply(tuple, axis=1).to_dict()
        
    def update_edge_weights(self, weighted_edges: pd.DataFrame):
        """
        Cập nhật trọng số thời gian từ ML vào đồ thị NetworkX.
        Input: weighted_edges (có các cột 'u', 'v', 'travel_time_min')
        """
        # Chuyển DataFrame thành dictionary dạng {(u, v): travel_time_min}
        # Lưu ý: OSMnx graph là MultiDiGraph nên có key cạnh (thường là 0)
        weight_dict = {}
        for _, row in weighted_edges.iterrows():
            u, v = int(row['u']), int(row['v'])
            time_min = float(row['travel_time_min'])
            
            # Gán cho cạnh đầu tiên giữa u và v (key=0)
            weight_dict[(u, v, 0)] = {'travel_time_min': time_min}
            
        # Cập nhật hàng loạt vào đồ thị G với độ phức tạp tối ưu
        nx.set_edge_attributes(self.G, weight_dict)

    def apply_logic_filter(self, cleaned_graph: nx.MultiDiGraph):
        """Thay thế đồ thị hiện tại bằng đồ thị đã lọc các đường vi phạm từ TV4."""
        self.G = cleaned_graph

    def get_heuristic_distance(self, node_a: int, node_b: int, max_speed_kmh: float = 120.0) -> float:
        """
        Tính Heuristic: Thời gian ngắn nhất (phút) theo đường chim bay giữa 2 node.
        Dùng công thức Haversine để tính khoảng cách thực tế trên bề mặt Trái Đất.
        Đây là Admissible Heuristic (luôn <= thời gian thực tế) giúp A* chạy đúng.
        """
        coord_a = self.node_coords.get(node_a)
        coord_b = self.node_coords.get(node_b)

        if not coord_a or not coord_b:
            return 0.0  # Fallback an toàn nếu lỗi data

        R = 6371000.0  # Bán kính Trái Đất (mét)

        lat1, lon1 = math.radians(coord_a[0]), math.radians(coord_a[1])
        lat2, lon2 = math.radians(coord_b[0]), math.radians(coord_b[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        dist_m = R * c

        # Đổi ra thời gian (phút) = (Quãng đường / Vận tốc tối đa)
        max_speed_m_per_min = max_speed_kmh * 1000.0 / 60.0
        return dist_m / max_speed_m_per_min


class DeliveryVehicle:
    """Quản lý trạng thái (State) của xe giao hàng."""
    
    def __init__(self, start_node: int, start_time: float, capacity: float):
        self.current_location = start_node
        self.current_time = start_time  # Tính bằng phút (VD: 8h sáng = 480.0)
        self.capacity = capacity
        
        # Danh sách điểm giao và khung giờ: {node_id: (start_min, end_min)}
        self.deliveries: Dict[int, Tuple[float, float]] = {} 
        self.route_history: List[int] = [start_node]
        
    def add_delivery_point(self, node_id: int, time_window: Tuple[float, float]):
        """Thêm một điểm cần giao vào lộ trình cùng với khung giờ yêu cầu."""
        self.deliveries[node_id] = time_window
        
    def update_state(self, new_location: int, time_spent: float):
        """Cập nhật vị trí và thời gian sau khi di chuyển hoàn tất một chặng."""
        self.current_location = new_location
        self.current_time += time_spent
        self.route_history.append(new_location)
        
    def pop_delivery(self, node_id: int):
        """Xóa điểm giao hàng khỏi danh sách sau khi đã hoàn thành."""
        if node_id in self.deliveries:
            del self.deliveries[node_id]