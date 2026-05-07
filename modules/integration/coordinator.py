"""
coordinator.py — Module điều phối toàn cục (Integration)
Kết nối các module: Graph (TV1), Search (TV2), ML (TV3) và Bayes Logic (TV4)
"""

import pandas as pd
from typing import Dict, Any, Optional

# Import các thành phần từ các module con theo chuẩn kiến trúc đã chia
from modules.graph.core_system import MapGraph, DeliveryVehicle
from modules.bayes_logic.bayes_model import BayesCongestionModel
from modules.ml.travel_time_predictor import predict_travel_time
from modules.search.astar import run_astar


class DeliveryCoordinator:
    """
    Nhạc trưởng điều phối toàn bộ luồng hoạt động của hệ thống định tuyến giao hàng.
    """
    
    def __init__(self, map_graph: MapGraph):
        self.map_graph = map_graph
        # Khởi tạo mạng Bayes gắn với dữ liệu bản đồ hiện tại
        self.bayes_model = BayesCongestionModel(source=self.map_graph.G)
        
        # Lưu lại trạng thái môi trường hiện tại
        self.current_weather = "clear"
        self.current_time_of_day = "normal"

    def _update_environment_and_weights(self, weather: str, time_of_day: str):
        """Hàm nội bộ: Chạy TV4 và TV3 để cập nhật lại trọng số bản đồ."""
        self.current_weather = weather
        self.current_time_of_day = time_of_day
        
        # 1. Chạy TV4: Cập nhật mạng Bayes để lấy ma trận kẹt xe mới
        self.bayes_model.update_realtime(weather=weather, time_of_day=time_of_day)
        congestion_df = self.bayes_model.as_feature_frame()
        
        # 2. Chạy TV3: Nội suy thời gian di chuyển (Travel Time) từ Học máy
        new_weighted_edges = predict_travel_time(self.map_graph.edges_df, congestion_df)
        
        # 3. Chạy TV1: Cập nhật trọng số mới vào bản đồ NetworkX
        self.map_graph.update_edge_weights(new_weighted_edges)
        
        return new_weighted_edges

    def plan_initial_route(self, vehicle: DeliveryVehicle, weather: str, time_of_day: str) -> Dict[str, Any]:
        """Lập kế hoạch lộ trình ban đầu trước khi xe xuất phát."""
        print(f"\n🚀 ĐANG LẬP TRÌNH KHỞI HÀNH | Thời tiết: {weather} | Giờ: {time_of_day}")
        
        # Cập nhật môi trường và lấy trọng số cạnh
        weighted_edges = self._update_environment_and_weights(weather, time_of_day)
        
        # Lấy danh sách tất cả các điểm cần giao
        destinations = list(vehicle.deliveries.keys())
        
        # Chạy TV2: Thuật toán A* TSP
        plan = run_astar(
            cleaned_graph=self.map_graph.G,
            weighted_edges=weighted_edges,
            origin=vehicle.current_location,
            destinations=destinations,
            time_windows=vehicle.deliveries,
            start_time=vehicle.current_time
        )
        return plan

    def trigger_replanning(self, vehicle: DeliveryVehicle, new_weather: str, new_time_of_day: str) -> Optional[Dict[str, Any]]:
        """
        Kích hoạt tính toán lại toàn bộ lộ trình khi môi trường (thời tiết/kẹt xe) thay đổi.
        """
        print(f"\n⚠️ CẢNH BÁO MÔI TRƯỜNG THAY ĐỔI | Thời tiết mới: {new_weather} | Giờ: {new_time_of_day}")
        print(f"📍 Vị trí hiện tại: Node {vehicle.current_location} | ⏰ Đồng hồ xe: {vehicle.current_time} phút")
        
        # Lấy danh sách các điểm chưa giao
        remaining_destinations = list(vehicle.deliveries.keys())
        if not remaining_destinations:
            print("✅ Xe đã hoàn thành tất cả các điểm giao. Không cần Re-plan.")
            return None
        
        print(f"📦 Các điểm còn lại cần giao: {remaining_destinations}")

        # Cập nhật môi trường (Kéo theo việc chạy lại mạng Bayes và Machine Learning)
        print("🔄 Đang phân tích mức độ kẹt xe & nội suy lại thời gian di chuyển...")
        weighted_edges = self._update_environment_and_weights(new_weather, new_time_of_day)

        # Chạy thuật toán tìm đường mới từ vị trí hiện tại
        print("🔄 Đang tìm lại lộ trình tối ưu (A* Re-routing)...")
        new_plan = run_astar(
            cleaned_graph=self.map_graph.G,
            weighted_edges=weighted_edges,
            origin=vehicle.current_location,           # Xuất phát từ chỗ xe đang đứng
            destinations=remaining_destinations,       # Chỉ xét các điểm chưa giao
            time_windows=vehicle.deliveries,           # Giữ nguyên ràng buộc giờ mở/đóng cửa
            start_time=vehicle.current_time            # Tính toán dựa trên giờ hiện tại của xe
        )
        
        if new_plan and new_plan.get("route"):
            print(f"✅ Re-planning hoàn tất! Thứ tự đi mới ưu việt nhất: {new_plan['visited_order']}")
        else:
            print("❌ Lỗi: Với tình trạng giao thông hiện tại, xe không thể giao kịp các điểm còn lại. Cần điều phối thủ công!")
            
        return new_plan