# INTERFACES.md — Giao ước Dữ liệu Giữa Các Module

> **Mục đích:** File này định nghĩa chính xác định dạng input/output của từng module.  
> Mọi thành viên **phải tuân thủ** các interface này để các module có thể ghép nối với nhau mà không phát sinh lỗi.  
> Nếu cần thay đổi, hãy tạo Pull Request và thông báo cho TV1 trước.

---

## Tổng quan Pipeline

```
[Bản đồ OSMnx]
      ↓
[TV4 - Logic Filter]  →  Đồ thị đã lọc (cleaned_graph)
      ↓
[TV4 - Bayes Module]  →  Ma trận xác suất kẹt xe (congestion_matrix)
      ↓
[TV3 - ML Module]     →  Trọng số cạnh dự đoán (predicted_weights)
      ↓
[TV2 - A* + CSP]      →  Lộ trình tối ưu (optimal_route)
      ↓
[TV1 - Visualizer]    →  Bản đồ hiển thị + Output
```

---

## Module 1 — `modules/graph/map_loader.py` (TV1)

### Mô tả
Tải bản đồ đường bộ từ OSMnx và chuyển thành cấu trúc đồ thị dùng chung.

### Output
```python
# Hàm chính
def load_map(place_name: str) -> dict:
    """
    Tải bản đồ và trả về graph_data.

    Returns:
        graph_data (dict): {
            "G": nx.MultiDiGraph,        # Đồ thị NetworkX gốc từ osmnx
            "nodes": pd.DataFrame,       # Thông tin các nút (node_id, x, y)
            "edges": pd.DataFrame        # Thông tin các cạnh (u, v, key, length, maxspeed, highway)
        }
    """
```

### Ví dụ sử dụng
```python
from modules.graph.map_loader import load_map

graph_data = load_map("Quan 1, Ho Chi Minh City, Vietnam")
G         = graph_data["G"]
nodes     = graph_data["nodes"]   # DataFrame: [node_id, x, y]
edges     = graph_data["edges"]   # DataFrame: [u, v, key, length, maxspeed, highway]
```

---

## Module 2 — `modules/bayes_logic/logic_filter.py` (TV4)

### Mô tả
Nhận đồ thị gốc, áp dụng các luật IF-THEN để lọc bỏ các cạnh không hợp lệ.

### Input
```python
graph_data: dict   # Output từ Module 1 (load_map)
vehicle_weight: float  # Tải trọng xe (tấn), ví dụ: 2.5
```

### Output
```python
def filter_graph(graph_data: dict, vehicle_weight: float) -> nx.MultiDiGraph:
    """
    Lọc đồ thị theo luật logic.

    Returns:
        cleaned_graph (nx.MultiDiGraph): Đồ thị đã xóa các cạnh vi phạm luật.
    """
```

### Ví dụ sử dụng
```python
from modules.bayes_logic.logic_filter import filter_graph

cleaned_graph = filter_graph(graph_data, vehicle_weight=2.5)
```

---

## Module 3 — `modules/bayes_logic/bayes_model.py` (TV4)

### Mô tả
Tính xác suất kẹt xe cho từng cạnh dựa trên điều kiện thời tiết và khung giờ.

### Input
```python
edges: pd.DataFrame    # Danh sách cạnh từ graph_data["edges"]
weather: str           # "rain" hoặc "clear"
time_of_day: str       # "peak" (giờ cao điểm) hoặc "normal"
```

### Output
```python
def compute_congestion(edges: pd.DataFrame, weather: str, time_of_day: str) -> pd.DataFrame:
    """
    Tính xác suất kẹt xe cho từng cạnh.

    Returns:
        congestion_df (pd.DataFrame): {
            "u": int,              # Node đầu cạnh
            "v": int,              # Node cuối cạnh
            "p_congestion": float  # Xác suất kẹt xe [0.0 - 1.0]
        }
    """
```

### Ví dụ sử dụng
```python
from modules.bayes_logic.bayes_model import compute_congestion

congestion_df = compute_congestion(
    edges=graph_data["edges"],
    weather="rain",
    time_of_day="peak"
)
# congestion_df có cột: u, v, p_congestion
```

---

## Module 4 — `modules/ml/travel_time_predictor.py` (TV3)

### Mô tả
Dự đoán thời gian di chuyển (phút) cho từng cạnh dựa trên đặc trưng đường và xác suất kẹt xe.

### Input
```python
edges: pd.DataFrame           # graph_data["edges"] — cần cột: u, v, key, length, maxspeed, highway
congestion_df: pd.DataFrame   # Output từ Module 3 — cần cột: u, v, p_congestion
```

### Output
```python
def predict_travel_time(edges: pd.DataFrame, congestion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dự đoán thời gian di chuyển cho từng cạnh.

    Returns:
        weighted_edges (pd.DataFrame): {
            "u": int,
            "v": int,
            "travel_time_min": float   # Thời gian di chuyển dự đoán (phút)
        }
    """
```

### Ví dụ sử dụng
```python
from modules.ml.travel_time_predictor import predict_travel_time

weighted_edges = predict_travel_time(edges=graph_data["edges"], congestion_df=congestion_df)
# weighted_edges có cột: u, v, travel_time_min
```

---

## Module 5 — `modules/search/astar.py` (TV2)

### Mô tả
Chạy thuật toán A* trên đồ thị đã có trọng số, có tích hợp kiểm tra ràng buộc CSP (Time Windows).

### Input
```python
cleaned_graph: nx.MultiDiGraph    # Output từ Module 2
weighted_edges: pd.DataFrame      # Output từ Module 4 — cột: u, v, key, travel_time_min
origin: int                       # node_id điểm xuất phát
destinations: list[int]           # Danh sách node_id các điểm giao hàng
time_windows: dict                # { node_id: (earliest_min, latest_min) }
                                  # Ví dụ: { 123456: (0, 30), 789012: (20, 60) }
start_time: float                 # Thời điểm bắt đầu (phút từ 0h), ví dụ: 480.0 (= 8:00 sáng)
```

### Output
```python
def run_astar(cleaned_graph, weighted_edges, origin, destinations, time_windows, start_time) -> dict:
    """
    Tìm lộ trình tối ưu bằng A*.

    Returns:
        result (dict): {
            "route": list[int],          # Danh sách node_id theo thứ tự di chuyển
            "total_time_min": float,     # Tổng thời gian di chuyển (phút)
            "visited_order": list[int]   # Thứ tự các điểm giao hàng đã ghé
        }
    """
```

### Ví dụ sử dụng
```python
from modules.search.astar import run_astar

result = run_astar(
    cleaned_graph=cleaned_graph,
    weighted_edges=weighted_edges,
    origin=origin_node,
    destinations=[node_A, node_B, node_C],
    time_windows={ node_A: (0, 30), node_B: (10, 50) },
    start_time=480.0
)

print(result["route"])           # [origin, ..., node_A, ..., node_B, ...]
print(result["total_time_min"])  # 47.3
```

---

## Ghi chú chung

- Tất cả `node_id` đều là kiểu `int` lấy từ OSMnx (osmid).
- Cột `key` trong `edges` là khóa cạnh của `MultiDiGraph`, dùng để phân biệt nhiều cạnh giữa cùng 2 node.
- Thời gian tính theo **phút** (float), không dùng giây hay giờ.
- Nếu một module chưa hoàn thiện, dùng **mock function** trả về dữ liệu giả đúng định dạng để không block các module khác.
- Mọi thay đổi interface phải được **thông báo trên nhóm chat** và cập nhật file này.