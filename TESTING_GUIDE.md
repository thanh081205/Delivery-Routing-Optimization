# Testing Guide

Tài liệu này tổng hợp cách test đầy đủ cho toàn bộ hệ thống `Delivery-Routing-Optimization`, bao gồm:

- test môi trường
- test riêng TV1, TV2, TV3, TV4
- test end-to-end toàn pipeline

## 1. Test Môi Trường

Kiểm tra các thư viện chính đã được cài:

```powershell
python -c "import osmnx, networkx, pandas, sklearn, joblib, matplotlib, folium; print('deps ok')"
```

Kỳ vọng:

```text
deps ok
```

## 2. Test Riêng Module ML Của TV3

### 2.1 Train model

```powershell
python -m modules.ml.train --synthetic-only
```

Mục đích:

- train lại model ML bằng dữ liệu synthetic
- sinh lại `features/travel_time_model.pkl`
- sinh lại artifact trong `modules/ml/artifacts`

### 2.2 Test integration demo của ML

```powershell
python -m modules.ml.test_integration
```

Mục đích:

- test nhanh luồng `Bayes -> ML`
- kiểm tra output `weighted_edges`

### 2.3 Test EDA

```powershell
python -m modules.ml.eda
```

Mục đích:

- sinh file thống kê EDA
- sinh hình cho báo cáo

## 3. Test Contract TV3 Theo Interface Mới Có `key`

```powershell
@'
from modules.ml.sample_data import demo_edges
from modules.bayes_logic.bayes_model import compute_congestion
from modules.ml.travel_time_predictor import predict_travel_time

edges = demo_edges()
if "key" not in edges.columns:
    edges["key"] = 0

congestion_df = compute_congestion(edges, weather="rain", time_of_day="peak")
if "key" not in congestion_df.columns:
    congestion_df["key"] = edges["key"].values

congestion_df["weather"] = "rain"
congestion_df["time_of_day"] = "peak"

weighted_edges = predict_travel_time(edges, congestion_df)

print(weighted_edges.to_string(index=False))

assert {"u", "v", "key", "travel_time_min"}.issubset(weighted_edges.columns)
assert len(weighted_edges) == len(edges)
assert weighted_edges["travel_time_min"].notna().all()
assert (weighted_edges["travel_time_min"] > 0).all()

print("\nTV3 OK")
'@ | python -
```

Kỳ vọng:

- output có cột `u`, `v`, `key`, `travel_time_min`
- in ra `TV3 OK`

## 4. Test Contract TV4

```powershell
@'
from modules.ml.sample_data import demo_edges
from modules.bayes_logic.bayes_model import compute_congestion

edges = demo_edges()
if "key" not in edges.columns:
    edges["key"] = 0

congestion_df = compute_congestion(edges, weather="rain", time_of_day="peak")

print(congestion_df.to_string(index=False))

assert {"u", "v", "key", "p_congestion"}.issubset(congestion_df.columns)
assert len(congestion_df) == len(edges)
assert congestion_df["p_congestion"].between(0.0, 1.0).all()

print("\nTV4 OK")
'@ | python -
```

Kỳ vọng:

- output có cột `u`, `v`, `key`, `p_congestion`
- in ra `TV4 OK`

## 5. Test Correctness Của TV2

Bài test này chứng minh A* thực sự dùng `travel_time_min` làm cost.

```powershell
@'
import pandas as pd
import networkx as nx
from modules.search.astar import run_astar

G = nx.MultiDiGraph()
G.add_node(1, x=0, y=0)
G.add_node(2, x=0, y=0)
G.add_node(3, x=0, y=0)
G.add_edge(1, 2, key=0, length=100)
G.add_edge(2, 3, key=0, length=100)
G.add_edge(1, 3, key=0, length=100)

weighted_edges = pd.DataFrame([
    {"u": 1, "v": 2, "key": 0, "travel_time_min": 1.0},
    {"u": 2, "v": 3, "key": 0, "travel_time_min": 1.0},
    {"u": 1, "v": 3, "key": 0, "travel_time_min": 50.0},
])

result = run_astar(
    cleaned_graph=G,
    weighted_edges=weighted_edges,
    origin=1,
    destinations=[3],
    time_windows={},
    start_time=480.0,
)

print(result)

assert result["route"] == [1, 2, 3]
assert abs(result["total_time_min"] - 2.0) < 1e-9

print("\nTV2 OK")
'@ | python -
```

Kỳ vọng:

- route phải là `[1, 2, 3]`
- in ra `TV2 OK`

## 6. Test TV1 Map Loader

```powershell
@'
from modules.graph.map_loader import load_map

graph_data = load_map(10.7729, 106.6578, 700)
edges = graph_data["edges"]
nodes = graph_data["nodes"]

print("nodes cols:", list(nodes.columns))
print("edges cols:", list(edges.columns))
print("num nodes:", len(nodes))
print("num edges:", len(edges))

assert "osmid" in nodes.columns
assert {"u", "v", "key", "length", "maxspeed", "highway"}.issubset(edges.columns)

print("\nTV1 OK")
'@ | python -
```

Kỳ vọng:

- `edges` có cột `key`
- in ra `TV1 OK`

## 7. Test End-to-End Toàn Pipeline

Đây là bài test chốt cuối cùng cho toàn hệ thống.

```powershell
@'
from modules.graph.map_loader import load_map
from modules.bayes_logic.logic_filter import filter_graph
from modules.bayes_logic.bayes_model import compute_congestion
from modules.ml.travel_time_predictor import predict_travel_time
from modules.search.astar import run_astar

LAT = 10.7729
LON = 106.6578
DIST = 700

graph_data = load_map(LAT, LON, DIST)
cleaned_graph = filter_graph(graph_data, vehicle_weight=2.5)

congestion_df = compute_congestion(graph_data["edges"], weather="rain", time_of_day="peak")
congestion_df["weather"] = "rain"
congestion_df["time_of_day"] = "peak"

weighted_edges = predict_travel_time(graph_data["edges"], congestion_df)

nodes = graph_data["nodes"]["osmid"].tolist()
origin = int(nodes[0])
destinations = [int(nodes[10]), int(nodes[20]), int(nodes[30])]

result = run_astar(
    cleaned_graph=cleaned_graph,
    weighted_edges=weighted_edges,
    origin=origin,
    destinations=destinations,
    time_windows={},
    start_time=480.0,
)

print("edges columns:", list(graph_data["edges"].columns))
print("congestion columns:", list(congestion_df.columns))
print("weighted columns:", list(weighted_edges.columns))
print(result)

assert {"u", "v", "key", "length", "maxspeed", "highway"}.issubset(graph_data["edges"].columns)
assert {"u", "v", "key", "p_congestion"}.issubset(congestion_df.columns)
assert {"u", "v", "key", "travel_time_min"}.issubset(weighted_edges.columns)
assert len(weighted_edges) == len(graph_data["edges"])
assert len(result["route"]) > 0

print("\nEND-TO-END OK")
'@ | python -
```

Kỳ vọng:

- không có traceback
- in ra `END-TO-END OK`
- route không rỗng

## 8. Thứ Tự Chạy Khuyến Nghị

Để test đầy đủ nhất, nên chạy theo thứ tự:

1. test môi trường
2. `python -m modules.ml.train --synthetic-only`
3. `python -m modules.ml.test_integration`
4. `python -m modules.ml.eda`
5. test TV3 contract
6. test TV4 contract
7. test TV2 correctness
8. test TV1 map loader
9. test end-to-end

## 9. Ghi Chú

- Ba lệnh:
  - `python -m modules.ml.train --synthetic-only`
  - `python -m modules.ml.test_integration`
  - `python -m modules.ml.eda`
  
  vẫn cần giữ vì đây là bộ test riêng cho TV3.

- Tuy nhiên, ba lệnh trên không đủ để chứng minh toàn pipeline đúng.

- Để chốt kỹ thuật cho cả nhóm, bắt buộc phải có thêm:
  - test contract có `key`
  - test correctness của A*
  - test end-to-end trên map thật
