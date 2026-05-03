# Update 03-05

Cả pipeline đã pass end-to-end.

## Kết quả hiện tại

- TV1: `load_map` + `core_system` OK, `edges` có `key`
- TV2: A* đọc đúng `travel_time_min` trên `MultiDiGraph`
- TV3: `weighted_edges` output đúng format `u, v, key, travel_time_min`
- TV4: `congestion_df` output đúng format `u, v, key, p_congestion`

## Test Full Pipeline

Pipeline:

`load_map -> filter_graph -> compute_congestion -> predict_travel_time -> run_astar`

đã chạy thành công trên map thật và trả về route hợp lệ.

## Đoạn ngắn cho báo cáo TV3

Module Machine Learning đã được tích hợp thành công vào pipeline tối ưu lộ trình. Mô hình dự đoán thời gian di chuyển nhận đầu vào là các cạnh đường đi cùng xác suất ùn tắc từ module Bayes, sau đó sinh ra trọng số động `travel_time_min` cho từng cạnh theo định dạng `u, v, key, travel_time_min`. Kết quả kiểm thử cho thấy module hoạt động đúng cả ở mức đơn lẻ lẫn khi ghép end-to-end với các module graph, Bayes và A* trên dữ liệu bản đồ thực.
