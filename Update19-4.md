Khoi làm xong module ML dự đoán thời gian di chuyển.

Phần hoàn thành gồm: 
- Hoàn thiện pipeline tiền xử lý feature cho edge: length, maxspeed, highway, weather, is_peak_hour, congestion_prob
- Viết script train mô hình dự đoán travel_time_min
- Thử 2 mô hình cơ bản là Linear Regression và Decision Tree, hiện đang chọn model tốt hơn tự động
- Đóng gói hàm predict cho từng edge và hàm batch predict cho toàn bộ edges
- Tích hợp được với output Bayes của TV4 qua p_congestion
- Có script test integration và script EDA để phục vụ báo cáo

Đầu vào/đầu ra hiện tại:
- Input: edges + congestion_df
- Output: weighted_edges gồm u, v, travel_time_min
- TV2 có thể dùng output này làm trọng số động cho A*

Cách test:
- python -m modules.ml.train --synthetic-only
- python -m modules.ml.test_integration
- python -m modules.ml.eda

Những gì cần làm tiếp:
- TV2 ghép weighted_edges từ ML vào A* để thay cost khoảng cách bằng travel_time_min
- TV1 chạy thử full pipeline trong notebook/hệ thống chính
- TV4 giữ ổn định format congestion_df gồm u, v, p_congestion để nối với ML
- Cả nhóm test end-to-end một lần trước khi chốt
