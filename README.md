# 🚚 Hệ thống AI Tối ưu Lộ trình Xe giao hàng

## 📌 Thông tin môn học

| | |
|:---|:---|
| **Môn học** | Nhập môn Trí tuệ nhân tạo (CO3061) |
| **Học kỳ** | II - Năm học 2025 – 2026 |
| **Giảng viên hướng dẫn** | TS. Trương Vĩnh Lân |

## 👥 Danh sách thành viên nhóm

| STT | Họ và tên | MSSV | Vai trò |
|:---:|:---|:---|:---|
| 1 | Vạn Trường Thành | 2353107 | Module Integration, Notebook, Tài liệu kỹ thuật |
| 2 | Lê Minh Trí | 2353229 | Module Search (A*), CSP Time Windows |
| 3 | Nguyễn Minh Khôi | 2352618 | Module ML, EDA, Phân tích mô hình |
| 4 | Trần Nguyễn Nhất Thông | 2353149 | Module Bayes Logic, Biểu diễn tri thức |

## 🎯 Mục tiêu bài tập lớn

Dự án xây dựng một hệ thống Trí tuệ Nhân tạo tích hợp giải quyết bài toán **Tối ưu lộ trình xe giao hàng trong môi trường giao thông đô thị có yếu tố không chắc chắn (VRPTW)**. Hệ thống áp dụng đầy đủ 5 thành phần cốt lõi của AI:

| Thành phần | Triển khai |
|:---|:---|
| **(A) Tìm kiếm** | Thuật toán A* với heuristic Haversine trên đồ thị OSMnx thực tế |
| **(B) Heuristic/CSP** | Hàm heuristic admissible + ràng buộc Time Windows |
| **(C) Biểu diễn tri thức** | Hệ luật IF-THEN trong `logic_filter.py` |
| **(D) Mạng Bayes** | `BayesCongestionModel` với CPT theo loại đường và điều kiện môi trường |
| **(E) Học máy** | Decision Tree Regressor dự đoán `travel_time_min` |

## 📂 Cấu trúc thư mục

```
Delivery-Routing-Optimization/
├── notebooks/
│   └── main_pipeline.ipynb       # Notebook chính — demo toàn bộ pipeline
├── modules/
│   ├── graph/
│   │   ├── map_loader.py          # Tải bản đồ từ OpenStreetMap (OSMnx)
│   │   └── core_system.py         # MapGraph, DeliveryVehicle
│   ├── search/
│   │   └── astar.py               # Thuật toán A* + TSP + CSP Time Windows
│   ├── bayes_logic/
│   │   ├── bayes_model.py         # Mạng Bayes tính p_congestion
│   │   └── logic_filter.py        # Hệ luật IF-THEN lọc đồ thị
│   ├── ml/
│   │   ├── train.py               # Huấn luyện và so sánh mô hình ML
│   │   ├── predictor.py           # Dự đoán travel_time_min từng cạnh
│   │   ├── preprocess.py          # Tiền xử lý đặc trưng
│   │   └── travel_time_predictor.py  # Batch prediction cho toàn bộ bản đồ
│   └── integration/
│       └── coordinator.py         # DeliveryCoordinator — điều phối pipeline
├── features/
│   └── travel_time_model.npy      # Artifact mô hình ML đã huấn luyện
├── reports/
│   └── final_report.pdf           # Báo cáo PDF chi tiết
└── data/
    └── mock_edge_travel_time.csv  # Dataset huấn luyện ML
```

## 🚀 Hướng dẫn chạy notebook

### Yêu cầu thư viện
Tất cả thư viện được cài đặt tự động trong notebook:
```
osmnx, networkx, folium, scikit-learn, pandas, numpy, matplotlib
```

### Các bước chạy
1. Mở notebook trên Google Colab theo link bên dưới
2. Chọn **Runtime → Run all**
3. Toàn bộ pipeline chạy tự động — không cần mount Google Drive hay cài đặt thủ công

> **Lưu ý:** Dữ liệu bản đồ được tải tự động từ OpenStreetMap qua public API. Mã nguồn được clone trực tiếp từ repository này.

## 🔗 Liên kết quan trọng

| | |
|:---|:---|
| 📓 **Google Colab Notebook** | [Mở trên Colab](https://colab.research.google.com/github/thanh081205/Delivery-Routing-Optimization/blob/main/notebooks/main_pipeline.ipynb) |
| 📄 **Báo cáo PDF** | Xem trong thư mục `reports/` |
| 💻 **GitHub Repository** | [thanh081205/Delivery-Routing-Optimization](https://github.com/thanh081205/Delivery-Routing-Optimization) |