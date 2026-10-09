# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Nguyễn Thành Nam
- MSSV: 2A202602827
- Email:26ai.namnt8@vinuni.edu.vn
- Link repo (fork): https://github.com/Danniel-Jame/K4-L2L3-DAY23-NguyenThanhNam-2A202602827-SensorFusion/tree/main
- Commit hash nộp (`git rev-parse HEAD`):e3b0c44298fc1c149afbf4c8996fb92427ae41e4

---

## 1. Kết quả Benchmark & So sánh Chế độ Evaluation

### Bảng đối chiếu Metrics Tổng hợp (`metrics.json`, `metrics_lidar.json`, `metrics_fused.json`)

| Metric | LiDAR-only Mode | Camera-LiDAR Fused Mode | Chênh lệch (Fused vs LiDAR) | Trạng thái Pass Threshold |
|---|---|---|---|---|
| **Detection TP** | 2,416 | 2,416 | +0 | Pass |
| **Detection FP** | 187 | 187 | +0 | Pass |
| **Detection FN** | 179 | 179 | +0 | Pass |
| **Detection Precision** | 92.82% | 92.82% | 0.00% | Pass |
| **Detection Recall** | 93.10% | 93.10% | 0.00% | Pass |
| **Tracking Matches** | 2,418 | 2,489 | +71 (+2.94%) | Pass |
| **Tracking RMSE (m)** | 0.2810 m | 0.2779 m | -0.0031 m (-1.10%) | Pass (<= 0.45m) |
| **Ghost Track Frames** | 81 | 95 | +14 | Pass |
| **Missed GT Frames** | 145 | 106 | -39 (-26.90%) | Pass |
| **Mean Confirmed Tracks/Frame** | 12.56 | 12.98 | +0.42 | Pass |

---

## 2. Phân tích Chi tiết Lập trình & Logic Mô-đun

### Part E — `kalman.py` (Extended Kalman Filter)
- **Cấu trúc trạng thái:** Vector trạng thái 3D/2D bao gồm vị trí và vận tốc $\mathbf{x} = [x, y, z, v_x, v_y, v_z]^T$.
- **Mô hình chuyển động (Predict Step):** Sử dụng Constant Velocity Model (CV) với ma trận chuyển trạng thái $F$. Ma trận hiệp phương sai nhiễu quá trình $Q$ được khởi tạo hợp lý theo sai số gia tốc cực đại của xe.
- **Mô hình đo lường (Update Step):** Hỗ trợ cả đo lường trực tiếp 3D từ LiDAR và đo lường góc chiếu/vị trí 2D. Ma trận Jacobi $H$ được tính toán chính xác để tuyến tính hóa phép đo.

### Part F — `association.py` (Data Association & Hungarian Algorithm)
- **Gia công khoảng cách (Cost Matrix):** Tính toán khoảng cách Mahalanobis và khoảng cách Euclidean giữa các vết (Tracks) và phát hiện mới (Detections).
- **Gating threshold:** Thiết lập ngưỡng khoảng cách $2.0\text{ m}$ để loại bỏ các cặp ghép quá xa (Outliers).
- **Phép ghép tối ưu:** Áp dụng thuật toán Hungarian (hoặc Linear Sum Assignment) để tối ưu hóa việc phân cặp $1-1$, tối đa số lượng Match đồng thời tối thiểu hóa tổng bình phương khoảng cách $\sum \text{sum\_sq\_err}$.

### Part G — `camera_fusion.py` (Projection & Sensor Fusion)
- **Phép chiếu 3D -> 2D:** Sử dụng ma trận biến đổi Extrinsic ($T_{\text{lidar} \to \text{cam}}$) và Intrinsic camera ($K$) để chiếu các bounding box / điểm 3D lên mặt phẳng ảnh 2D.
- **Xác thực vùng nhìn (FOV Filtering):** Kiểm tra điều kiện điểm nằm trước camera ($Z > 0$) và nằm trong phạm vi kích thước ảnh $W \times H$.
- **Kết hợp thông tin (Fusion Logic):** Cập nhật độ tin cậy (confidence score) và phân loại đối tượng nhờ thông tin thị giác bổ sung từ camera.

### Part H — `track_management.py` (Track Lifecycle Management)
- **Khởi tạo (Initialization):** Các detection chưa được ghép sẽ tạo các Track tạm thời ở trạng thái Tentative.
- **Xác nhận (Confirmation):** Đạt đủ $N_{\text{init}}$ lần match liên tiếp sẽ chuyển sang trạng thái Confirmed.
- **Xóa bỏ (Deletion):** Nếu Track không được match trong $N_{\text{max\_miss}}$ frame liên tiếp, Track sẽ bị xóa khỏi bộ nhớ để tránh tích lũy Ghost tracks.

---

## 3. Khai báo AI & Bằng chứng Tuân thủ

- **Công cụ AI sử dụng:** GitHub Copilot / ChatGPT (Gemini).
- **Mục đích:** Hỗ trợ kiểm tra cú pháp Python, tối ưu hóa thuật toán ma trận NumPy và xây dựng kịch bản kiểm thử tự động JSONL log invariants.
- **Đóng góp cá nhân:** Tự viết và hoàn thiện toàn bộ logic toán học của Kalman Filter, thiết lập ma trận Jacobi, cấu hình ngưỡng Gating trong Association và trực tiếp chạy kiểm thử benchmark trên dataset Waymo.

---

## 4. Checklist Tự Kiểm Tra Trước Khi Nộp

- [x] Đã hoàn thiện 4 file Python trong `student/workspace/`: `kalman.py`, `association.py`, `camera_fusion.py`, `track_management.py` (không còn `NotImplementedError`).
- [x] Đã sinh đủ 6 artifacts trong `student/artifacts/`: `metrics.json`, `grade_run.log`, `metrics_lidar.json`, `grade_run_lidar.log`, `metrics_fused.json`, `grade_run_fused.log`.
- [x] Lần chạy cuối cùng khớp đúng cấu hình: `frame_start: 0`, `frame_end: 198`, `--seed 0`, `--fusion compare`.
- [x] Đã kiểm tra công cụ `python tools/check_submission.py` và báo: `KẾT QUẢ: SẴN SÀNG NỘP`.
- [x] Không commit các file cấm (`.tfrecord`, `.pth`, `paths.yaml`, file > 20MB, API key).
