# Dashboard Power BI · Mục tiêu – Triết lý – Kỹ thuật – Kết quả (bản nháp M2-12)

> Bản nháp 25/11/2026. Mục 6 điền sau khi có ABT V1.0 và `m2_key_numbers.csv` đã khóa ở Gate 3.

## 1. Mục tiêu và người dùng

- **Người dùng:** giảng viên khi chấm và vấn đáp; người mua/bán xe cũ muốn biết khoảng giá hợp lý theo phân khúc.
- **Mục tiêu:** trả lời Q1–Q3 trong [`src/analysis/M2_ANALYSIS_PLAN.md`](../src/analysis/M2_ANALYSIS_PLAN.md)
  bằng tương tác; mọi con số khớp với slide và báo cáo. Q4–Q5 thuộc TV3 (trang 3 tùy chọn).

## 2. Bố cục

| Trang | Câu hỏi | Visual | Bộ lọc |
|---|---|---|---|
| 1 · Thị trường và chất lượng dữ liệu | Q1, Q1b | Thẻ KPI (số tin, giá trung vị, tuổi trung vị, % đại lý); phễu làm sạch từ nhật ký TV1; phân phối giá; giá trung vị theo loại xe và tình trạng | Cụm hãng, loại người bán, bang |
| 2 · Khấu hao và tương quan | Q2, Q3 | Đường giá trung vị theo tuổi theo cụm hãng; bảng hãng (n, Pearson động, %/năm); Pearson theo đời xe; bảng tĩnh độ dốc có kiểm soát | Cụm hãng, loại xe, tuổi xe (mặc định 1–20) |
| 3 · Chẩn đoán dự báo (tùy chọn) | Q5 | Giá thật so với dự báo; % sai số tuyệt đối theo cụm hãng, đời xe | Mô hình, cụm hãng |

## 3. Triết lý thiết kế

- Mỗi visual trả lời một câu hỏi; tiêu đề là câu hỏi hoặc kết luận; luôn hiển thị n.
- Đi từ tổng quan đến chi tiết: KPI → phân phối → so sánh nhóm.
- Màu cố định theo cụm hãng (JP, US, DE), trùng với hình Python; tối đa 5 màu mỗi trang; không dùng 3D.
- Hệ số bị ẩn khi n < 100 để người xem không đọc hệ số của nhóm quá nhỏ.

## 4. Kỹ thuật

- Star Schema gọn: 1 Fact và 4 Dim với khóa tự nhiên, lọc một chiều. Với khoảng 200.000 dòng, lý do chính
  là lọc rõ ràng và thuộc tính mô tả nằm ở Dim, không phải hiệu năng.
- DAX hai lượt cho Pearson và độ dốc; bảng đối chiếu với Python (xem [`dax_measures.md`](dax_measures.md)).
- Tham số `BiFolder` để đổi nguồn dữ liệu từ bản dev sang V1.0 bằng một lần sửa.

## 5. Giới hạn

- Measure động là liên hệ biên trong bộ lọc, chưa kiểm soát số dặm hay loại xe; bảng độ dốc có kiểm soát lấy từ Python.
- Giá rao bán, dữ liệu cắt ngang 5 tuần, thông tin tự khai; mọi kết luận là liên hệ, không phải nhân quả.

## 6. Con số đại diện (điền từ `m2_key_numbers.csv` sau Gate 3)

| Khóa | Giá trị | Ý nghĩa |
|---|---|---|
| `price_median` | — | Giá trung vị |
| `rho_car_age_log_price` | — | Độ chặt quy luật tuổi–giá |
| `pct_per_10k_miles_cluster_de_odometer` / `_jp_odometer` | — | Mất giá theo số dặm của xe Đức và xe Nhật |
