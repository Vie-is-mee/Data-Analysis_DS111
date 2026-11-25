# Kế hoạch phân tích TV2 (M2-01)

> **Trạng thái:** bản nháp 25/11/2026, chờ nhóm duyệt ở Sync #1. Mọi định nghĩa dưới đây được chốt
> **trước khi xem kết quả chính thức** trên ABT V1.0, để tránh chọn ngưỡng theo kết quả đẹp.
> Căn cứ: `docs/DATA_CONTRACT.md` 1.1 trên `main` (năm đến 2027, gộp tin đăng lại theo VIN,
> cờ `condition_missing`, `cylinders_was_missing`, cột `seller_type`).

## 1. Câu hỏi nghiên cứu

| Mã | Câu hỏi | Người | Bằng chứng chính |
|---|---|---|---|
| Q1 | Đặc trưng nào liên hệ mạnh nhất với `log_price` khi đo trên cùng thang (r² cho biến số, η² cho biến phân loại)? | TV2 | `eda_*`, `correlation_results`, `categorical_association`, `common_scale_ranking`, `feature_ranking_m2` |
| Q1b | Các liên hệ đó có bị nhiễu bởi loại người bán, loại xe hoặc số dặm không? | TV2 | `confounding_*`, `partial_correlations` |
| Q2 | Tốc độ mất giá theo tuổi và theo số dặm khác nhau thế nào giữa ba cụm hãng Nhật, Mỹ, Đức khi đã kiểm soát số dặm và loại xe? | TV2 | `group_depreciation`, `median_curves` |
| Q3 | Quan hệ tuổi–giá có đổi chiều ở xe cổ không, từ khoảng tuổi nào? | TV2 | `median_curves`, `correlation_results` (scope `year_era`), `era_ranges` |
| Q4 | Mỗi bước tiền xử lý có cải thiện mô hình không? | TV1, TV3 | Exp0–Exp3 (TV3) |
| Q5 | Mô hình dự báo tốt đến đâu, sai số tập trung ở phân khúc nào? | TV3 (TV2 hỗ trợ trang 3) | Exp4, `predictions_test.parquet` |

## 2. Định nghĩa đã chốt

| Mục | Quy ước | Lý do |
|---|---|---|
| Đơn vị phân tích | Một dòng ABT = một tin sau khi gộp tin đăng lại theo VIN (hợp đồng 1.1) | Đại lý đăng lặp cùng một xe; giữ cả thì thống kê lệch về xe đại lý và khoảng tin cậy hẹp giả |
| Cụm hãng | JP = toyota, honda; US = ford, chevrolet; DE = bmw, mercedes-benz, audi (`brand_lookup.csv`) | Mỗi hãng có hàng nghìn tin; đại diện ba thị trường mà đề tài so sánh |
| Cửa sổ tuổi cho độ dốc | 1–20 năm | Đường trung vị giá theo tuổi có dạng chữ U; xe 0 tuổi có giá đặc thù |
| Đời xe | Classic < 2000, Old 2000–2010, Mid 2011–2016, Modern ≥ 2017 (hợp đồng) | Dùng chung với TV1, TV3 |
| Biến kiểm soát | `odometer` (mỗi 10.000 dặm), `type`; phân tầng theo `seller_type` | Ba nguồn nhiễu đã thấy trong audit |
| Ngưỡng n | ≥ 30 cặp cho một hệ số; ≥ 100 tin cho một nhóm; ≥ 30 tin cho một loại xe khi kiểm soát `type` | Nhóm nhỏ cho hệ số và độ dốc không ổn định |
| Khoảng tin cậy 95% | Pearson: Fisher z; Spearman: Bonett–Wright; riêng phần: Fisher z với n − 3 − k; độ dốc: OLS sai số HC3 | p-value ≈ 0 với n vài trăm nghìn nên không cho biết độ mạnh |
| Hệ số chính | Spearman nếu có biến thứ bậc, hoặc \|ρ\| − \|r\| > 0,10, hoặc \|skew\| > 1; ngược lại Pearson | Quy tắc kinh nghiệm, ghi trong `correlation_engine.recommend_method` |
| Kendall τ_b | Chỉ báo để kiểm tra độ bền | SciPy tính O(n log n), không chậm; Spearman so trực tiếp được với Pearson |
| Xi-lanh | Chỉ dùng giá trị quan sát (`cylinders_was_missing = False`) | Khoảng 27% giá trị đã được điền theo nhóm |
| Phạm vi dữ liệu | Toàn ABT cho mô tả và dashboard; tập `train` cho ranking làm căn cứ mô hình | Khớp TV3, không dùng thông tin tập test |
| Độ nhạy | (a) chỉ số dặm ≥ 1.000; (b) ABT chưa gộp tin đăng lại | Số dặm rất nhỏ là điểm đòn bẩy của `log_odometer`; đơn vị phân tích thay đổi trung vị giá |

## 3. Câu hỏi → bảng, hình, trang dashboard

| Câu hỏi | Bảng (`outputs/reports/m2/`) | Hình (`figures/`) | Dashboard |
|---|---|---|---|
| Q1 | `eda_numeric`, `eda_categorical`, `correlation_results` (scope `all`, `odometer>=1000`), `categorical_association`, `common_scale_ranking`, `feature_ranking_m2` | `eda_Q1_price_distribution`, `eda_Q1_pearson_vs_spearman`, `eda_Q1_common_scale_ranking`, `eda_Q1_price_by_*` | Trang 1: KPI, phân phối giá, giá theo loại xe/tình trạng |
| Q1b | `confounding_condition_missing`, `confounding_condition_by_seller`, `confounding_cramers_v`, `confounding_type_mix`, `confounding_cylinders_price`, `partial_correlations` | `eda_Q1b_condition_by_seller` | Trang 1: slicer loại người bán |
| Q2 | `group_depreciation` (brand, cluster, cluster theo loại xe), `median_curves` | `eda_Q2_cluster_age_curves`, `eda_Q2_depreciation_by_brand`, `eda_Q2_depreciation_by_cluster` | Trang 2: đường giá theo tuổi theo cụm; bảng %/năm |
| Q3 | `median_curves` (all), `correlation_results` (scope `year_era`), `era_ranges` | `eda_Q3_median_price_by_age`, `eda_Q3_rho_by_era` | Trang 2: Pearson động theo đời xe |
| Số chính | `m2_key_numbers` | — | Mọi thẻ KPI lấy số từ đây (Gate 3) |

## 4. Quyết định còn chờ nhóm

1. Thư mục hình cuối cho slide: `reports/figures/` (lên Git) hay giữ `outputs/reports/m2/figures/` và chia sẻ qua Drive.
2. Trang 3 (dự báo): cần TV3 xuất thêm một mô hình tuyến tính nếu muốn so Linear với mô hình tốt nhất.
3. Sheet 06 của Workbook dùng bảng `correlation_results.csv` của TV2 thay cho ma trận tính trong exporter của TV1.
4. Rubric chính thức của giảng viên (dashboard, Power BI, quy tắc tương quan, báo cáo viết).

## 5. Cách chạy

```powershell
# Từ gốc repo, sau khi có ABT của TV1
python -X utf8 -m src.analysis.run_m2 --abt data/processed/used_cars_clean_v1.parquet `
    --dataset-version v1.0 --transform-log data/processed/m1_transform_log.csv --bi
python -X utf8 -m pytest -q tests/test_m2_correlation.py tests/test_m2_association.py `
    tests/test_m2_groups.py tests/test_m2_pipeline_smoke.py
# So số chính giữa hai phiên bản (ví dụ v0.9 với V1.0)
python -X utf8 -m src.analysis.key_numbers compare --old old/m2_key_numbers.csv --new new/m2_key_numbers.csv
```

Máy cá nhân của TV2 không đủ tài nguyên nên các lần chạy trên dữ liệu thật dùng Kaggle
(kernel riêng của TV2: `00_DS111` chạy pytest, `01_DS111` dựng ABT phát triển bằng pipeline TV1 rồi chạy `run_m2`).
Số liệu từ ABT phát triển mang `dataset_version` bắt đầu bằng `dev-` và **không dùng làm kết quả chính thức**.

## 6. Giới hạn cần ghi trong báo cáo

- Giá rao bán, không phải giá giao dịch; tình trạng và tình trạng pháp lý do người bán tự khai.
- Dữ liệu cắt ngang 5 tuần (30/09–03/11/2026): "mất giá" là chênh lệch giữa các đời xe tại cùng thời điểm.
- Không có phiên bản (trim), tùy chọn, lịch sử tai nạn: sai số định giá xe sang và xe cổ sẽ cao.
- Mọi kết luận là liên hệ, không phải nhân quả.
