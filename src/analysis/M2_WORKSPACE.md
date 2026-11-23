
# Không gian TV2

Làm trên nhánh `feature/m2-eda-dashboard`, thêm mã trong `src/analysis/` và xây báo cáo Power BI trong `dashboard/`. Tạo Parquet mồi ngay từ gốc repository.

```powershell
python -m src.core.bootstrap_and_validate bootstrap --samples 5000
```

Dùng `data/mock/mock_clean_sample.parquet` để thử Pearson/Spearman trên từng cặp Series theo hãng và `year_era` và dựng Star Schema. Giữ `listing_id` làm khóa liên kết khi chuyển sang dữ liệu TV1.

Nạp `outputs/predictions/predictions_test.parquet` cho Trang 3. Tệp có 9 cột trong [Data Contract](../../docs/DATA_CONTRACT.md) và thêm `manufacturer_clean`, `car_age`, `year_era`, `abs_pct_error`. Dùng `actual_price_usd` và `predicted_price_usd` cho scatter, `residual_usd` cho phân tích phần dư. Lọc theo `dataset_version` và đối chiếu `listing_id` với `outputs/models/split_assignments.parquet` của cùng lần chạy.
