
# Không gian TV2

Làm trên nhánh `feature/m2-eda-dashboard`, thêm mã trong `src/analysis/` và xây báo cáo Power BI trong `dashboard/`. Tạo Parquet mồi ngay từ gốc repository.

```powershell
python -m src.core.bootstrap_and_validate bootstrap --samples 5000
```

Dùng `data/mock/mock_clean_sample.parquet` để thử Pearson/Spearman trên từng cặp Series theo hãng và `year_era` và dựng Star Schema. Giữ `listing_id` làm khóa liên kết khi chuyển sang dữ liệu TV1.

Nạp `outputs/predictions/predictions_test.parquet` cho Trang 3. Tệp có 9 cột trong [Data Contract](../../docs/DATA_CONTRACT.md) và thêm `manufacturer_clean`, `car_age`, `year_era`, `abs_pct_error`. Dùng `actual_price_usd` và `predicted_price_usd` cho scatter, `residual_usd` cho phân tích phần dư. Lọc theo `dataset_version` và đối chiếu `listing_id` với `outputs/models/split_assignments.parquet` của cùng lần chạy.

## Mã nguồn TV2 (cập nhật 25/11/2026)

Kế hoạch phân tích, định nghĩa đã chốt và ma trận câu hỏi → bảng/hình: [M2_ANALYSIS_PLAN.md](M2_ANALYSIS_PLAN.md).

| File | Mã việc | Nội dung |
|---|---|---|
| `correlation_engine.py` | M2-03 | Pearson/Spearman/Kendall cho từng cặp `Series`, n, khoảng tin cậy, Δ, tương quan riêng phần |
| `eda_summary.py` | M2-04 | Thống kê đơn biến, phễu làm sạch từ nhật ký TV1, câu nhận định nháp |
| `group_analysis.py` + `brand_lookup.csv` | M2-06, M2-20 | Cụm hãng, đường trung vị theo tuổi, độ dốc %/năm (OLS log, HC3) có kiểm soát |
| `association.py` | M2-18 | η², Kruskal–Wallis ε², Cramér's V, Cliff's δ, trung vị theo nhãn, xếp hạng chung r²/η² |
| `confounding.py` | M2-19 | Cơ chế thiếu dữ liệu, phân tầng theo người bán, cơ cấu loại xe theo cụm |
| `ranking.py` | M2-07 | Dùng lại ranking TV3 trên train, thêm liên hệ biên |
| `figures.py` | M2-13 | Hình 300 DPI có mã câu hỏi, phương pháp, n, phiên bản dữ liệu |
| `key_numbers.py` | M2-21 | `m2_key_numbers.csv` và lệnh so hai phiên bản |
| `bi_export.py` | M2-02, M2-08 | Star Schema cho Power BI, bảng đối chiếu DAX |
| `run_m2.py` | M2-10 | Chạy toàn bộ bằng một lệnh, ghi `run_manifest.json` |

```powershell
python -X utf8 -m src.analysis.run_m2 --abt data/processed/used_cars_clean_v1.parquet --dataset-version v1.0 --bi
python -X utf8 -m pytest -q tests/test_m2_correlation.py tests/test_m2_association.py tests/test_m2_groups.py tests/test_m2_pipeline_smoke.py
```

Đầu ra nằm ở `outputs/reports/m2/` (Git bỏ qua). Mô hình dữ liệu và DAX ở [dashboard/dax_measures.md](../../dashboard/dax_measures.md).
