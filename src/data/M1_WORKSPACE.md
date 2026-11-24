
# Không gian TV1

Làm trên nhánh `feature/m1-data-workbook`. Đặt mã tiền xử lý trong `src/data/` và quản lý sheet 00–07 trong Workbook dự án.

- Dùng `from src.core.formula_store import FormulaMetadata, FormulaStore`. Đăng ký Series và hồ sơ qua `apply_and_register`, rồi dùng `to_dataframe_report()` để nối nhật ký biến đổi vào Workbook.
- Xuất `data/processed/used_cars_clean_v1.parquet` theo [Data Contract](../../docs/DATA_CONTRACT.md). Giữ khóa `listing_id` ổn định và nhãn phân loại dạng chuỗi. TV3 tạo bảng chia chính thức theo ID.
- Giữ dữ liệu thô, Parquet, Workbook cục bộ và tệp joblib ngoài Git. Chỉ nạp FormulaStore do nhóm tạo và tin cậy.

Chạy kiểm định từ gốc repository trước khi mở PR.

```powershell
python -m src.core.bootstrap_and_validate validate --path data/processed/used_cars_clean_v1.parquet --store data/processed/formula_store_v1.joblib
```

## Các module TV1

| File | Mã việc | Sheet |
|---|---|---|
| `raw_profile.py` | M1-02 · kiểm kê 26 cột (`python -X utf8 -m src.data.raw_profile`) | 01 |
| `transform_log.py` | Nhật ký hành động / lý do / số dòng trước-sau | 00 |
| `missing_handler.py` | M1-04 · Bước 1 thiếu dữ liệu | 02 |
| `formatter.py` | M1-05 · Bước 2 chuẩn hóa, lọc giá 2 tầng, cylinders | 03 |
| `feature_engineer.py` | M1-07 · Bước 3 đặc trưng qua FormulaStore | 04 |
| `selector.py` | M1-10 · Bước 5 giữ 22 cột hợp đồng, VIF | 06 |
| `pipeline_m1.py` | Chạy toàn bộ: `python -X utf8 -m src.data.pipeline_m1` | — |

Các chỗ còn quyết định được đánh dấu `TODO(M1-xx)`. Kiểm thử: `python -X utf8 -m pytest -q tests/test_m1_pipeline_smoke.py`.
