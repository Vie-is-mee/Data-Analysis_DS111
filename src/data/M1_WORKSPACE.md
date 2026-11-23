
# Không gian TV1

Làm trên nhánh `feature/m1-data-workbook`. Đặt mã tiền xử lý trong `src/data/` và quản lý sheet 00–07 trong Workbook dự án.

- Dùng `from src.core.formula_store import FormulaMetadata, FormulaStore`. Đăng ký Series và hồ sơ qua `apply_and_register`, rồi dùng `to_dataframe_report()` để nối nhật ký biến đổi vào Workbook.
- Xuất `data/processed/used_cars_clean_v1.parquet` theo [Data Contract](../../docs/DATA_CONTRACT.md). Giữ khóa `listing_id` ổn định và nhãn phân loại dạng chuỗi. TV3 tạo bảng chia chính thức theo ID.
- Giữ dữ liệu thô, Parquet, Workbook cục bộ và tệp joblib ngoài Git. Chỉ nạp FormulaStore do nhóm tạo và tin cậy.

Chạy kiểm định từ gốc repository trước khi mở PR.

```powershell
python -m src.core.bootstrap_and_validate validate --path data/processed/used_cars_clean_v1.parquet --store data/processed/formula_store_v1.joblib
```
