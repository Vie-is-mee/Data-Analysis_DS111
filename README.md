# DS111 – Phân tích dữ liệu và định giá xe cũ Craigslist

Đồ án nhóm ba người, dùng Python 3.11 trên CPU. Pipeline TV3 so sánh OLS, Ridge, Decision Tree, Random Forest và LightGBM để dự báo giá xe đã qua sử dụng.

## Vai trò và vùng sở hữu

| Thành viên | Nhánh | Vùng thực hiện |
|---|---|---|
| TV1 – dữ liệu và Workbook | `feature/m1-data-workbook` | [src/data](src/data/M1_WORKSPACE.md), `workbook/` (sheet 00–07) |
| TV2 – EDA và Power BI | `feature/m2-eda-dashboard` | [src/analysis](src/analysis/M2_WORKSPACE.md), `dashboard/` |
| TV3 – mô hình và điều phối | `feature/m3-models` | `src/models/`, `scripts/`, `outputs/` (tệp sinh cục bộ) |

Phần dùng chung gồm `src/core/`, `configs/`, `requirements.txt` và [Hợp đồng dữ liệu](docs/DATA_CONTRACT.md). Mỗi người phát triển trên nhánh riêng và mở PR vào `main`.

## Chạy thử trên Windows PowerShell

```powershell
git clone https://github.com/Vie-is-mee/Data-Analysis_DS111.git
cd Data-Analysis_DS111
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
```

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.core.bootstrap_and_validate bootstrap --samples 5000
.\.venv\Scripts\python.exe -X utf8 -m src.core.bootstrap_and_validate validate --parquet data/mock/mock_clean_sample.parquet --formula-store data/mock/mock_formula_store.joblib
.\.venv\Scripts\python.exe -X utf8 scripts/run_m3_baseline.py --mode mock --fast
.\.venv\Scripts\python.exe -X utf8 scripts/run_m3_baseline.py --mode mock
```

Mock lấy mẫu từ CSV nếu có hoặc tạo dữ liệu giả lập. Cả hai chỉ dùng thử kỹ thuật. Chỉ nạp FormulaStore do nhóm tự tạo và tin cậy. `--fast` giảm số cây của Random Forest và LightGBM, giữ nguyên số dòng và fold. Bỏ cờ này khi nghiệm thu cấu hình đầy đủ.

## Chạy trên dữ liệu TV1 bàn giao

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.core.bootstrap_and_validate validate --parquet data/processed/used_cars_clean_v1.parquet --formula-store data/processed/formula_store_v1.joblib
.\.venv\Scripts\python.exe -X utf8 scripts/run_m3_baseline.py --mode production
```

Production chia 70/15/15 trên danh sách `listing_id` đã sắp xếp với seed 42. Khi bộ ID đổi, tạo phiên bản bàn giao mới và giữ manifest của phiên bản trước. TV3 ghi manifest riêng, giữ file Parquet TV1 làm nguồn đối chiếu. Exp0 ghép `year` và `odometer` từ `data/raw/vehicles.csv` nếu đủ khóa duy nhất. Các thí nghiệm dùng cùng mẫu và giá tham chiếu để so sánh. Khi thiếu CSV hoặc khóa không ghép được, Exp0 ghi rõ dùng proxy từ mẫu đã lọc.

## Đọc kết quả

- `outputs/models/ablation_summary.csv` lưu đối chứng Exp0–Exp4, gồm tuổi xe bình phương và One-Hot so với Ordinal giả thuyết cho `title_status_clean`.
- `outputs/models/split_assignments.parquet` lưu `listing_id`, nhãn chia tập và phiên bản dữ liệu.
- `outputs/predictions/predictions_test.parquet` chứa dự báo test theo hợp đồng, kèm hãng, tuổi xe, đời xe và sai số phần trăm cho TV2.
- `workbook/Project_Workbook_M3_Tracking.xlsx` có ba sheet TV3. Workbook gốc của TV1 được quản lý riêng.

Mỗi lần chạy cập nhật các đường dẫn đầu ra trên. Giữ bản production riêng khi chạy lại mock. Git bỏ qua dữ liệu, Workbook và kết quả sinh tự động.

Imputer, scaler và encoder chỉ fit trên phần train của từng fold. Target Encoding dùng ánh xạ OOF cho hàng train và ánh xạ train đầy đủ cho validation/test. R² dùng trung bình của tập đang đánh giá và Adjusted R² để trống khi thiếu bậc tự do. Mô hình log nghịch đảo về USD với Duan Smearing. Trong CV hệ số học từ phần train của fold, còn hệ số cuối học từ dự báo OOF trên toàn train. Mô hình được chọn theo RMSE USD trung bình của CV, dùng validation nếu hòa. Chỉ ứng viên đã chốt được chấm test.
