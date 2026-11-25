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
```

Với nguồn từ 2026, TV1 chuẩn bị Parquet nội bộ có `listing_id` (int64, duy nhất), `VIN` và `posting_date` sau các bộ lọc nghiệp vụ và trước bước gộp tin đăng lại. Tạo manifest từ tệp này. Lệnh chỉ in tổng hợp chất lượng VIN và chỉ xuất ID, khóa nhóm và tập chia.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.models.split_manifest --vin-listings data/processed/vin_candidates_local.parquet --output data/processed/split_manifest_v1.parquet
```

TV1 lọc ABT theo đúng ID được giữ trong manifest và gán `split_set` theo ID. Kiểm định lại ABT đã chốt rồi chạy:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_m3_baseline.py --mode production --split-manifest data/processed/split_manifest_v1.parquet
```

Manifest nhóm xe là đầu vào bắt buộc cho production từ 2026. Dữ liệu cũ vẫn dùng được đường chạy theo ID. Khi ABT thay đổi, lưu phiên bản và manifest riêng. So sánh mô hình trên tập đánh giá chung đã khóa.

### Chạy thử H02 v0.9

Chạy validator với Parquet và FormulaStore của H02. Dùng manifest nhóm xe đã khóa cho chính phiên bản đó, cùng CSV có `id`, `year` và `odometer`. Chỉ dùng kết quả H02 để phát triển.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_m3_baseline.py --mode production `
    --abt data/processed/used_cars_clean_v0_9.parquet --dataset-version v0.9 `
    --split-manifest data/processed/split_manifest_v0_9.parquet `
    --raw-csv data/raw/vehicles.csv --require-raw-exp0 `
    --output-root outputs/h02_v0_9
```

Exp0 và Exp1 dùng cùng các ID trong ABT, cùng giá tham chiếu và cùng cách chia. Exp0 lấy `year` và `odometer` từ CSV gốc. Exp1 dùng tuổi xe tại ngày đăng và số dặm đã chuẩn hóa. Chênh lệch giữa hai cấu hình đo thay đổi đặc trưng trên cohort đã giữ. Khi khóa không ghép đủ hoặc không duy nhất, `--require-raw-exp0` dừng lần chạy trước khi xuất kết quả. Nếu bỏ cờ này, Exp0 có thể dùng proxy từ ABT và ghi rõ trong bảng mô hình.

## Đọc kết quả

- `outputs/models/ablation_summary.csv` lưu đối chứng Exp0–Exp4, gồm tuổi xe bình phương và One-Hot so với Ordinal giả thuyết cho `title_status_clean`.
- `outputs/models/split_assignments.parquet` lưu `listing_id`, nhãn chia tập và phiên bản dữ liệu.
- `outputs/predictions/predictions_test.parquet` chứa dự báo test theo hợp đồng, kèm hãng, tuổi xe, đời xe và sai số phần trăm cho TV2.
- `workbook/Project_Workbook_M3_Tracking.xlsx` có ba sheet TV3. Workbook gốc của TV1 được quản lý riêng.

Mỗi lần chạy cập nhật các đường dẫn đầu ra trên. Giữ bản production riêng khi chạy lại mock. Git bỏ qua dữ liệu, Workbook và kết quả sinh tự động.

Imputer, scaler và encoder chỉ fit trên phần train của từng fold. Target Encoding dùng ánh xạ OOF cho hàng train và ánh xạ train đầy đủ cho validation/test. R² dùng trung bình của tập đang đánh giá và Adjusted R² để trống khi thiếu bậc tự do. Mô hình log nghịch đảo về USD với Duan Smearing. Trong CV hệ số học từ phần train của fold, còn hệ số cuối học từ dự báo OOF trên toàn train. Mô hình được chọn theo RMSE USD trung bình của CV, dùng validation nếu hòa. Chỉ ứng viên đã chốt được chấm test.
