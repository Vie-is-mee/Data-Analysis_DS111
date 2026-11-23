# DS111 – Phân tích dữ liệu và định giá xe cũ Craigslist

Dự án nhóm 3 người, Python 3.11 chạy trên CPU, sử dụng bộ dữ liệu Used Cars – Craigslist (khoảng 426.000 tin rao). Nghiên cứu ảnh hưởng của tuổi xe, số dặm, hãng xe, tình trạng pháp lý và các đặc trưng khác đến giá USD; so sánh mô hình tuyến tính (OLS, Ridge) với Decision Tree, Random Forest và LightGBM bằng thực nghiệm loại bỏ/thêm từng kỹ thuật (ablation).

> **Tình trạng hiện tại – BƯỚC 1/4:** Mới hoàn tất các tài liệu và tệp cấu hình nền tảng: `.gitignore`, `requirements.txt`, `configs/project_config.py`, `docs/DATA_CONTRACT.md`, `README.md`. Bộ sinh dữ liệu mồi, kiểm tra bàn giao, cấu trúc công thức, huấn luyện, kiểm thử và bảng điều khiển **chưa được triển khai**; không chạy lệnh gọi những mô-đun chưa tồn tại.

## Vai trò và vùng sở hữu

| Thành viên | Nhánh | Vùng thực hiện |
|---|---|---|
| TV1 – xử lý dữ liệu & Workbook | `feature/m1-data-workbook` | `src/data/`, `workbook/` |
| TV2 – khám phá dữ liệu & Power BI | `feature/m2-eda-dashboard` | `src/analysis/`, `dashboard/` |
| TV3 – mô hình, thực nghiệm & điều phối | `feature/m3-models` | `src/models/`, `scripts/`, `outputs/` |

`src/core/`, `configs/`, `docs/`, `README.md`, `requirements.txt` là phần dùng chung, TV3 điều phối thay đổi. Mỗi người làm trên nhánh riêng, gửi yêu cầu gộp thay đổi (Pull Request/PR), rà soát và kiểm thử trước khi nhập vào `main`.

## Cây thư mục dự kiến (không phải tất cả đã có)

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── configs/
│   ├── __init__.py
│   └── project_config.py
├── docs/DATA_CONTRACT.md
├── data/
│   ├── raw/vehicles.csv             # tự tải; không đưa vào Git
│   ├── mock/mock_clean_sample.parquet
│   └── processed/used_cars_clean_v1.parquet
├── workbook/Project_Workbook.xlsx
├── src/
│   ├── core/                       # formula store, sinh mẫu, kiểm định
│   ├── data/                       # TV1
│   ├── analysis/                   # TV2
│   └── models/                     # TV3
├── scripts/run_m3_baseline.py
├── outputs/
│   ├── models/
│   └── predictions/predictions_test.parquet
└── tests/test_baseline_smoke.py
```

## Năm lệnh khởi động trên Windows PowerShell

Chạy trong thư mục gốc của dự án; cần cài Git và Python 3.11 trước. Đây là **5 lệnh hoạt động ở Bước 1**, chưa phải 5 lệnh chạy toàn bộ huấn luyện:

```powershell
git clone https://github.com/Vie-is-mee/Data-Analysis_DS111.git
cd Data-Analysis_DS111
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "from configs.project_config import CONFIG; print(CONFIG.root); print(CONFIG.clean_parquet)"
```

TV1 và TV2 thực hiện cùng năm lệnh trên, rồi chuyển sang nhánh riêng (chỉ sau khi nhánh tồn tại): `git switch feature/m1-data-workbook` hoặc `git switch feature/m2-eda-dashboard`. TV3 chọn `feature/m3-models`. Nếu nhánh chưa tồn tại, người phụ trách tạo bằng `git switch -c <tên-nhánh>` và `git push -u origin <tên-nhánh>`.

## Trình tự chia việc và bàn giao

1. **Bước 1 – Nền tảng:** Khóa cấu hình, hợp đồng dữ liệu, quy tắc làm việc (đã soạn).
2. **Bước 2 – Hạ tầng chung:** TV3 tạo `FormulaStore` tách giá trị `pandas.Series` và thông tin công thức, sinh tập mồi 5.000 dòng từ nguồn đã tải, xây `validate_handoff.py`; TV1 và TV2 có thể phát triển trên tập mồi.
3. **Bước 3 – Hoàn thiện ba nhánh:** TV1 tạo tập sạch `used_cars_clean_v1.parquet` và Workbook; TV2 khám phá phân phối/tương quan từng `Series` và Dashboard; TV3 xây `ModelRegistry`, phép đo, mã hóa tránh rò rỉ, OLS/Ridge/Tree/Forest/LightGBM, đối chứng Exp 0–4.
4. **Bước 4 – Kiểm thử & tích hợp:** Chạy kiểm tra tự động, nghiệm thu bản sạch, ghép dữ liệu thật, tính phần dư trên test, xuất `predictions_test.parquet`, cập nhật Sheet 08 và chuẩn bị báo cáo.

**Quy định đo lường:** `R² = 1 - SS_res/SS_tot`, có đối chứng dự đoán trung bình tập huấn luyện. Nếu học `log(price)`, dự báo USD phải áp dụng Duan Smearing tính từ phần dư train; đối chiếu RMSE, MAE, MAPE trên USD. Xếp hạng ảnh hưởng các yếu tố phải dùng cùng tập mẫu, loại bỏ đa cộng tuyến bằng VIF khi phù hợp và chuẩn hóa cùng thang đo. Không dùng dữ liệu test để học cách mã hóa, lọc đặc trưng hay chỉnh mô hình.

Chi tiết cột dữ liệu, quy tắc phân chia, phép biến đổi, đầu ra và phương án bảo vệ nhánh nằm ở [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md). Số liệu và kết quả trong báo cáo phải là số đo chạy thực tế, không dùng con số minh họa làm kết luận.
