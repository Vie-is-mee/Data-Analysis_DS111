# Nhật ký làm việc M1 · Dữ liệu & Workbook

Mục đích: lưu **quyết định – lý do – số liệu thật** theo thời gian để viết báo cáo, slide
và trả lời vấn đáp. Mỗi lần làm việc thêm một mục mới ở cuối; không xóa mục cũ.
Số liệu lấy từ `data/processed/m1_transform_log.csv` và `workbook/01_Raw_Profile.csv`.

---

## 23/11/2026 · Ngày 1 — Khởi động

- Đọc bảng theo dõi DS111: M1 phụ trách 18 việc (M1-01 → M1-18), hai mốc then chốt:
  bàn giao H02 ngày 29/11 và V1.0 + Gate 2 ngày 05/12.
- Dựng khung thư mục ban đầu (sau đó thay bằng cấu trúc repo của nhóm, xem 24/11).

## 24/11/2026 · Ngày 2 — Nối repo, khung pipeline, kiểm kê dữ liệu thật

### Repo và phân nhánh
- Repo nhóm đã có `main` (cấu hình, DATA_CONTRACT) và `feature/m3-models` (FormulaStore,
  bộ kiểm định bàn giao, mô hình). Tạo nhánh **`feature/m1-data-workbook`** từ nhánh M3.
- Khác biệt kế hoạch ↔ repo (đã báo M3): FormulaStore do M3 viết (M1-03 dùng lại);
  mã hóa (M1-08) chuyển sang M3 trong từng fold; M1 chỉ bàn giao **1 file**
  `used_cars_clean_v1.parquet` (22 cột) + `formula_store_v1.joblib`.

### Khung `src/data` (commit `933180b`)
| Module | Mã việc | Sheet |
|---|---|---|
| `raw_profile.py` | M1-02 | 01 |
| `missing_handler.py` | M1-04 | 02 |
| `formatter.py` | M1-05 | 03 |
| `feature_engineer.py` | M1-07 | 04 |
| `selector.py` | M1-10 | 06 |
| `pipeline_m1.py`, `transform_log.py` | — | 00 |

### Kiểm kê dữ liệu thật (M1-02) — phát hiện chính
- Dữ liệu **không phải bản Kaggle 2021** như kế hoạch giả định: **295.451 dòng × 41 cột**,
  tin đăng **30/09 → 03/11/2026**.
- Quyết định cột: 16 giữ · 6 giữ phụ trợ làm sạch (VIN, title, seller_type,
  address_country, odometer_broken, odometer_rolled_over) · 19 bỏ. Lý do từng cột ở Sheet 01.
- Vấn đề chất lượng (số liệu để đưa lên slide):

| Vấn đề | Số lượng |
|---|---|
| price = 0 / < 500 USD | 10.288 (3,5%) / 12.653 (4,3%) |
| Xe đời 2023–2027 (vượt `max_year=2022` của hợp đồng) | 33.896 (11,5%) |
| condition thiếu | 135.490 (45,9%) |
| cylinders thiếu | 80.167 (27,1%) |
| manufacturer thiếu | 10.792 (3,65%) |
| Đồng hồ hỏng / quay vòng | 2.044 / 2.293 |
| Tin ở Canada (giá có thể CAD) | 535 |
| size, county | trống 100% |

- **Manufacturer thiếu không ngẫu nhiên:** Craigslist chỉ cho chọn 42 hãng; hãng ngoài
  danh sách bị để trống (tìm trong title: Isuzu ~779, Maserati ~370, Hummer ~342,
  Scion ~251, Bentley ~173, Lamborghini ~113, Rolls-Royce ~78…). Nếu gán `unknown` sẽ trộn
  xe siêu sang với xe phổ thông.
- Quyết định: giữ `manufacturer_clean` ở **cấp hãng**, không gộp lên tập đoàn
  (Porsche và VW cùng tập đoàn nhưng giá/khấu hao khác hẳn). Đề xuất bảng tra phụ
  `brand_origin / brand_group / brand_tier` cho M2 (ANOVA theo xuất xứ) — chưa làm.

### Đối chiếu phạm vi môn học (slide Bài 1–5)
- Phần M1 nằm trong Bài 1 (CRISP-DM, ABT, hiểu dữ liệu), Bài 2 (xử lý khuyết, định dạng,
  tạo/biến đổi đặc trưng, loại đặc trưng), Bài 4 (IQR/boxplot, tương quan).
- Điều chỉnh để bám slide: lọc giá tầng 2 dùng **IQR** (Bài 4) thay phân vị;
  condition thiếu → nhãn cố định `unknown` (Bài 2); Missingness Indicator =
  `condition_clean == "unknown"` (không cần thêm cột).
- Ngoài phạm vi (cần giải thích nếu giữ): VIF (slide dạy tương quan/heatmap),
  FormulaStore/validator (công cụ nhóm), Target Encoding OOF, Duan Smearing, MAPE (phần M3).
- Trình bày báo cáo theo CRISP-DM: Sheet 01 = Data Understanding; Sheet 02–06 =
  Data Preparation; file parquet bàn giao = **ABT**.

### Cập nhật pipeline theo dữ liệu thật (việc 2)
| Thay đổi | Module | Căn cứ slide |
|---|---|---|
| Cứu manufacturer từ `title` ("2019 Maserati Ghibli" → maserati), có 19 hãng ngoài danh sách; chỉ tin tiêu đề mở đầu bằng năm | missing_handler | Bài 2 · tạo đặc trưng từ chuỗi |
| Loại `odometer_broken` / `odometer_rolled_over` = 1 | missing_handler | Bài 1/4 · giá trị không hợp lệ |
| Loại tin Canada | formatter | Bài 1 · tính nhất quán đơn vị |
| Gộp alias hãng: rover → land rover; giữ `alfa-romeo`, `aston-martin` | formatter | Bài 2 · định dạng bằng replace |
| Lọc giá tầng 2: IQR (k=1,5) trên log(price) theo hãng, hãng < 30 tin không lọc — **tắt mặc định** (`--price-group-filter`) | formatter | Bài 4 · boxplot/IQR |
| Bỏ trùng VIN, giữ tin mới nhất — **tắt mặc định** (`--dedup-vin`) | formatter | Bài 2/3 · loại trùng lặp |

Test bắt được 2 lỗi trong lúc làm (ghi lại cho phần "kiểm thử" của báo cáo):
tiêu đề "Land Rover…" không khớp khi dữ liệu chưa có nhãn `land rover`; và
"Smart key remote…" bị nhận nhầm là hãng `smart` → thêm điều kiện tiêu đề mở đầu bằng năm.

### Kết quả chạy trên dữ liệu thật (cấu hình hiện tại: max_year = 2022)
| Bước | Dòng bị loại / sửa |
|---|---|
| Thiếu year | −26 |
| Cứu manufacturer từ title | **4.050 / 10.792** dòng thiếu được cứu |
| Thiếu odometer | −4.023 |
| odometer_broken / rolled_over | −2.023 / −2.170 |
| Tin Canada | −535 |
| price ngoài 500–150.000 | −12.825 |
| year ngoài 1950–2022 | **−29.835** (xe đời mới) |
| odometer > 500.000 | −547 |
| cylinders điền median nhóm | 63.703 ô |
| **Còn lại** | **243.467 / 295.451 (82,4%)** |

- Bộ kiểm định của M3: **33/33 PASS — READY FOR HOT-SWAP**; 5 công thức trong FormulaStore.
- Hệ quả của `max_year = 2022`: `car_age` nhỏ nhất = 4 (không có xe dưới 4 tuổi) —
  bằng chứng cần đổi lên 2027.
- So sánh các cờ (bảng đối chứng cho Sheet 03 / rủi ro R1):

| Cấu hình | Số dòng | % giữ | Giá trung vị |
|---|---|---|---|
| Mặc định | 243.467 | 82,4% | 11.749 |
| + IQR theo hãng | 239.201 | 81,0% | 11.900 |
| + Bỏ trùng VIN | 177.191 | 60,0% | 9.999 |
| + Cả hai | 174.189 | 59,0% | 10.000 |

(Số đã cập nhật sau lượt rà soát bên dưới.)

- **Tin trùng VIN** là đại lý đăng chéo: 29.690 xe xuất hiện trong 112.831 tin
  (trung vị 3 tin/xe, tối đa 167); 97,8% do đại lý; 83,6% nhóm cùng giá; 39,2% đăng
  ở nhiều vùng. → Lập luận mạnh để bỏ trùng (tránh cùng xe ở cả train và test),
  nhưng tụt dưới mốc 200.000 dòng khi còn `max_year = 2022`.

### Câu hỏi mở với M3 (đã gửi)
1. Đổi `max_year` 2022 → 2027.
2. Bật bỏ trùng VIN? (−81.671 dòng; cân nhắc cùng mốc 200.000 dòng).
3. `split_set` dùng `assign_production_splits`; `cylinders_num` điền median theo nhóm.
4. Thêm `seller_type` vào hợp đồng?
5. Phần ngoài phạm vi môn học của M3 (Target Encoding, Duan Smearing, MAPE).

### Ý tưởng cho slide
- 1 slide "Dữ liệu thật khác giả định": 2021 → 2026, 26 → 41 cột, 426k → 295k dòng.
- 1 slide phễu làm sạch (waterfall/funnel) từ bảng "Kết quả chạy" ở trên.
- 1 slide "Thiếu dữ liệu không ngẫu nhiên": hãng ngoài danh sách Craigslist.
- 1 slide "Tin đăng lại": 97,8% đại lý, ví dụ 1 VIN đăng 167 lần.

### Rà soát trước khi push — 4 lỗi đã sửa
| # | Vấn đề | Sửa | Ảnh hưởng |
|---|---|---|---|
| 1 | `01_Raw_Profile.csv` (đưa lên Git) chứa **SĐT thật** lấy từ `description` trong cột ví dụ | Ẩn ví dụ của 11 cột nhạy cảm (description, VIN, url, địa chỉ, tọa độ…); test kiểm tra | 0 SĐT còn sót |
| 2 | Bỏ trùng VIN chạy **trước** lọc giá: nếu tin mới nhất có giá 0 thì mất luôn cả chiếc xe | Chuyển bước bỏ trùng ra **sau** lọc miền giá/năm/số dặm; thêm test | Giữ thêm 72 xe (177.119 → 177.191) |
| 3 | Sắp xếp theo ngày đăng không ổn định khi trùng giờ → chạy lại có thể giữ tin khác | Sắp xếp ổn định theo (ngày đăng, id) | Kết quả tái lập được |
| 4 | IQR lọc cả nhóm `unknown` (trộn nhiều hãng) → "lệch so với cùng hãng" vô nghĩa | Không lọc nhóm `unknown` | IQR loại 4.266 thay vì 4.357 |
| — | Cảnh báo số dòng dùng ngưỡng 5.000 (mẫu mồi) | Đổi sang mục tiêu 200.000 (R1) | Chỉ là thông báo |

Kiểm thử: 6/6 test pass; dữ liệu thật vẫn 243.467 dòng, kiểm định 33/33 PASS.
Bài học cho báo cáo: thứ tự các bước làm sạch ảnh hưởng kết quả (lọc trước, bỏ trùng sau).

## 24/11/2026 · Ngày 2 (tiếp) — M1-01 Workbook 9 sheet + exporter (phần đầu M1-14)

- Viết `src/data/workbook_exporter.py`: `python -X utf8 -m src.data.workbook_exporter`
  tạo `workbook/Project_Workbook.xlsx` (file .xlsx không lên Git → chia sẻ qua Drive).
- **Chỉ thay sheet 00–07**, giữ nguyên các sheet 08* do M3 ghi vào cùng file
  (`src.models.workbook_sync` cũng mở file có sẵn và chỉ thay sheet của M3). Có test kiểm tra.

| Sheet | Nội dung | Căn cứ |
|---|---|---|
| 00_Log | Thông tin phiên bản + toàn bộ log biến đổi | CRISP-DM (Bài 1) |
| 01_Raw_Profile | Hồ sơ 41 cột + giữ/bỏ + lý do | Bài 1 · Hiểu dữ liệu |
| 02_Missing | Bảng cách xử lý khuyết theo cột + log | Bài 2 |
| 03_Format | Log chuẩn hóa / lọc miền / lọc ngoại lệ | Bài 2, Bài 4 |
| 04_Features | Công thức FormulaStore, thống kê trước/sau (mean, std, skewness) | Bài 2 |
| 05_Encoding | Kế hoạch mã hóa từng cột (M3 thực hiện trong fold) | Bài 2 |
| 06_Selection | Từ điển 22 cột; tương quan Pearson/Spearman với log_price kèm p-value; ma trận tương quan tô màu như heatmap; VIF | Bài 2, Bài 4 |
| 07_Sample_Comparison | 300 listing_id cố định (seed 42), gốc ↔ sạch, cột "Trường đã thay đổi" | M1-13 |
| 08_Experiment_Tracking | Chỗ trống cho M3 | — |

### Phát hiện từ Sheet 06 (đưa vào báo cáo/slide)
- **year và car_age tương quan r = −1 tuyệt đối, VIF = ∞**: mọi tin đăng trong năm 2026
  nên car_age = 2026 − year. Hai biến mang cùng một thông tin → mô hình tuyến tính
  chỉ dùng **một** (đề xuất car_age). Báo M3.
- car_age – log_price: Pearson chỉ **−0,185** nhưng Spearman **−0,448**. Quan hệ đơn điệu
  nhưng không tuyến tính: xe cổ (đời cũ) lại đắt, và `max_year = 2022` đã cắt hết xe mới.
  → Lý do dùng Spearman (Bài 4) và thử car_age² (Bài 5 · hồi quy đa thức).
- odometer – log_price: Pearson −0,446, Spearman −0,491 (yếu tố mạnh nhất trong nhóm số).
- miles_per_year gần như không tương quan với giá (r = 0,045).
- p-value ≈ 0 với mọi cặp vì n = 243.467 (Bài 4: mẫu lớn → r nhỏ vẫn có ý nghĩa thống kê;
  cần đọc độ lớn r chứ không chỉ p).

### Sheet 07 (300 mẫu)
- Không đổi: 120 dòng; condition khuyết → unknown: 134; cylinders điền: 68; drive: 63;
  manufacturer: 15 (ví dụ: khuyết → maserati từ title; rover → land rover).
- Sửa trong lúc làm: cylinders ban đầu bị đánh dấu "đổi" ở cả 300 dòng chỉ vì định dạng
  ('6 cylinders' → 6) → so sánh sau khi tách số.

Kiểm thử: 8/8 test M1 pass (thêm 2 test exporter: giữ sheet M3, mẫu Sheet 07 cố định).
