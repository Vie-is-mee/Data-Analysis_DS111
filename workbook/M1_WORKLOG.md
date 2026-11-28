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

## 24/11/2026 · Ngày 2 (tiếp) — Làm theo phản hồi của M3

M3 trả lời (một phần): M1-08 → TV1 chuẩn hóa nhãn + giải thích ma trận mã hóa, **TV3 fit encoder**
(Exp3A/3B); M1-11 vẫn là ABT sạch của TV1. Đồng ý thêm `condition_missing`; cứu hãng từ title
được nếu **lưu nguồn suy diễn và kiểm tra mâu thuẫn**; IQR theo hãng **chỉ gắn cờ rồi kiểm tra**,
ngưỡng học trên train, giữ nguyên test; trình bày chính bám slide (OLS/Ridge, One-Hot hãng,
Ordinal condition, Label Encoding làm đối chứng, MSE/RMSE/R² trên USD); VIF, Target Encoding,
Duan, MAPE, Adj-R², LightGBM để phần mở rộng; giữ 70/15/15.

### Thay đổi
- Parquet bàn giao có **22 cột hợp đồng + 4 cột cờ** (validator cho phép cột thừa):

| Cột | Ý nghĩa | Số liệu thật (243.467 dòng) |
|---|---|---|
| `condition_missing` | 1 = condition gốc rỗng, trước khi điền 'unknown' | 106.477 (43,7%) |
| `manufacturer_source` | original / title / none | 235.423 / 3.267 / 4.777 |
| `manufacturer_title_conflict` | 1 = hãng gốc ≠ hãng đọc được từ title | 504 |
| `price_outlier_iqr` | 1 = ngoài 1,5·IQR của log(price) cùng hãng | 4.553 (1,87%) |

- **IQR: bỏ lọc, đổi thành gắn cờ** (`formatter.flag_price_outliers`), chạy sau khi có `split_set`.
  Q1/Q3 chỉ tính trên dòng train; hãng có < 30 dòng train hoặc 'unknown' dùng ngưỡng chung của
  toàn train (fallback, 4.961 dòng). Bỏ tùy chọn `--price-group-filter`. Số dòng không đổi.
  Tỉ lệ cờ đều giữa các tập (train 1,87% / val 1,92% / test 1,83%); 3.825 ca giá thấp
  bất thường, 728 ca giá cao. Có test chứng minh đổi giá val/test không làm đổi cờ train.
- Sheet 05: hãng → One-Hot (nhánh chính), Label Encoding làm đối chứng, Target Encoding = mở rộng;
  condition → Ordinal + cờ. Sheet 06: VIF ghi rõ là phần mở rộng; từ điển có thêm 4 cột cờ.

### Phát hiện khi kiểm tra mâu thuẫn hãng (đưa vào báo cáo)
- Lần chạy đầu báo 4.613 ca "mâu thuẫn", nhưng xem cặp (hãng gốc, 2 từ đầu title) thì
  **3.824 ca là `ram` ↔ "Dodge Ram"** (Ram tách khỏi Dodge năm 2010, tin xe cũ vẫn ghi Dodge)
  và ~120 ca `toyota` ↔ "Scion" (thương hiệu con của Toyota). Đây không phải lỗi.
  → Thêm `BRAND_FAMILY` (dodge≈ram, scion≈toyota) khi so sánh → còn **504 ca** lệch thật
  (ví dụ người bán chọn "mini" cho xe Honda Odyssey — có lẽ nhầm với minivan).
- Chỉ gắn cờ, không sửa: không đủ căn cứ biết bên nào đúng.
- `condition_missing` trùng khớp hoàn toàn với `condition_clean == 'unknown'` trên dữ liệu thật
  (không có nhãn condition lạ), khác nhau chỉ ở dữ liệu giả lập test.

Kiểm thử: 23/23 test pass (thêm test cờ IQR, ngưỡng chỉ học từ train, condition_missing,
nguồn hãng, gia đình hãng). Validator 33/33 PASS; giữ 243.467/295.451 dòng như trước.

Còn chờ M3: max_year 2027, bật dedup VIN, cách điền cylinders, seller_type. Cần M3 ghi
4 cột cờ mới vào DATA_CONTRACT (vùng `docs/` của M3, M1 không tự sửa).

## 24/11/2026 · Ngày 2 (tối) — Khớp DATA_CONTRACT 1.1 của M3 (nhánh `feature/m3-data-2026`)

Đọc bảng theo dõi (Drive, M3 sửa 24/11) + nhánh mới của M3. M3 đã chốt: **năm model tới 2027**
và `year ≤ posting_year + 1`; cờ `condition_missing`, `cylinders_was_missing` dạng **boolean**;
`seller_type` (private/dealer/unknown, `owner` → `private`); xử lý VIN bằng **chữ số kiểm tra
49 CFR 565.15** (`src/core/vin_reposts.py`) và chia tập theo **nhóm xe** (`split_manifest.py`);
One-Hot cho cả `manufacturer_clean` và `state`. M1-07 có thêm "tuổi xe bậc 2".

### Đã làm
- Đổi 4 cờ sang `bool` (trước là int8, validator mới của M3 sẽ FAIL).
- Thêm `cylinders_was_missing` (ghi sau khi tách số và quy xe điện = 0, trước khi điền median)
  để TV3 điền lại trong từng fold: 63.703 dòng.
- Thêm `seller_type`: dealer 174.510 / private 68.957. Giá trung vị dealer 13.495 vs private 7.500.
- Thêm quy tắc `year ≤ posting_year + 1` (hiện chưa loại dòng nào vì max_year vẫn 2022 trên nhánh M1).
- **M1-07**: thêm `car_age_squared` vào FormulaStore (6 công thức), đặt đúng tên M3 dùng trong
  ablation → không lệch. Hồi quy log_price ~ car_age: R² = 0,034; thêm car_age²: **R² = 0,192**
  (đỉnh parabol ≈ 32 năm: xe trên ~32 tuổi lại tăng giá — xe cổ). Đưa vào slide Bài 5.
- Validator: bản hiện tại 33/33 PASS; **bản 1.1 của M3 (chạy trong worktree tạm) 36/36 PASS**.

### Ước lượng khi chuyển max_year = 2027 (chạy thử, chưa đổi config — config là vùng M3)
| | max_year 2022 | max_year 2027 |
|---|---|---|
| Số dòng | 243.467 | **272.563** (+29.096) |
| Giá trung vị | 11.749 | 12.995 |
| car_age min | 4 | 0 |
Xe đời 2023–2027: 9.587 / 8.388 / 6.807 / 3.985 / 329. Không dòng nào vi phạm `year ≤ posting_year + 1`.

### Còn lại (cần M1 làm tiếp)
1. Gộp nhánh M3 (`feature/m3-data-2026`) vào nhánh M1 để có max_year 2027 + validator/manifest mới.
2. Xuất `data/processed/vin_candidates_local.parquet` (listing_id, VIN, posting_date sau lọc nghiệp
   vụ, trước gộp tin) — chỉ lưu nội bộ, không lên Git. M3 tạo manifest; M1 lọc ABT theo ID được giữ
   và lấy `split_set` từ manifest (thay `assign_production_splits` + bỏ `dedup_by_vin` của M1).
3. Báo số dòng + giá trung vị trước/sau gộp VIN **theo seller_type** (yêu cầu M1-11 / hợp đồng).

## 24/11/2026 · Ngày 2 (khuya) — Gộp nhánh M3, năm 2027, luồng manifest VIN

- Gộp `origin/feature/m3-data-2026` vào nhánh M1 (commit `d31f647`, không xung đột).
  Có `max_year = 2027`, validator 1.1, `vin_reposts.py`, `split_manifest.py`.
  PR #1 đã được gộp vào `feature/m3-models` (`228c7b8`).
- **Bỏ cách gộp VIN riêng của M1** (`dedup_by_vin`, `--dedup-vin`): theo hợp đồng 1.1, M3 quyết định
  bằng chữ số kiểm tra VIN. Pipeline chạy 2 lượt:
  1. `pipeline_m1` → ABT phát triển + `vin_candidates_local.parquet` (listing_id, VIN, posting_date
     sau lọc nghiệp vụ, trước gộp; trong `data/processed/` đã gitignore).
  2. M3: `split_manifest --vin-listings ...` → manifest.
  3. `pipeline_m1 --split-manifest <file>` → chỉ giữ ID trong manifest, `split_set` lấy từ manifest;
     manifest cũ (có ID không còn trong ABT) bị từ chối.
- Sheet 03 có bảng trước/sau gộp VIN theo `seller_type` (`m1_vin_report.csv`).

### Số liệu thật (manifest chạy thử bằng code của M3, chưa phải manifest chính thức)
| | Trước gộp | Sau gộp |
|---|---|---|
| Số dòng (2027) | 272.563 | **194.083** (65,7% dữ liệu gốc) |
| Giá trung vị | 12.995 | 10.999 |

| seller_type | Trước | Sau | Bị gộp | % | Trung vị trước → sau |
|---|---|---|---|---|---|
| dealer | 201.081 | 124.027 | 77.054 | **38,3%** | 14.999 → 13.500 |
| private | 71.482 | 70.056 | 1.426 | 2,0% | 7.600 → 7.500 |

Trạng thái VIN: qua chữ số kiểm tra 160.938; thiếu 109.455; sai dạng 1.594; không qua kiểm tra 571;
giữ chỗ 5. Split theo nhóm: 135.842 / 29.136 / 29.105. Validator 1.1: 36/36 PASS.

**Ý nghĩa cho báo cáo:** đại lý đăng lại cùng một xe rất nhiều lần (38% tin đại lý là tin lặp). Không gộp
thì cùng một xe nằm ở cả train và test → R² ảo. Sau gộp còn 194.083 dòng: dưới mốc 200k M1 tự đặt
nhưng trên ngưỡng R1 của nhóm (100k). Giá trung vị giảm vì tin lặp chủ yếu là xe đại lý giá cao.

### Lỗi phát hiện khi rà soát
- **2 VIN thật từ dữ liệu nằm trong `tests/test_m1_pipeline_smoke.py`** (từ commit `1f2c9f7`, đã lên
  GitHub qua PR #1). Đã thay bằng VIN mẫu công khai (qua chữ số kiểm tra, kiểm tra không có trong
  dữ liệu). Lịch sử Git cũ vẫn còn → báo M3 quyết định có cần viết lại lịch sử không.
  Bài học: quét cả chuỗi 17 ký tự giống VIN khi rà diff, không chỉ SĐT/URL.

Kiểm thử: 32/32 (cả test M3), thêm test manifest đặt split_set, báo cáo VIN theo người bán,
từ chối manifest cũ.

## 25/11/2026 — Quyết định của M3 về dữ liệu thật trong lịch sử Git

- M3: M1 đẩy bản đã thay dữ liệu mẫu lên nhánh M1; M3 kiểm tra để bản mới nhất của hai nhánh
  (`feature/m3-models`, `feature/m1-data-workbook`) sạch, chạy test rồi **squash vào `main`**.
  Dữ liệu thật trong lịch sử PR #1 được ghi nhận riêng, thống nhất sau nếu cần xóa.
  (Lúc kiểm tra 25/11: `main` và `feature/m3-data-2026` sạch; M3 đã gộp PR #3 vào `main`.)
- Quét lại TOÀN BỘ file theo dõi bởi Git (không chỉ diff) với danh sách VIN của dữ liệu gốc →
  phát hiện thêm **số điện thoại thật** (xuất hiện trong 284 tin, có lẽ của một đại lý) trong
  test ẩn dữ liệu cá nhân của `tests/test_m1_pipeline_smoke.py` (cũng từ commit `1f2c9f7`).
  Đã thay bằng số hư cấu dải 555-01xx. Chuỗi `12345678901234567` trong code M3 là VIN giữ chỗ,
  không phải dữ liệu cá nhân.
- Ghi nhận cho M3: commit `1f2c9f7` chứa 2 VIN thật + 1 số điện thoại thật.

## 26/11 — Gửi danh sách VIN cho TV3 để khóa manifest v0.9
- Bản `vin_candidates_local.parquet` cũ chỉ còn 769 dòng VIN rỗng do một lần chạy test ghi đè, nên phải chạy lại lượt 1 trên dữ liệu thật.
- Bản mới có 272.563 dòng, 163.108 dòng có VIN, `listing_id` không trùng, gồm 3 cột listing_id/VIN/posting_date.
- Chạy thử bằng code TV3 ra 194.083 dòng (train 135.842, val 29.136, test 29.105), khớp lần đo 25/11.
- File được gửi riêng qua Drive, không đưa lên Git vì có VIN. Mỗi khi đổi luật làm sạch trước bước VIN thì phải xuất lại file này.
- Bài học: chạy test có thể ghi đè file trong `data/processed/`. Kiểm tra số dòng trước khi gửi file cho người khác.

## 26/11 — ABT v0.9 cho H02 (dùng manifest TV3 đã khóa)
- TV3 gửi `split_manifest_v0_9.parquet`: 194.083 dòng, `listing_id` không trùng, mọi ID đều có trong danh sách VIN của M1.
- Lượt 2: `pipeline_m1 --split-manifest ... --out used_cars_clean_v0_9.parquet --store formula_store_v0_9.joblib`.
  - ABT có 194.083 dòng × 29 cột, không còn cột VIN, split khớp manifest 100%.
  - Chia: train 135.842 / val 29.136 / test 29.105.
- Gộp tin theo người bán:
  - dealer: 201.081 → 124.027 (gộp 38,3%), giá trung vị 14.999 → 13.500.
  - private: 71.482 → 70.056 (gộp 2,0%), giá trung vị 7.600 → 7.500.
  - Tổng: 272.563 → 194.083 (gộp 28,8%), giá trung vị 12.995 → 10.999.
  - Ý cho báo cáo: đại lý đăng lại cùng một xe rất nhiều lần, nên nếu không gộp thì mô hình bị lệch về xe đại lý giá cao.
- Validator: 36/36 PASS, READY FOR HOT-SWAP. Workbook xuất lại có bảng VIN theo người bán ở Sheet 03.
- **Đã tìm ra nguyên nhân file VIN bị hỏng:** `test_m1_pipeline_passes_handoff_validation` gọi `main()` mà không truyền `--vin-out`/`--vin-report`, nên mỗi lần chạy test lại ghi đè file VIN thật và xóa báo cáo VIN thật.
  - Sửa: test truyền đường dẫn trong `tmp_path`.
  - Kiểm tra bằng md5 trước/sau khi chạy pytest: file thật không đổi. 32/32 test PASS.

## 26/11 — Báo cáo gộp VIN: thêm năm model và odometer theo người bán
- `vin_report` có thêm năm trung vị và odometer trung vị trước/sau gộp. Phần này nằm ngoài yêu cầu hợp đồng 1.1, vốn chỉ đòi số dòng và giá.
- Kết quả trên ABT v0.9:

| Người bán | Năm (trước → sau) | Odometer (trước → sau) | Giá (trước → sau) |
|---|---|---|---|
| dealer | 2016 → 2016 | 104.044 → 106.712 | 14.999 → 13.500 |
| private | 2011 → 2011 | 128.000 → 128.000 | 7.600 → 7.500 |
| Tổng | 2015 → 2015 | 109.131 → 113.430 | 12.995 → 10.999 |

- **Ý cho báo cáo (hiệu ứng cơ cấu):**
  - Trong từng nhóm, giá chỉ giảm nhẹ: dealer −10%, private −1%. Nhưng giá trung vị toàn bộ giảm tới 15%.
  - Lý do: tỷ trọng tin dealer giảm từ 73,8% xuống 63,9%. Dealer có giá cao hơn nên khi tỷ trọng giảm, giá chung kéo xuống.
  - Tin dealer bị đăng lại thường là xe giá cao hơn và đi ít hơn một chút. Năm model không đổi.
  - Kết luận: không gộp thì mẫu bị lệch về phía tin đại lý.
- Kiểm tra: 32/32 test PASS, validator 36/36 PASS. ABT v0.9 không đổi, chỉ `m1_vin_report.csv` có thêm cột. Workbook Sheet 03 đã xuất lại.

## 27/11 — M1-08: Bảng nhãn chuẩn ở Sheet 05
- `build_label_table` liệt kê cho từng cột phân loại: mỗi nhãn sau chuẩn hóa, mã Ordinal (nếu có), số dòng, % dòng, số dòng train và các nhãn gốc đã gộp vào.
  - Các cột: condition, title_status, year_era, fuel, transmission, drive, type, seller_type, manufacturer.
  - Nhãn gốc được lấy thẳng từ luật trong code (`MANUFACTURER_ALIASES`, owner → private), nên bảng không lệch khỏi code.
- Trên ABT v0.9 bảng có 108 dòng nhãn. Những điểm đáng đưa vào báo cáo:
  - **condition:** `unknown` chiếm 37,7%, là nhãn lớn nhất. Nhãn này không có mã thứ tự và được mã hóa riêng bằng cờ `condition_missing`.
  - **title_status:** `clean` chiếm 92,3%. `parts only` chỉ có 89 tin (0,05%) → nhãn hiếm. Mã thứ tự chỉ là thực nghiệm, Exp 3 sẽ đối chứng với One-Hot.
  - **drive:** `unknown` chiếm 24,9%, còn nhiều hơn rwd (14,6%).
  - **manufacturer:** 60 hãng. Có 4 nhóm cách viết được gộp: chevy/chev → chevrolet; benz/mercedes → mercedes-benz; vw/volkswagon → volkswagen; rover/landrover → land rover. Lamborghini chỉ có 3 tin (2 tin ở train) → TV3 cần xử lý nhãn hiếm hoặc chưa từng gặp khi One-Hot.
  - **seller_type:** chỉ còn dealer và private (nhãn owner đã gộp vào private).
- `state` (51 bang) không đưa vào bảng vì chỉ chuyển về chữ thường.
- Kiểm tra: thêm test mỗi cột đủ nhãn và tổng số dòng đúng, mã condition tăng dần, `unknown` không có mã, chevy được gộp vào chevrolet. 71/71 test PASS. Đã xuất lại Workbook từ ABT v0.9.

## 28/11 — Làm theo yêu cầu TV3 sau khi xem phần dư H02 (chuẩn bị ABT v1.0)
**1. Lọc tin giá ảo, tin rác (tầng 1b, `formatter.filter_fake_listings`).** Bước này chạy TRƯỚC khi xuất VIN.

| Luật | Số dòng bị loại (trước gộp VIN) |
|---|---|
| Giá giữ chỗ 123456 | 12 |
| price == odometer (ví dụ 143.412) | 134 |
| model là quảng cáo chung ('all makes and models', 'any make', 'all types', 'cars for sale', 'public auto auction') | 238 |

- Tổng: 272.563 → 272.179 dòng (−384).
- Đuôi giá ≥ 100k chỉ giảm 1.005 → 989. Phần lớn đuôi cao là giá đại lý thật (139.997, 134.997...). Vì vậy tỷ trọng MSE của đuôi có thể giảm ít, cần TV3 đo lại.
- Tin model rác thực ra là tin **giá thấp** (trung vị 500–3.000), không nằm ở đuôi cao. Vẫn loại vì chúng không mô tả một chiếc xe cụ thể.
- Chưa lọc giá 12345 (64 tin, 29 tin là cùng một chiếc SL600 đời 1995). Giá này hợp lý cho xe đó nên cần TV3 quyết thêm. Các giá lặp chữ số như 22222 hay 33333 cũng chưa lọc.
- Hạn chế: vài tin price == odometer là số tròn (ví dụ 10.000/10.000) và có thể trùng thật. Ảnh hưởng chỉ 0,05% số dòng.

**2. Cứu thêm hãng xe tải thương mại.**
- Thêm freightliner (gồm frht), international (gồm intl, navistar), kenworth, peterbilt, hino, cùng mack, western star, sterling cho đủ nhóm.
- Cứu thêm từ đầu cột model khi title không có năm, ví dụ '*kenworth* t680'. `manufacturer_source` có thêm giá trị 'model'.
- Trên file gốc: số dòng còn thiếu hãng giảm 6.742 → 3.990 (−41%). Cứu từ title 5.597 dòng (trước là 4.050), cứu từ model 1.205 dòng.
- Trong ABT trước gộp VIN: freightliner 536, international 676, hino 151, peterbilt 120, kenworth 100, sterling 30, mack 24, western star 9. Giá trung vị 25–49 nghìn USD, không nằm ở đuôi cao.

**3. Ba lỗi nhỏ.**
- `feature_engineer`: LaTeX của car_age² bị một ký tự TAB thay cho `\t`, do bản vá trước dùng chuỗi thường. Đã sửa và thêm test để không công thức nào chứa ký tự điều khiển.
  - Lý do của car_age² bỏ số cứng −0,45/−0,19 vì đã cũ. Số mới trước gộp VIN: Spearman −0,55, Pearson −0,30; trên v0.9: −0,53 và −0,24.
- Sheet 06 bỏ `.corr()` và scipy, chuyển sang dùng `correlate_pairs` của TV2.
  - Số liệu khớp scipy đến 1e-12.
  - Bảng có thêm KTC 95% của Pearson và phương pháp TV2 đề xuất.
- Pipeline xuất thêm `vehicles_exp0_thin.csv`: 3 cột id/year/odometer nguyên văn, 295.451 dòng, id duy nhất, 6,9 MB.
  - TV3 truyền file này vào `--raw-csv` để Exp0 chạy bằng dữ liệu gốc thật thay cho proxy.
  - Đã thử `read_raw_exp0_features`: ghép đủ 272.179/272.179 id.

**Kiểm tra.**
- 73/73 test PASS. Test không chạm vào `data/processed` (kiểm bằng md5).
- Validator 36/36 PASS trên ABT trước gộp VIN.
- Danh sách VIN mới có 272.179 dòng (163.086 dòng có VIN); manifest thử ra 193.707 dòng.
- Gửi TV3 `vin_candidates_local.parquet` và `vehicles_exp0_thin.csv` để khóa `split_manifest_v1.parquet`.

## 28/11 (chiều) — Làm theo 6 mục TV3 chốt cho nhóm hãng 'unknown' (chuẩn bị ABT v1.0)
Đầu vào: 2.722 dòng `unknown` trong ABT trước gộp VIN, chia 5 nhóm (xem mục 28/11 sáng). TV3 chốt cả 5 nhóm, và giá 12345 thì **giữ, không lọc**: giá này nằm sát trung vị thị trường, lọc cứng dễ xóa oan. Chỉ lọc các ca chắc chắn giả (123456, price == odometer, model rác).

**Cách khớp vẫn giữ như cũ:** chỉ so ở **đầu** title (sau năm) hoặc **đầu** cột model. Khi phân tích, gần như toàn bộ từ khóa của các nhóm đều nằm ở hai vị trí này (ví dụ nhóm 1: 158/170 dòng, nhóm 2: 128/131 dòng). Khớp ở giữa chuỗi dễ nhầm, ví dụ 'holiday rambler' là RV chứ không phải hãng Rambler.

**Nhóm 1 · Hãng xe tải/van/bus thương mại → hãng riêng (`OTHER_COMMERCIAL_BRANDS`).** 207 dòng, giá trung vị 26.883.
- brightdrop, autocar, workhorse, blue bird/bluebird, am general, thomas built, ic bus, mci, ud trucks, grumman, vpg, e-one, seagrave, stewart-stevenson.
- kw → kenworth (thêm vào bảng xe tải của buổi sáng).
- Chữ 'military' không phải tên hãng nên giữ `unknown`. Riêng hmmwv/humvee suy ra am general (nhóm 4).

**Nhóm 2 · Lỗi chính tả → hãng chuẩn (`MISSPELLED_BRANDS`).** Chỉ đưa vào các cách viết đếm thấy thật trong dữ liệu:
- volkswagon/volkwagen/wolkswagen;
- hyundia/hyndai/hyyundai/hyunday;
- izuzu, cheverolet/chevorlet/cheverlot, nissian/nisson, mercedez/mercadez, infinti/infinite, porche, chrystler, cadilac, lincon/lincold, lexis, mazada/mazdo, suburu, studabaker, ponitac.
- Thêm viết tắt mb/mbz → mercedes-benz, chev → chevrolet.
- Cột model bỏ năm 2 hoặc 4 chữ số ở đầu ('86 chevorlet c10' → chevrolet).
- Bảng nhãn chuẩn Sheet 05 tự ghi các cách viết đã gộp này.

**Nhóm 3 · Hãng cổ hoặc đã ngừng → hãng riêng (`CLASSIC_BRANDS`).** 518 dòng, giá trung vị 11.450.
- Lớn nhất: mg 135, triumph 92, studebaker 45, austin-healey 26, daihatsu 26, amc 22.
- shelby 20 dòng, giá trung vị 69.500. Đúng điều TV3 lo: tách ra để `unknown` không lẫn xe cổ đắt tiền.
- austin-healey tách riêng khỏi austin vì là hai thương hiệu khác nhau.
- Thêm ineos (Grenadier, 9 dòng, trung vị 79.700). Đây là hãng mới, không có trong danh sách 42 hãng của Craigslist.
- Hạn chế: Triumph cũng sản xuất xe máy, có 1 tin 'street triple' bị tính vào hãng triumph.

**Nhóm 4 · Suy hãng từ tên dòng xe 1-1 (`MODEL_NAME_BRANDS`).** Nguồn ghi riêng là `manufacturer_source = 'model_name'`.
- Chỉ đưa vào tên thuộc duy nhất một hãng: corvette/camaro/silverado → chevrolet; mustang/f150/super duty → ford; camry/tacoma → toyota; wrangler → jeep...
- Giữ `unknown` cho tên dùng chung: sprinter (Mercedes/Dodge/Freightliner), 1500/2500/3500, e350 (Ford hay Mercedes), ranger (Ford/Polaris).
- Cũng giữ `unknown` cho tên trùng RV: sierra, outback, forester.
- Có test bảo đảm không tên nào thuộc 2 hãng.
- Thứ tự cứu: tên hãng ở title → tên hãng ở model → tên dòng xe ở model → tên dòng xe ở title.
- Kết quả: 509 dòng trong ABT (ford 200, chevrolet 147, toyota 37, jeep 27, mazda 26...).

**Nhóm 5 · Loại xe không phải ô tô (`formatter.filter_non_car`).** Sheet 03 ghi dòng log "Loại ngoài phạm vi định giá ô tô".
- Từ khóa gồm:
  - hãng RV (keystone, winnebago, jayco, forest river, thor, fleetwood...) và chữ rv/camper/motorhome;
  - rơ-moóc, kể cả rơ-moóc sơ-mi (wabash, great dane, dry van);
  - xe máy/ATV (yamaha, polaris, kawasaki, can am, ducati, ktm);
  - xe golf (golf cart, club car, ezgo, cushman), food truck, thuyền, máy công trình/nông nghiệp (kubota, forklift, kalmar).
- **Chỉ xét dòng chưa rõ hãng (`none`) hoặc hãng chỉ suy từ tên model (`model_name`).** Ô tô có hãng rõ mà ghi 'with camper shell' thì không bị loại. Lý do xét cả `model_name`: 'charger bass boat' (thuyền), 'cherokee by forest river' (RV) và 'avenger primtime trailer' (rơ-moóc) bị nhóm 4 suy nhầm thành dodge/jeep, nhưng vẫn bị loại đúng.
- Bỏ cụm 'trade for …' và 'with/and trailer' trước khi so khớp. Có 6 tin được giữ đúng nhờ vậy, ví dụ 'f150 … trade for motorhome', 'f650 … with available trailers'.
- Bị loại 367 dòng (356 `none` + 11 `model_name`), giá trung vị 12.900. Đã đọc toàn bộ danh sách, không thấy ô tô bị loại nhầm.
- Bước này chạy TRƯỚC khi xuất VIN.

**Kết quả lượt 1 (trước gộp VIN).**
- Số dòng: 272.179 → 271.812 dòng (−367).
- `unknown`: 2.722 → **901 dòng (−67%)**, còn 0,33% số dòng, giá trung vị 8.500.
- Số hãng: 111.
- Nguồn hãng: original 263.550 · title 5.596 · model 1.256 · model_name 509 · none 901.
- Trên file gốc: cứu từ title 6.428 dòng (sáng nay 5.597), từ model 1.578 (sáng nay 1.205), suy từ tên dòng xe 612.
- Phần `unknown` còn lại chủ yếu là:
  - tin không ghi hãng ('wheelchair handicap van', 'cargo van');
  - tin rao mua xe;
  - xe quân sự;
  - tin có title mở đầu bằng tên hãng nhưng không có năm (151 dòng, ví dụ 'mercedes benz sprinter 2004'). Luật 24/11 chỉ tin title có năm ở đầu nên chưa cứu nhóm này. Nếu TV3 muốn cứu, việc này không làm đổi tập dòng, nên không cần khóa lại manifest.

**Kiểm tra.**
- 75/75 test PASS. Test mới: `test_brand_groups_model_name_and_non_car_filter` (12 ca của 5 nhóm) và `test_model_name_table_is_one_to_one`. md5 cho thấy test không ghi vào `data/processed`.
- Sửa `test_m1_new_cleaning_rules`: dòng 'Smart key remote' vốn có model ngẫu nhiên 'camry' nên nay suy ra toyota. Đặt model rỗng để test vẫn kiểm đúng điều cần kiểm.
- Validator 36/36 PASS, READY FOR HOT-SWAP.
- File gửi TV3, đều không rỗng:
  - `vin_candidates_local.parquet`: 5,3 MB, 271.812 dòng, 162.924 dòng có VIN, khớp đủ listing_id của ABT.
  - `vehicles_exp0_thin.csv`: 6,9 MB, 295.451 dòng, id duy nhất. `read_raw_exp0_features` ghép đủ 271.812/271.812 id.
- Workbook xuất lại: Sheet 03 có dòng loại ngoài phạm vi, Sheet 05 có 182 dòng.
