# HỢP ĐỒNG DỮ LIỆU – DS111 (phiên bản 1.1 · đề xuất cho nguồn 2026)

**Trạng thái:** Giao ước ban đầu cho ba thành viên. Chỉ thay đổi bằng pull request được cả nhóm rà soát; thay đổi phá vỡ tương thích phải tăng phiên bản. File thật và file mồi phải có cùng cấu trúc cột.

## 1. Dữ liệu TV1 bàn giao

- Đường dẫn: `data/processed/used_cars_clean_v1.parquet`, định dạng Apache Parquet, mỗi dòng là một tin đăng duy nhất.
- Tên cột, chữ thường và dấu gạch dưới đúng bảng dưới đây; kiểu dữ liệu phải đúng khi đọc bằng `pandas.read_parquet()`/PyArrow.
- Tệp này lưu các trường để mô tả và phân tích, **không phải dữ liệu đã mã hóa thành biến đầu vào mô hình**. TV3 thực hiện mã hóa và khớp quy tắc điền khuyết phụ thuộc dữ liệu huấn luyện bên trong từng lần chia tập. Không tạo đặc trưng sử dụng nhãn đích ngoài `log_price`, và tuyệt đối không lấy `price`, `log_price` làm biến đầu vào để dự báo giá.

| Cột | Kiểu pandas khi đọc | Miền và diễn giải |
|---|---|---|
| `listing_id` | `int64` | Khóa duy nhất không rỗng; nếu không có ID gốc, dùng khóa ổn định và mô tả cách sinh |
| `price` | `float64` | Giá USD, 500–150.000 (ngoại lệ phải được ghi nhận riêng) |
| `log_price` | `float64` | `numpy.log(price)`, sai số tính toán cho phép 1e-8 |
| `year` | `int32` | Năm model từ 1950–2027, tối đa posting_year + 1 |
| `posting_year` | `int32` | Năm tin đăng được đăng; có chứng cứ từ `posting_date` |
| `car_age` | `int32` | `max(posting_year - year, 0)`, không âm, không quá 100 |
| `odometer` | `float64` | Số dặm không âm, không quá 500.000 (ghi rõ cách xử lý ngoại lệ) |
| `log_odometer` | `float64` | `numpy.log1p(odometer)` |
| `miles_per_year` | `float64` | `odometer / max(car_age, 1)` |
| `manufacturer_clean` | `string` | Hãng chuẩn hóa chữ thường; `unknown` nếu không xác định được |
| `model_clean` | `string` | Dòng xe chuẩn hóa, `unknown` nếu thiếu |
| `condition_clean` | `string` | `salvage`, `fair`, `good`, `excellent`, `like new`, `new`, `unknown` |
| `title_status_clean` | `string` | `clean`, `rebuilt`, `salvage`, `lien`, `missing`, `parts only`, `unknown` |
| `cylinders_num` | `float64` | Số xi-lanh hữu hạn, không âm; xe điện/thiếu số xi-lanh cần có quy ước được chứng minh thay vì gán bừa |
| `fuel` | `string` | Nhiên liệu đã chuẩn hóa; thiếu ghi `unknown` |
| `transmission` | `string` | Hộp số đã chuẩn hóa; thiếu ghi `unknown` |
| `drive` | `string` | Dẫn động đã chuẩn hóa; thiếu ghi `unknown` |
| `type` | `string` | Kiểu xe đã chuẩn hóa; thiếu ghi `unknown` |
| `state` | `string` | Mã bang đã chuẩn hóa chữ thường; thiếu ghi `unknown` |
| `region` | `string` | Vùng đã chuẩn hóa chữ thường; thiếu ghi `unknown` |
| `year_era` | `string` | `Classic`, `Old`, `Mid`, `Modern` theo các mốc trước 2000, 2000–2010, 2011–2016, từ 2017 |
| `split_set` | `string` | Chỉ nhận `train`, `val`, `test`; TV3 chịu trách nhiệm chia cố định, không để TV1 quyết định bằng thông tin nhãn đích |

**VIN và đơn vị phân tích 2026.** TV1 cung cấp metadata nội bộ sau lọc nghiệp vụ và trước bước gộp tin. Chuẩn hóa VIN bằng cách bỏ khoảng trắng hai đầu, viết hoa và kiểm tra 17 ký tự cho phép. Mã lặp một ký tự và các chuỗi mẫu đã liệt kê được coi là giá trị giữ chỗ. Nếu định dạng và chữ số kiểm tra phù hợp [49 CFR 565.15(c)](https://www.ecfr.gov/current/title-49/subtitle-B/chapter-V/part-565/subpart-B/section-565.15), giữ tin có thời điểm đăng hợp lệ mới nhất, rồi ưu tiên listing ID lớn hơn khi hòa. VIN đúng dạng nhưng không qua chữ số kiểm tra được giữ toàn bộ và gom theo chuỗi trùng khớp. VIN thiếu, sai dạng và giữ chỗ được giữ theo ID tin, với giới hạn là có thể tồn tại xe đăng lại chưa nhận diện. Việc chọn tin chỉ dùng VIN, ngày đăng và ID. Báo số dòng và trung vị giá trước/sau theo loại người bán để thấy ảnh hưởng của việc gộp tin.

**Chia tập.** TV3 dùng danh sách ID được giữ sau xử lý VIN để khóa manifest, seed 42. Dữ liệu production từ 2026 cần manifest có `vehicle_group_id`. Mỗi nhóm chỉ thuộc một tập và dùng GroupKFold trong train. Tỷ lệ 70/15/15 theo số nhóm là xấp xỉ theo số dòng. Danh sách ID và `split_set` trong ABT phải khớp manifest. Dùng cùng manifest cho mọi đối chứng. So sánh hai phiên bản trên tập đánh giá chung đã khóa. Tệp VIN và khóa nhóm chỉ lưu nội bộ.

**Cột bổ sung.** Giữ nguyên 22 cột bắt buộc. ABT 2026 bàn giao thêm `condition_missing` và `cylinders_was_missing` dạng boolean không rỗng, ghi trước khi điền. Quy ước xi-lanh ghi cờ sau chuẩn hóa giá trị quan sát và sau quy đổi xe điện về 0 có căn cứ. Production từ 2026 yêu cầu cờ xi-lanh. EDA chỉ dùng xi-lanh quan sát và có thể báo xe điện riêng. Giữ `seller_type` dạng string với nhãn `private`, `dealer`, `unknown` (ánh xạ `owner` thành `private`) để phân tích độ phủ. Khi có cờ xi-lanh, TV3 khôi phục vị trí đã điền trong ABT và học mode theo hãng/dòng, hãng, toàn train trong từng fold. Cờ tình trạng và loại người bán phục vụ EDA ở phiên bản này. Giữ mã nhóm xe trong manifest nội bộ và tránh đưa VIN thô lên Git hoặc bảng dự báo.

**Kiểm tra bàn giao:** đúng cột/kiểu, không ID trùng, số vô cực bằng 0, các trường số bắt buộc không rỗng, các trường chuỗi không rỗng sau khi chuẩn hóa, miền giá/tuổi/dặm đúng quy ước; tỷ lệ giữ dòng và nhật ký xử lý ngoại lệ phải ghi trong Workbook. TV1 chạy bộ kiểm thử ở Bước 2/4 trước khi bàn giao và ghi nhận rõ kết quả `PASS` hoặc `FAIL`. **Không được tuyên bố đã có bộ kiểm thử hoạt động ở Bước 1/4.**

## 2. Công thức và phương pháp dùng chung

Phép biến đổi lý thuyết:

\[
P=P_0 e^{-k\,car\_age}(1+odometer)^{-\beta}
\quad\Longrightarrow\quad
\ln P=\ln P_0-k\,car\_age-\beta\ln(1+odometer).
\]

Đây là **tuyến tính theo tham số** khi dùng các đặc trưng `car_age`, `log_odometer`; không có nghĩa mọi quan hệ trong dữ liệu thật đều tuyến tính. TV3 phải thử nghiệm đối chứng với giá gốc, biến đổi log và đặc trưng đa thức.

- `condition_clean`: có thứ bậc rõ về chất lượng, mã hóa `salvage=0, fair=1, good=2, excellent=3, like new=4, new=5`; `unknown` phải được xử lý riêng.
- `title_status_clean`: thứ tự thí nghiệm `parts only=0, missing=1, salvage=2, rebuilt=3, lien=4, clean=5` **chỉ là giả thuyết**. Trạng thái pháp lý không hình thành thứ tự chất lượng phổ quát. Bắt buộc so sánh với One-Hot; không kết luận Ordinal phù hợp trước khi có số liệu.
- Baseline production dùng One-Hot cho `fuel`, `transmission`, `drive`, `type`, `manufacturer_clean` và `state` trên train của từng fold. OLS dùng `drop_first=True`. Nếu còn nhiều bản ghi một xe, dùng GroupKFold và encoder không dùng giá. Đối chứng OOF Target Encoding của mock vẫn giữ cho phần mở rộng.
- Lưu `pandas.Series` biến đổi trong vùng nhớ công thức và siêu dữ liệu trong cấu trúc riêng. `ModelRegistry` quản lý mô hình, thông số, tập chia, kết quả và phiên bản dữ liệu.

## 3. Đầu ra TV3 cho TV2

- Tệp `outputs/predictions/predictions_test.parquet` chỉ chứa các dòng **test** chưa từng được dùng khớp mô hình/siêu tham số/encoder; mỗi cặp `listing_id + experiment_id + model_name` là duy nhất.

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `listing_id` | `int64` | Liên kết với `used_cars_clean_v1.parquet` |
| `split_set` | `string` | Phải bằng `test` |
| `experiment_id` | `string` | Ví dụ `Exp3B` |
| `model_name` | `string` | Ví dụ `Ridge`, `RandomForest` |
| `actual_price_usd` | `float64` | Giá thực tế, lớn hơn 0 |
| `predicted_price_usd` | `float64` | Giá dự báo dương và hữu hạn |
| `residual_usd` | `float64` | `actual_price_usd - predicted_price_usd` |
| `absolute_error_usd` | `float64` | Giá trị tuyệt đối của phần dư |
| `dataset_version` | `string` | Ví dụ `v1.0`, để chống trộn kết quả khác bộ dữ liệu |

TV3 công bố số đo `R²`, `SS_res`, `SS_tot`, trung bình `y_mean_baseline`, Adjusted `R²`, RMSE, MAE, MAPE theo cả thang log và USD khi phù hợp. Khi mô hình học log, phép nghịch đảo dùng hiệu chỉnh Duan Smearing khớp **chỉ trên phần dư train** (hoặc out-of-fold của train), không lấy hệ số hiệu chỉnh từ test. Không tính Adjusted R² nếu số mẫu không lớn hơn số biến cộng 1; ghi không xác định thay vì gán kết quả giả.

## 4. Phân nhánh và bảo vệ các vùng làm việc

- `main`: nhánh ổn định, chỉ nhận PR đã kiểm thử và đánh giá.
- `feature/m1-data-workbook`: TV1 chủ trì `src/data/`, `workbook/`, phần xuất dữ liệu.
- `feature/m2-eda-dashboard`: TV2 chủ trì `src/analysis/`, `dashboard/`, hình phân tích.
- `feature/m3-models`: TV3 chủ trì `src/models/`, `scripts/`, `outputs/` (trừ file sinh tự động không đưa vào Git).
- `configs/`, `docs/DATA_CONTRACT.md`, `src/core/`, `README.md`, `requirements.txt` là vùng dùng chung: chỉ TV3 điều phối thay đổi sau trao đổi với nhóm; thành viên khác đề xuất qua PR.
- Không sửa tệp của người khác mà chưa thống nhất; không `git push --force` lên `main`; không đưa `vehicles.csv`, dữ liệu cá nhân, trọng số nặng hoặc mật khẩu lên Git.
- Một PR cần mô tả thay đổi, đầu vào/đầu ra, kết quả chạy kiểm thử và ảnh hưởng tới hợp đồng. Sau khi gộp, các thành viên cập nhật nhánh từ `main`.

## 5. Thứ tự thực hiện

Bước 1/4: cấu hình và tài liệu; Bước 2/4: `FormulaStore`, tạo mẫu và kiểm tra bàn giao; Bước 3/4: mã hóa/mô hình/đối chứng; Bước 4/4: chạy kiểm thử, kết xuất bảng kết quả, ghép Workbook và bàn giao. Các khâu chưa được xây dựng không được ghi là đã chạy.
