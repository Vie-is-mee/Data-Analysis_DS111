# Power BI · Mô hình dữ liệu và DAX measures (TV2)

> Bản văn bản của mọi measure trong `UsedCars_Analytics.pbix`. File `.pbix` là nhị phân và bị Git bỏ qua,
> nên nhóm review DAX qua file này. Dữ liệu nạp từ `outputs/reports/m2/bi/`, do
> `python -X utf8 -m src.analysis.run_m2 --bi` tạo (hoặc thư mục `m2_dev/bi/` trong output của kernel Kaggle `01_DS111`).

## 1. Nạp dữ liệu (Power Query)

| Bảng | File | Khóa | Ghi chú |
|---|---|---|---|
| `Fact_Listings` | `Fact_Listings.parquet` | `listing_id` | Mỗi dòng một tin; không có VIN, URL, mô tả, `model_clean` |
| `Dim_Manufacturer` | `Dim_Manufacturer.parquet` | `manufacturer_clean` | `origin`, `segment`, `analysis_cluster` từ `src/analysis/brand_lookup.csv` |
| `Dim_State` | `Dim_State.parquet` | `state` | `state_name` |
| `Dim_Condition` | `Dim_Condition.parquet` | `condition_clean` | `sort_order`, `is_unknown` |
| `Dim_YearEra` | `Dim_YearEra.parquet` | `year_era` | `sort_order`, `year_from`, `year_to` |

- Tạo tham số `BiFolder` (đường dẫn thư mục `bi`), mỗi bảng đọc bằng
  `Parquet.Document(File.Contents(BiFolder & "\Fact_Listings.parquet"))`. Đổi từ bản dev sang V1.0 chỉ cần sửa tham số.
  Nếu trình kết nối Parquet báo lỗi kiểu dữ liệu, dùng các file `.csv` cùng tên (UTF-8 có BOM).
- Tắt *Auto date/time* (File → Options → Current File → Data Load): dữ liệu không có cột ngày.
- Đặt kiểu số thập phân cho `price`, `log_price`, `odometer`, `log_odometer`, `miles_per_year`; số nguyên cho `car_age`, `year`.

## 2. Quan hệ

| Từ (1) | Đến (*) | Hướng lọc |
|---|---|---|
| `Dim_Manufacturer[manufacturer_clean]` | `Fact_Listings[manufacturer_clean]` | Single |
| `Dim_State[state]` | `Fact_Listings[state]` | Single |
| `Dim_Condition[condition_clean]` | `Fact_Listings[condition_clean]` | Single |
| `Dim_YearEra[year_era]` | `Fact_Listings[year_era]` | Single |

- Không có quan hệ nhiều–nhiều hay hai chiều.
- `Dim_Condition[condition_clean]` → *Sort by column* `Dim_Condition[sort_order]`; `Dim_YearEra[year_era]` → `Dim_YearEra[sort_order]`.
- Slicer dùng cột của bảng Dim (ví dụ `Dim_Manufacturer[analysis_cluster]`), không dùng cột trùng tên trong Fact.

## 3. Measures

Pearson và độ dốc dùng cách tính **hai lượt** (tính trung bình trước, rồi cộng tích độ lệch) để ổn định số học,
giống hệt `src/analysis/bi_export.py::_two_pass` nên đối chiếu được với Python. Hệ số bị ẩn (BLANK) khi n < 100.

```DAX
Số tin = COUNTROWS ( Fact_Listings )

Giá trung vị = MEDIAN ( Fact_Listings[price] )

Tuổi xe trung vị = MEDIAN ( Fact_Listings[car_age] )

Pearson tuổi – log giá =
VAR n = COUNTROWS ( Fact_Listings )
VAR mx = AVERAGE ( Fact_Listings[car_age] )
VAR my = AVERAGE ( Fact_Listings[log_price] )
VAR sxy = SUMX ( Fact_Listings, ( Fact_Listings[car_age] - mx ) * ( Fact_Listings[log_price] - my ) )
VAR sxx = SUMX ( Fact_Listings, ( Fact_Listings[car_age] - mx ) ^ 2 )
VAR syy = SUMX ( Fact_Listings, ( Fact_Listings[log_price] - my ) ^ 2 )
RETURN IF ( n >= 100, DIVIDE ( sxy, SQRT ( sxx * syy ) ) )

Pearson số dặm – log giá =
VAR n = COUNTROWS ( Fact_Listings )
VAR mx = AVERAGE ( Fact_Listings[odometer] )
VAR my = AVERAGE ( Fact_Listings[log_price] )
VAR sxy = SUMX ( Fact_Listings, ( Fact_Listings[odometer] - mx ) * ( Fact_Listings[log_price] - my ) )
VAR sxx = SUMX ( Fact_Listings, ( Fact_Listings[odometer] - mx ) ^ 2 )
VAR syy = SUMX ( Fact_Listings, ( Fact_Listings[log_price] - my ) ^ 2 )
RETURN IF ( n >= 100, DIVIDE ( sxy, SQRT ( sxx * syy ) ) )

Độ dốc log theo tuổi =
VAR n = COUNTROWS ( Fact_Listings )
VAR mx = AVERAGE ( Fact_Listings[car_age] )
VAR my = AVERAGE ( Fact_Listings[log_price] )
VAR sxy = SUMX ( Fact_Listings, ( Fact_Listings[car_age] - mx ) * ( Fact_Listings[log_price] - my ) )
VAR sxx = SUMX ( Fact_Listings, ( Fact_Listings[car_age] - mx ) ^ 2 )
RETURN IF ( n >= 100, DIVIDE ( sxy, sxx ) )

% giá thay đổi mỗi năm tuổi =
VAR b = [Độ dốc log theo tuổi]
RETURN IF ( NOT ISBLANK ( b ), EXP ( b ) - 1 )

n hiển thị = "n = " & FORMAT ( [Số tin], "#,0" )
```

- Định dạng `% giá thay đổi mỗi năm tuổi` là Percentage, 1 chữ số thập phân.
- Các measure trên là **liên hệ biên trong bộ lọc hiện tại**, chưa kiểm soát số dặm hay loại xe. Độ dốc có kiểm soát
  (OLS, sai số HC3) lấy từ `group_depreciation.csv` và hiển thị dạng bảng tĩnh, ghi rõ nguồn.
- Đường cong tuổi–giá hình chữ U, nên trang khấu hao có slicer `car_age` (mặc định 1–20) trước khi đọc độ dốc.
- Không viết Spearman bằng DAX: cần `RANKX` trong từng ngữ cảnh lọc, chậm và khó kiểm chứng. Spearman lấy từ bảng Python.

## 4. Đối chiếu DAX với Python (tiêu chí Gate 2)

1. Mở `bi/dax_parity_check.csv`: mỗi dòng là một tổ hợp lọc (toàn bộ; toyota; bmw; suv; Modern; tuổi 1–20), có sẵn cột `*_python`.
2. Trong Power BI, tạo một Table visual gồm `Số tin`, `Giá trung vị`, hai measure Pearson và `% giá thay đổi mỗi năm tuổi`.
3. Áp đúng bộ lọc của từng dòng bằng slicer, chép giá trị vào các cột `*_dax`, lưu file vào `dashboard/dax_parity_check.csv`.
4. Đạt khi: `n` và trung vị khớp tuyệt đối; Pearson và %/năm lệch < 0,005. Cùng dữ liệu và cùng công thức thì lệch phải gần 0.
