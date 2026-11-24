"""M1-05 · Bước 2 · Chuẩn hóa → Workbook Sheet 03.

Chuẩn kiểu ngày/số, lọc giá ảo 2 tầng, gộp tên hãng, chuẩn cylinders,
đưa nhãn phân loại về đúng miền trong DATA_CONTRACT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from configs.project_config import (
    CONDITION_ORDINAL,
    CONFIG,
    TITLE_STATUS_ORDINAL_EXPERIMENTAL,
)
from src.data.transform_log import TransformLog

SHEET = "03_Format"

# Sheet 01: dữ liệu thật dùng 'alfa-romeo', 'aston-martin' (có gạch) → giữ nguyên.
# 'rover' (2.581 tin) và 'land rover' (14 tin) là cùng một hãng → gộp.
MANUFACTURER_ALIASES: dict[str, str] = {
    "chevy": "chevrolet",
    "chev": "chevrolet",
    "vw": "volkswagen",
    "volkswagon": "volkswagen",
    "mercedes": "mercedes-benz",
    "mercedes benz": "mercedes-benz",
    "benz": "mercedes-benz",
    "rover": "land rover",
    "landrover": "land rover",
    "alfa romeo": "alfa-romeo",
    "aston martin": "aston-martin",
    "harley davidson": "harley-davidson",
}
# Cột chuỗi giữ tên gốc trong hợp đồng (chỉ chuẩn hóa chữ thường + unknown).
PLAIN_TEXT_COLUMNS: tuple[str, ...] = ("fuel", "transmission", "drive", "type", "state", "region")


def normalize_text(values: pd.Series, default: str = "unknown") -> pd.Series:
    """Chữ thường, bỏ khoảng trắng thừa, rỗng → ``default``."""
    s = values.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return s.replace("", pd.NA).fillna(default).astype("string")


def to_allowed_labels(values: pd.Series, allowed: set[str]) -> pd.Series:
    s = normalize_text(values)
    return s.where(s.isin(allowed), "unknown").astype("string")


def parse_cylinders(raw: pd.Series) -> pd.Series:
    """'6 cylinders' → 6.0; 'other'/khuyết → NaN."""
    return pd.to_numeric(raw.astype("string").str.extract(r"(\d+)")[0], errors="coerce").astype("float64")


def standardize_labels(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    out = df.copy()
    manu = normalize_text(out["manufacturer"])
    out["manufacturer_clean"] = manu.replace(MANUFACTURER_ALIASES).astype("string")
    out["model_clean"] = normalize_text(out["model"])
    out["condition_clean"] = to_allowed_labels(out["condition"], set(CONDITION_ORDINAL))
    out["title_status_clean"] = to_allowed_labels(
        out["title_status"], set(TITLE_STATUS_ORDINAL_EXPERIMENTAL)
    )
    for col in PLAIN_TEXT_COLUMNS:
        out[col] = normalize_text(out[col])
    merged = int((manu != out["manufacturer_clean"]).sum())
    log.record("03_format", SHEET, "Chuẩn hóa nhãn chuỗi + gộp alias hãng",
               "Tránh trùng hãng do cách viết; nhãn ngoài miền hợp đồng → unknown",
               df, out, affected=merged)
    return out


def parse_dates(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    out = df.copy()
    posted = pd.to_datetime(out["posting_date"], errors="coerce", utc=True)
    out["posting_year"] = posted.dt.year
    before = out
    out = out[out["posting_year"].notna()].copy()
    out["posting_year"] = out["posting_year"].astype("int32")
    log.record("03_format", SHEET, "Tách posting_year từ posting_date",
               "Cần năm đăng tin để tính car_age; ngày không đọc được thì loại", before, out)
    return out


def filter_ranges(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """Tầng 1: miền cứng theo DATA_CONTRACT (giá, năm, số dặm)."""
    out = df
    rules = (
        ("price", CONFIG.min_price_usd, CONFIG.max_price_usd, "Giá ảo/giá placeholder ngoài 500–150.000 USD"),
        ("year", CONFIG.min_year, CONFIG.max_year, "Năm sản xuất ngoài 1950–2022"),
        ("odometer", 0, CONFIG.max_odometer_miles, "Số dặm âm hoặc > 500.000"),
    )
    for col, lo, hi, reason in rules:
        before = out
        out = out[out[col].between(lo, hi)]
        log.record("03_format", SHEET, f"Lọc miền {col} [{lo}, {hi}]", reason, before, out)
    return out


def filter_price_by_group(df: pd.DataFrame, log: TransformLog, enabled: bool = False,
                          k: float = 1.5, min_group: int = 30) -> pd.DataFrame:
    """Tầng 2: lọc giá ngoại lệ theo quy tắc IQR trong từng hãng (Bài 4 · Boxplot).

    Ngoại lệ: price ngoài [Q1 - k·IQR, Q3 + k·IQR] của CÙNG hãng. Lọc trên
    log(price) vì giá lệch phải mạnh; IQR trên thang gốc sẽ cắt oan xe đắt.
    Hãng có < ``min_group`` tin thì không lọc (IQR không ổn định).
    TODO(M1-05): so sánh có/không bật ở Sheet 03 (rủi ro R1) rồi chốt với nhóm.
    """
    if not enabled:
        log.record("03_format", SHEET, "Lọc giá tầng 2 IQR theo hãng (chưa bật)",
                   "Chờ chốt; bật bằng --price-group-filter", df, df)
        return df
    logp = np.log(df["price"].astype("float64"))
    g = logp.groupby(df["manufacturer_clean"])
    q1, q3, n = g.transform("quantile", 0.25), g.transform("quantile", 0.75), g.transform("size")
    iqr = q3 - q1
    # 'unknown' trộn nhiều hãng → "khác thường so với cùng hãng" không có nghĩa, không lọc.
    skip = (n < min_group) | df["manufacturer_clean"].eq("unknown")
    inside = logp.between(q1 - k * iqr, q3 + k * iqr) | skip
    out = df[inside]
    log.record("03_format", SHEET, f"Lọc giá tầng 2: IQR (k={k}) của log(price) theo hãng",
               "Giá bất thường so với xe cùng hãng (quy tắc boxplot)", df, out)
    return out


def keep_us_only(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """535 tin ở Canada: giá có thể là CAD → loại để giữ một đơn vị USD."""
    if "address_country" not in df.columns:
        return df
    out = df[df["address_country"].fillna("US").str.upper().ne("CA")]
    log.record("03_format", SHEET, "Loại tin ngoài Mỹ (address_country = CA)",
               "Đơn vị giá có thể là CAD, không đồng nhất USD", df, out)
    return out


def dedup_by_vin(df: pd.DataFrame, log: TransformLog, enabled: bool = False) -> pd.DataFrame:
    """Tin đăng lại cùng một xe (cùng VIN 17 ký tự) → giữ tin đăng mới nhất.

    Bài 2/3: loại bỏ trùng lặp. Lý do thêm: cùng một xe nằm ở cả train và test
    gây rò rỉ. Dòng không có VIN hợp lệ không bị đụng tới (không đủ căn cứ).
    TODO: chờ TV3 đồng ý rồi bật bằng --dedup-vin.
    """
    if not enabled or "VIN" not in df.columns:
        log.record("03_format", SHEET, "Bỏ trùng VIN (chưa bật)", "Chờ TV3 xác nhận", df, df)
        return df
    vin = df["VIN"].astype("string").str.strip().str.upper()
    posted = pd.to_datetime(df["posting_date"], errors="coerce", utc=True)
    valid = vin.str.fullmatch(r"[A-HJ-NPR-Z0-9]{17}").fillna(False)
    # Sắp xếp ổn định: ngày đăng mới nhất trước, hòa thì id lớn hơn trước → chạy lại ra cùng kết quả.
    order = pd.DataFrame({"posted": posted, "id": df["id"]})[valid]
    newest_first = order.sort_values(["posted", "id"], ascending=False, kind="mergesort").index
    dup = vin.loc[newest_first].duplicated(keep="first")
    out = df.drop(index=dup[dup].index)
    log.record("03_format", SHEET, "Bỏ tin trùng VIN, giữ tin đăng mới nhất",
               "Cùng một xe đăng nhiều lần; tránh lặp mẫu giữa train/test", df, out)
    return out


def standardize_cylinders(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """Chuẩn cylinders_num; hợp đồng yêu cầu số hữu hạn, không rỗng.

    - Xe điện → 0 (quy ước có căn cứ: động cơ điện không có xi-lanh).
    - Còn khuyết → median theo (hãng, dòng xe) → theo hãng → toàn bộ.
    TODO(M1-05): quy ước điền theo nhóm cần TV3 xác nhận (học từ dữ liệu,
    trước khi chia tập). Ghi rõ lý do ở Sheet 03.
    """
    out = df.copy()
    cyl = parse_cylinders(out["cylinders"])
    cyl = cyl.mask(cyl.isna() & out["fuel"].eq("electric"), 0.0)
    missing = int(cyl.isna().sum())
    for keys in (["manufacturer_clean", "model_clean"], ["manufacturer_clean"]):
        cyl = cyl.fillna(cyl.groupby([out[k] for k in keys]).transform("median"))
    cyl = cyl.fillna(cyl.median())
    out["cylinders_num"] = cyl.astype("float64")
    log.record("03_format", SHEET, "Chuẩn cylinders_num",
               "Tách số từ chuỗi; điện = 0; khuyết điền median theo nhóm", df, out, affected=missing)
    return out


def drop_duplicate_ids(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    out = df.drop_duplicates(subset="id", keep="first").copy()
    out["listing_id"] = out["id"].astype("int64")
    log.record("03_format", SHEET, "Loại id trùng, tạo listing_id",
               "listing_id phải duy nhất theo hợp đồng", df, out)
    return out


def run(df: pd.DataFrame, log: TransformLog, price_group_filter: bool = False,
        dedup_vin: bool = False) -> pd.DataFrame:
    df = drop_duplicate_ids(df, log)
    df = keep_us_only(df, log)
    df = standardize_labels(df, log)
    df = parse_dates(df, log)
    df = filter_ranges(df, log)
    # Bỏ trùng SAU khi lọc miền: nếu tin mới nhất có giá 0 ("gọi để biết giá")
    # thì vẫn giữ được tin cũ hợp lệ của cùng xe, thay vì mất cả chiếc xe.
    df = dedup_by_vin(df, log, enabled=dedup_vin)
    df = filter_price_by_group(df, log, enabled=price_group_filter)
    df = standardize_cylinders(df, log)
    return df.reset_index(drop=True)
