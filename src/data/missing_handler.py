"""M1-04 · Bước 1 · Xử lý thiếu dữ liệu → Workbook Sheet 02.

Nguyên tắc: xử lý khuyết theo bản chất từng cột, ghi log lượng mất/giữ,
không xóa hàng loạt vô căn cứ.

Lưu ý rò rỉ (DATA_CONTRACT §1): điền khuyết phụ thuộc phân phối dữ liệu
nên do TV3 khớp trong từng fold. Ở đây TV1 chỉ dùng quy tắc xác định
(không học từ dữ liệu) hoặc loại dòng có ghi lý do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.transform_log import TransformLog

SHEET = "02_Missing"


def coerce_numeric(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """Ép id/price/year/odometer sang số; giá trị không đọc được thành NaN."""
    out = df.copy()
    for col in ("id", "price", "year", "odometer"):
        before_na = out[col].isna().sum()
        out[col] = pd.to_numeric(out[col], errors="coerce")
        bad = int(out[col].isna().sum() - before_na)
        log.record("02_missing", SHEET, f"Ép {col} sang số",
                   "CSV đọc dạng chuỗi; giá trị không phải số coi như khuyết",
                   df, out, affected=bad)
    return out


def drop_missing_required(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """Loại dòng thiếu trường không thể suy ra: id, price, year, posting_date."""
    out = df
    for col, reason in (
        ("id", "Không có khóa listing_id thì không truy vết được"),
        ("price", "Biến mục tiêu; không được điền giá"),
        ("year", "Không tính được car_age"),
        ("posting_date", "Không xác định posting_year để tính car_age"),
    ):
        before = out
        out = out[out[col].notna()]
        log.record("02_missing", SHEET, f"Loại dòng thiếu {col}", reason, before, out)
    return out


# Craigslist chỉ cho chọn 42 hãng; hãng ngoài danh sách bị để trống manufacturer
# (Sheet 01: Maserati, Bentley, Isuzu, Hummer... nằm trong 3,65% khuyết).
EXTRA_BRANDS: dict[str, str] = {
    "maserati": "maserati", "bentley": "bentley", "lamborghini": "lamborghini",
    "rolls royce": "rolls-royce", "mclaren": "mclaren", "bugatti": "bugatti",
    "lotus": "lotus", "genesis": "genesis", "polestar": "polestar",
    "rivian": "rivian", "lucid": "lucid", "scion": "scion", "isuzu": "isuzu",
    "hummer": "hummer", "oldsmobile": "oldsmobile", "plymouth": "plymouth",
    "saab": "saab", "suzuki": "suzuki", "smart": "smart",
}
# Cách viết thường gặp trong tiêu đề → tên hãng chuẩn của dữ liệu
TITLE_ALIASES: dict[str, str] = {
    "chevy": "chevrolet", "vw": "volkswagen", "mercedes": "mercedes-benz",
    "mercedes benz": "mercedes-benz", "benz": "mercedes-benz",
    "land rover": "land rover", "range rover": "land rover", "alfa romeo": "alfa-romeo",
    "aston martin": "aston-martin", "harley davidson": "harley-davidson",
}


# Hãng con/thương hiệu tách ra: không tính là mâu thuẫn khi so hãng gốc với title.
# Ram tách khỏi Dodge năm 2010 (tin cũ ghi "Dodge Ram"); Scion là thương hiệu của Toyota.
BRAND_FAMILY: dict[str, str] = {"dodge": "ram", "scion": "toyota"}


def _normalize_brand(values: pd.Series) -> pd.Series:
    """Đưa tên hãng về cùng cách viết với formatter (vd 'rover' → 'land rover')."""
    from src.data.formatter import MANUFACTURER_ALIASES, normalize_text
    return normalize_text(values, default="").replace(MANUFACTURER_ALIASES).replace(BRAND_FAMILY)


def rescue_manufacturer(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """Cứu manufacturer bị thiếu từ title ('2022 Ford Explorer ST' → 'ford').

    Bỏ năm ở đầu tiêu đề rồi khớp từ khóa hãng ở ĐẦU chuỗi (tên dài ưu tiên,
    để 'land rover' thắng 'rover'). Chỉ khớp đầu chuỗi để tránh nhầm như
    'smart key'. Quy tắc xác định, không học từ giá → không rò rỉ.

    Theo yêu cầu TV3, lưu nguồn suy diễn và kiểm tra mâu thuẫn:
    - ``manufacturer_source``: 'original' (có sẵn) / 'title' (cứu từ title) / 'none'.
    - ``manufacturer_title_conflict`` = True khi hãng gốc KHÁC hãng đọc được từ title
      (chỉ gắn cờ, không sửa: không đủ căn cứ để biết bên nào đúng).
    """
    out = df.copy()
    if "title" not in out.columns:
        out["manufacturer_source"] = out["manufacturer"].notna().map(
            {True: "original", False: "none"}).astype("string")
        out["manufacturer_title_conflict"] = False
        return out
    known = {m.lower() for m in out["manufacturer"].dropna()}
    lookup = {k.replace("-", " "): k for k in known} | EXTRA_BRANDS | TITLE_ALIASES
    keys = sorted(lookup, key=len, reverse=True)

    def find(text: str) -> object:
        return next((lookup[k] for k in keys if text == k or text.startswith(k + " ")), pd.NA)

    # Chỉ tin tiêu đề mở đầu bằng năm sản xuất ('2019 Maserati ...'): tránh khớp
    # nhầm tin không phải xe như 'Smart key remote'.
    title = out["title"].fillna("").str.lower().str.replace("-", " ")
    has_year = title.str.match(r"^\s*(19|20)\d{2}\s+")
    text = title.str.replace(r"^\s*\d{4}\s+", "", regex=True).str.strip()
    original = out["manufacturer"].notna()
    from_title = pd.Series(pd.NA, index=out.index, dtype="object")
    from_title[has_year] = text[has_year].map(find)

    missing_before = int((~original).sum())
    out.loc[~original & has_year, "manufacturer"] = from_title[~original & has_year]
    rescued_mask = ~original & out["manufacturer"].notna()
    out["manufacturer_source"] = np.select([original, rescued_mask], ["original", "title"], "none")
    out["manufacturer_source"] = out["manufacturer_source"].astype("string")

    both = original & from_title.notna()
    conflict = both & (_normalize_brand(out["manufacturer"]) != _normalize_brand(from_title.astype("string")))
    out["manufacturer_title_conflict"] = conflict.astype(bool)

    rescued = int(rescued_mask.sum())
    log.record("02_missing", SHEET, "Cứu manufacturer từ title",
               f"Tiêu đề bắt đầu bằng năm + tên hãng; cứu {rescued}/{missing_before} dòng thiếu; "
               "lưu nguồn ở manufacturer_source", df, out, affected=rescued)
    log.record("02_missing", SHEET, "Gắn cờ manufacturer_title_conflict",
               f"Hãng gốc khác hãng trong title ở {int(conflict.sum())}/{int(both.sum())} dòng "
               "đọc được cả hai; chỉ gắn cờ, không sửa", out, out, affected=int(conflict.sum()))
    return out


def flag_condition_missing(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """``condition_missing`` = True khi condition gốc rỗng (trước khi điền 'unknown').

    Tách khỏi ``condition_clean == 'unknown'`` (gồm cả nhãn lạ) để TV3 phân tích
    nguyên nhân thiếu và thử qua CV (Bài 2 · Missingness Indicator).
    """
    out = df.copy()
    raw = out["condition"].astype("string").str.strip()
    out["condition_missing"] = (raw.isna() | raw.eq("")).fillna(True).astype(bool)
    log.record("02_missing", SHEET, "Gắn cờ condition_missing",
               "Giữ thông tin thiếu gốc trước khi điền 'unknown'", df, out,
               affected=int(out["condition_missing"].sum()))
    return out


def drop_unreliable_odometer(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """Người bán tự báo đồng hồ hỏng/quay vòng → số dặm không phản ánh hao mòn."""
    out = df
    for flag in ("odometer_broken", "odometer_rolled_over"):
        if flag not in out.columns:
            continue
        before = out
        out = out[pd.to_numeric(out[flag], errors="coerce").fillna(0).eq(0)]
        log.record("02_missing", SHEET, f"Loại dòng {flag} = 1",
                   "Số dặm do người bán báo là không tin cậy", before, out)
    return out


def handle_odometer(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """Xử lý odometer khuyết.

    Mặc định: loại dòng (tỷ lệ khuyết nhỏ, tránh điền theo nhóm trước khi chia tập).
    TODO(M1-04): kế hoạch ghi "điền odometer theo nhóm" — nếu làm, thống nhất
    với TV3 vì median theo nhóm là quy tắc học từ dữ liệu (phải khớp trên train).
    """
    out = df[df["odometer"].notna()]
    log.record("02_missing", SHEET, "Loại dòng thiếu odometer",
               "Chưa thống nhất quy tắc điền theo nhóm với TV3", df, out)
    return out


def run(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    df = coerce_numeric(df, log)
    df = drop_missing_required(df, log)
    df = rescue_manufacturer(df, log)
    df = handle_odometer(df, log)
    df = drop_unreliable_odometer(df, log)
    # condition thiếu 46% → formatter điền nhãn 'unknown' (Bài 2: giá trị cố định);
    # cờ condition_missing giữ riêng phần thiếu gốc (TV3 đồng ý 24/11).
    df = flag_condition_missing(df, log)
    return df
