"""M1-04 · Bước 1 · Xử lý thiếu dữ liệu → Workbook Sheet 02.

Nguyên tắc: xử lý khuyết theo bản chất từng cột, ghi log lượng mất/giữ,
không xóa hàng loạt vô căn cứ.

Lưu ý rò rỉ (DATA_CONTRACT §1): điền khuyết phụ thuộc phân phối dữ liệu
nên do TV3 khớp trong từng fold. Ở đây TV1 chỉ dùng quy tắc xác định
(không học từ dữ liệu) hoặc loại dòng có ghi lý do.
"""

from __future__ import annotations

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


def rescue_manufacturer(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """Cứu manufacturer bị thiếu từ từ đầu tiên của model nếu đó là một hãng đã biết.

    Quy tắc xác định, không học từ giá → không rò rỉ.
    TODO(M1-04): kiểm tra thêm trường hợp hãng 2 từ ("land rover", "alfa romeo").
    """
    out = df.copy()
    known = set(out["manufacturer"].dropna().str.strip().str.lower())
    first_word = out["model"].str.strip().str.lower().str.split().str[0]
    mask = out["manufacturer"].isna() & first_word.isin(known)
    out.loc[mask, "manufacturer"] = first_word[mask]
    log.record("02_missing", SHEET, "Cứu manufacturer từ model",
               "Từ đầu của model trùng tên hãng đã biết", df, out, affected=int(mask.sum()))
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
    return df
