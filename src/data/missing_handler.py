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


def rescue_manufacturer(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """Cứu manufacturer bị thiếu từ title ('2022 Ford Explorer ST' → 'ford').

    Bỏ năm ở đầu tiêu đề rồi khớp từ khóa hãng ở ĐẦU chuỗi (tên dài ưu tiên,
    để 'land rover' thắng 'rover'). Chỉ khớp đầu chuỗi để tránh nhầm như
    'smart key'. Quy tắc xác định, không học từ giá → không rò rỉ.
    """
    if "title" not in df.columns:
        return df
    out = df.copy()
    known = {m.lower() for m in out["manufacturer"].dropna()}
    lookup = {k.replace("-", " "): k for k in known} | EXTRA_BRANDS | TITLE_ALIASES
    keys = sorted(lookup, key=len, reverse=True)

    def find(text: str) -> object:
        return next((lookup[k] for k in keys if text == k or text.startswith(k + " ")), pd.NA)

    # Chỉ tin tiêu đề mở đầu bằng năm sản xuất ('2019 Maserati ...'): tránh khớp
    # nhầm tin không phải xe như 'Smart key remote'.
    title = out["title"].fillna("").str.lower().str.replace("-", " ")
    mask = out["manufacturer"].isna() & title.str.match(r"^\s*(19|20)\d{2}\s+")
    text = title[mask].str.replace(r"^\s*\d{4}\s+", "", regex=True).str.strip()
    missing_before = int(out["manufacturer"].isna().sum())
    out.loc[mask, "manufacturer"] = text.map(find)
    rescued = missing_before - int(out["manufacturer"].isna().sum())
    log.record("02_missing", SHEET, "Cứu manufacturer từ title",
               f"Tiêu đề bắt đầu bằng năm + tên hãng; cứu {rescued}/{missing_before} dòng thiếu",
               df, out, affected=rescued)
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
    # Ghi chú: condition thiếu 46% → giữ nhãn 'unknown' (Bài 2: giá trị cố định).
    # Cột đánh dấu khuyết (Missingness Indicator) = (condition_clean == 'unknown'),
    # TV3 tự suy ra được nên không cần thêm cột vào hợp đồng.
    return df
