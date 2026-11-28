"""M1-05 · Bước 2 · Chuẩn hóa → Workbook Sheet 03.

Chuẩn kiểu ngày/số, lọc giá ảo (tầng 1) + gắn cờ IQR (tầng 2), gộp tên hãng, chuẩn cylinders,
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
    # Hợp đồng 1.1: seller_type ∈ {private, dealer, unknown}; 'owner' → 'private'.
    seller = normalize_text(out["seller_type"]) if "seller_type" in out else pd.Series(
        "unknown", index=out.index, dtype="string")
    seller = seller.replace({"owner": "private"})
    out["seller_type"] = seller.where(seller.isin({"private", "dealer"}), "unknown").astype("string")
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
        ("year", CONFIG.min_year, CONFIG.max_year, f"Năm model ngoài {CONFIG.min_year}–{CONFIG.max_year}"),
        ("odometer", 0, CONFIG.max_odometer_miles, "Số dặm âm hoặc > 500.000"),
    )
    for col, lo, hi, reason in rules:
        before = out
        out = out[out[col].between(lo, hi)]
        log.record("03_format", SHEET, f"Lọc miền {col} [{lo}, {hi}]", reason, before, out)
    # Hợp đồng 1.1: năm model tối đa posting_year + 1 (xe đời sau bán sớm 1 năm là bình thường).
    before = out
    out = out[out["year"] <= out["posting_year"] + 1]
    log.record("03_format", SHEET, "Lọc year > posting_year + 1",
               "Năm model không thể vượt năm đăng tin quá 1 năm", before, out)
    return out


# Giá giữ chỗ dạng chuỗi số liên tiếp (TV3 28/11: nằm ở đuôi 100k–150k, đẩy MSE lên).
PLACEHOLDER_PRICES: frozenset[float] = frozenset({123456.0})
# model là quảng cáo chung của bãi xe ('all makes and models', 'cars for sale'...),
# không mô tả một chiếc xe cụ thể → không định giá được.
SPAM_MODEL_PATTERN = (r"\ball makes?\b|\bany makes?\b|\ball types\b|"
                      r"\bcars for sale\b|\bpublic auto auction\b")


def filter_fake_listings(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """Tầng 1b: loại tin giá ảo/tin rác mà lọc miền không bắt được (yêu cầu TV3 28/11).

    Chạy TRƯỚC khi xuất danh sách VIN, nên đổi luật ở đây thì phải gửi lại
    vin_candidates cho TV3 khóa manifest mới.
    """
    rules = (
        ("Lọc giá giữ chỗ 123456", "Giá là dãy số liên tiếp, không phải giá bán thật",
         lambda d: ~d["price"].isin(PLACEHOLDER_PRICES)),
        ("Lọc price == odometer", "Hai trường trùng khớp tuyệt đối → một trong hai bị nhập nhầm",
         lambda d: d["price"].ne(d["odometer"])),
        ("Lọc tin quảng cáo chung (model rác)",
         "model kiểu 'all makes and models', 'cars for sale' không mô tả một xe cụ thể",
         lambda d: ~d["model_clean"].str.contains(SPAM_MODEL_PATTERN, regex=True).fillna(False)),
    )
    out = df
    for step, reason, keep in rules:
        before = out
        out = out[keep(out).to_numpy(dtype=bool)]
        log.record("03_format", SHEET, step, reason, before, out)
    return out


# TV3 chốt 28/11 · nhóm 5: RV, rơ-moóc, xe máy/ATV, xe golf, food truck, thuyền, máy công trình.
NON_CAR_PATTERN = (
    r"\b(?:keystone|winnebago|jayco|forest river|coachmen|coachman|airstream|thor|tiffin|newmar|"
    r"heartland|grand design|holiday rambler|dutchmen|gulf stream|four winds|fleetwood|"
    r"rvs?|campers?|motorhomes?|motor home|mobile home|travel trailer|fifth wheel|5th wheel|"
    r"toy hauler|trailers?|trailor|car hauler|great dane|wabash|dry van|"
    r"yamaha|polaris|kawasaki|can am|canam|ducati|ktm|ninja|atvs?|utvs?|side by side|dirt bike|"
    r"motorcycles?|scooter|moped|dune buggy|sand rail|go kart|mini bike|"
    r"golf carts?|club car|ez go|ezgo|cushman|food truck|food trailer|concession|boats?|pontoon|"
    r"jet skis?|forklift|skid steer|excavator|john deere|kubota|massey ferguson|kalmar)\b"
)
# Chỉ xét dòng chưa xác định được hãng từ tên hãng: 'none' hoặc hãng chỉ suy từ tên model
# ('charger bass boat' → dodge). Ô tô có hãng rõ mà nhắc 'trailer hitch' không bị đụng tới.
NON_CAR_SCOPE: frozenset[str] = frozenset({"none", "model_name"})


# 'f150 ... trade for motorhome', 'f650 with available trailers': từ khóa sau 'trade for' (thứ
# người bán muốn đổi lấy) hoặc 'with/and trailer' (đồ bán kèm) không mô tả chiếc xe đang rao
# → bỏ các cụm này trước khi so khớp.
TRADE_PHRASE = (r"\btrades?\s+(?:for|4)\s+(?:an?\s+)?\w+(?:\s+\w+)?|"
                r"\b(?:with|and|plus|w)\s+(?:an?\s+|available\s+)?trailers?\b")


def _words(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True)


def filter_non_car(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """Loại tin không phải ô tô (yêu cầu TV3 28/11), chạy trước khi xuất danh sách VIN."""
    text = _words(df["model"]) if "model" in df else pd.Series("", index=df.index, dtype="string")
    if "title" in df:
        text = _words(df["title"]) + " | " + text
    text = text.str.replace(TRADE_PHRASE, " ", regex=True)
    source = (df["manufacturer_source"] if "manufacturer_source" in df
              else df["manufacturer_clean"].eq("unknown").map({True: "none", False: "original"}))
    non_car = source.isin(NON_CAR_SCOPE) & text.str.contains(NON_CAR_PATTERN, regex=True)
    out = df[~non_car.to_numpy(dtype=bool)]
    log.record("03_format", SHEET, "Loại ngoài phạm vi định giá ô tô",
               "Dòng chưa rõ hãng có từ khóa RV/rơ-moóc/xe máy/ATV/xe golf/food truck/thuyền/máy "
               "công trình ở title hoặc model", df, out)
    return out


def flag_price_outliers(df: pd.DataFrame, log: TransformLog, k: float = 1.5,
                        min_group: int = 30) -> pd.DataFrame:
    """Tầng 2: GẮN CỜ giá ngoại lệ theo IQR trong từng hãng (Bài 4 · Boxplot).

    TV3 chốt (24/11): chỉ gắn cờ ``price_outlier_iqr`` rồi kiểm tra, KHÔNG lọc,
    vì xe sang/xe hiếm có thể vượt ngưỡng mà giá vẫn hợp lệ.
    - Ngưỡng [Q1 - k·IQR, Q3 + k·IQR] trên log(price) (giá lệch phải mạnh).
    - Học ngưỡng CHỈ trên dòng train rồi áp cho cả val/test → không rò rỉ.
    - Fallback: hãng có < ``min_group`` dòng train hoặc 'unknown' dùng ngưỡng
      chung của toàn bộ train.
    Cần cột split_set, nên gọi sau khi chia tập. Không xóa dòng.
    """
    out = df.copy()
    logp = np.log(out["price"].astype("float64"))
    is_train = out["split_set"].eq("train")
    train = pd.DataFrame({"brand": out.loc[is_train, "manufacturer_clean"], "logp": logp[is_train]})
    stats = train.groupby("brand")["logp"].agg(q1=lambda x: x.quantile(0.25),
                                              q3=lambda x: x.quantile(0.75), n="size")
    stats = stats[(stats["n"] >= min_group) & (stats.index != "unknown")]
    g_q1, g_q3 = train["logp"].quantile(0.25), train["logp"].quantile(0.75)
    brand = out["manufacturer_clean"]
    q1 = brand.map(stats["q1"]).astype("float64").fillna(g_q1)
    q3 = brand.map(stats["q3"]).astype("float64").fillna(g_q3)
    iqr = q3 - q1
    flag = ~logp.between(q1 - k * iqr, q3 + k * iqr)
    out["price_outlier_iqr"] = flag.astype(bool)
    fallback = int((~brand.isin(stats.index)).sum())
    log.record("03_format", SHEET, f"Gắn cờ price_outlier_iqr: IQR (k={k}) log(price) theo hãng",
               f"Ngưỡng học trên train; {len(stats)} hãng đủ ≥{min_group} mẫu, "
               f"{fallback} dòng dùng ngưỡng chung; chỉ gắn cờ, không lọc",
               df, out, affected=int(flag.sum()))
    return out


def keep_us_only(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """535 tin ở Canada: giá có thể là CAD → loại để giữ một đơn vị USD."""
    if "address_country" not in df.columns:
        return df
    out = df[df["address_country"].fillna("US").str.upper().ne("CA")]
    log.record("03_format", SHEET, "Loại tin ngoài Mỹ (address_country = CA)",
               "Đơn vị giá có thể là CAD, không đồng nhất USD", df, out)
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
    # Hợp đồng 1.1: cờ ghi SAU khi tách số và quy đổi xe điện = 0, TRƯỚC khi điền.
    # TV3 dùng cờ này để điền lại trong từng fold (học trên train).
    out["cylinders_was_missing"] = cyl.isna().astype(bool)
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


def run(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    df = drop_duplicate_ids(df, log)
    df = keep_us_only(df, log)
    df = standardize_labels(df, log)
    df = parse_dates(df, log)
    df = filter_ranges(df, log)
    df = filter_fake_listings(df, log)
    df = filter_non_car(df, log)
    # Gộp tin đăng lại theo VIN: KHÔNG làm ở đây. Theo DATA_CONTRACT 1.1, TV3 quyết định
    # bằng src.core.vin_reposts (chữ số kiểm tra 49 CFR 565.15) từ danh sách VIN mà
    # pipeline xuất SAU bước này; pipeline lọc lại theo manifest (xem pipeline_m1).
    # Lọc giá tầng 2 (IQR) chỉ gắn cờ, chạy trong pipeline sau khi có split_set.
    df = standardize_cylinders(df, log)
    return df.reset_index(drop=True)
