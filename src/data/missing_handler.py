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
    "saab": "saab", "suzuki": "suzuki", "smart": "smart", "ineos": "ineos",
}
# Hãng xe tải thương mại (TV3 28/11): trước đây rơi vào 'unknown', trộn chung với xe con.
COMMERCIAL_TRUCK_BRANDS: dict[str, str] = {
    "freightliner": "freightliner", "frht": "freightliner",
    "international": "international", "intl": "international", "navistar": "international",
    "kenworth": "kenworth", "peterbilt": "peterbilt", "hino": "hino",
    "mack": "mack", "western star": "western star", "sterling": "sterling", "kw": "kenworth",
}
# TV3 chốt 28/11 · nhóm 1: hãng xe tải/van/bus thương mại khác → hãng riêng.
OTHER_COMMERCIAL_BRANDS: dict[str, str] = {
    "brightdrop": "brightdrop", "autocar": "autocar", "workhorse": "workhorse",
    "blue bird": "blue bird", "bluebird": "blue bird", "am general": "am general",
    "thomas built": "thomas built", "thomas": "thomas built", "ic bus": "ic bus",
    "ic corporation": "ic bus", "mci": "mci", "ud trucks": "ud trucks", "ud": "ud trucks",
    "grumman": "grumman", "vpg": "vpg", "e one": "e-one", "seagrave": "seagrave",
    "stewart stevenson": "stewart-stevenson", "stewart and stevenson": "stewart-stevenson",
}
# Nhóm 3: hãng cổ/đã ngừng sản xuất → hãng riêng, để 'unknown' không lẫn xe cổ đắt (Shelby...).
CLASSIC_BRANDS: dict[str, str] = {
    "triumph": "triumph", "mg": "mg", "mgb": "mg", "mga": "mg", "mgtd": "mg",
    "austin healey": "austin-healey", "austin": "austin", "studebaker": "studebaker",
    "amc": "amc", "american motors": "amc", "rambler": "rambler", "nash": "nash",
    "hudson": "hudson", "packard": "packard", "willys": "willys", "kaiser": "kaiser",
    "jensen": "jensen", "sunbeam": "sunbeam", "shelby": "shelby", "daihatsu": "daihatsu",
    "geo": "geo", "fisker": "fisker", "edsel": "edsel", "desoto": "desoto", "de soto": "desoto",
    "delorean": "delorean", "checker": "checker", "merkur": "merkur", "opel": "opel",
    "lancia": "lancia", "citroen": "citroen", "renault": "renault", "peugeot": "peugeot",
    "morris": "morris", "excalibur": "excalibur",
}
# Nhóm 2: lỗi chính tả tên hãng (đếm được trong dòng unknown) → hãng chuẩn.
MISSPELLED_BRANDS: dict[str, str] = {
    "volkswagon": "volkswagen", "volkwagen": "volkswagen", "volswagen": "volkswagen",
    "hyundia": "hyundai", "hyndai": "hyundai", "hyyundai": "hyundai", "hundai": "hyundai",
    "hyudai": "hyundai", "izuzu": "isuzu", "cheverolet": "chevrolet", "chevorlet": "chevrolet",
    "chevolet": "chevrolet", "nissian": "nissan", "mercedez": "mercedes-benz", "mazada": "mazda",
    "suburu": "subaru", "infinti": "infiniti", "infinity": "infiniti", "porche": "porsche",
    "chrystler": "chrysler", "chryler": "chrysler", "cadilac": "cadillac",
    "studabaker": "studebaker", "ponitac": "pontiac", "hyunday": "hyundai", "cheverlot": "chevrolet",
    "wolkswagen": "volkswagen", "wolkwagen": "volkswagen", "nisson": "nissan", "nissen": "nissan",
    "nisan": "nissan", "lexis": "lexus", "mazdo": "mazda", "cadallic": "cadillac",
    "lincold": "lincoln", "lincoon": "lincoln", "lincon": "lincoln", "merceded": "mercedes-benz",
    "mercadez": "mercedes-benz", "infinite": "infiniti",
}
# Cách viết thường gặp trong tiêu đề → tên hãng chuẩn của dữ liệu
TITLE_ALIASES: dict[str, str] = {
    "chevy": "chevrolet", "chev": "chevrolet", "vw": "volkswagen", "mercedes": "mercedes-benz",
    "mercedes benz": "mercedes-benz", "benz": "mercedes-benz", "mbz": "mercedes-benz",
    "mb": "mercedes-benz",
    "land rover": "land rover", "range rover": "land rover", "alfa romeo": "alfa-romeo",
    "aston martin": "aston-martin", "harley davidson": "harley-davidson",
}
# Nhóm 4: tên dòng xe CHỈ thuộc một hãng → suy ra hãng (manufacturer_source = 'model_name').
# Cố ý bỏ tên dùng chung nhiều hãng/nhiều loại xe, giữ 'unknown': sprinter (Mercedes/Dodge/
# Freightliner), 1500/2500/3500, e350 (Ford E-350/Mercedes E350), ranger (Ford/Polaris),
# sierra/outback/forester (trùng tên RV), colorado, nova, amg (Mercedes-AMG/AM General).
_MODEL_NAMES: dict[str, tuple[str, ...]] = {
    "chevrolet": ("corvette", "camaro", "silverado", "suburban", "tahoe", "chevelle", "impala",
                  "monte carlo", "el camino", "bel air", "blazer", "s10", "c10", "k10", "astro",
                  "z28", "z 28", "iroc"),
    "ford": ("mustang", "f150", "f 150", "f250", "f 250", "f350", "f 350", "f35o", "f 35o",
             "f450", "f 450", "f550", "f 550", "f100", "f 100", "super duty", "superduty",
             "powerstroke", "bronco", "explorer", "expedition", "excursion", "econoline", "thunderbird",
             "taurus", "crown victoria", "model a", "model t", "transit"),
    "toyota": ("camry", "corolla", "tacoma", "tundra", "4runner", "prius", "rav4", "highlander",
               "sienna", "sequoia", "land cruiser", "fj cruiser", "supra", "celica"),
    "honda": ("civic", "accord", "odyssey", "crv", "cr v", "ridgeline"),
    "jeep": ("wrangler", "grand cherokee", "cherokee", "jeepster", "cj5", "cj 5", "cj7", "cj 7",
             "comanche"),
    "dodge": ("charger", "challenger", "durango", "dakota", "viper", "grand caravan", "caravan",
              "avenger", "super bee"),
    "chrysler": ("pt cruiser", "town country", "town and country"),
    "lincoln": ("town car",),
    "cadillac": ("escalade",),
    "gmc": ("yukon", "jimmy"),
    "mercury": ("grand marquis",),
    "pontiac": ("grand prix", "firebird", "trans am", "grand am"),
    "mazda": ("mazda3", "mazda5", "mazda6", "miata", "rx7", "rx 7", "rx8", "rx 8"),
    "subaru": ("wrx", "impreza", "crosstrek"),
    "volkswagen": ("jetta", "passat", "beetle", "vanagon", "tiguan", "e golf"),
    "nissan": ("altima", "sentra", "maxima", "frontier", "pathfinder", "xterra", "rogue"),
    "hyundai": ("elantra", "sonata"),
    "tesla": ("model 3", "model s", "model x", "model y"),
    "am general": ("hmmwv", "humvee", "hummvee"),
    "amc": ("pacer", "gremlin", "javelin", "amx"),
    "suzuki": ("grand vitara", "samurai"),
}
MODEL_NAME_BRANDS: dict[str, str] = {name: brand for brand, names in _MODEL_NAMES.items()
                                     for name in names}


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
    - ``manufacturer_source``: 'original' (có sẵn) / 'title' (cứu từ title) /
      'model' (cứu từ đầu cột model, khi title không có năm hoặc không khớp) /
      'model_name' (suy từ tên dòng xe chỉ thuộc một hãng, vd 'corvette' → chevrolet) / 'none'.
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
    lookup = ({k.replace("-", " "): k for k in known} | EXTRA_BRANDS | COMMERCIAL_TRUCK_BRANDS
              | OTHER_COMMERCIAL_BRANDS | CLASSIC_BRANDS | MISSPELLED_BRANDS | TITLE_ALIASES)
    keys = sorted(lookup, key=len, reverse=True)
    name_keys = sorted(MODEL_NAME_BRANDS, key=len, reverse=True)

    def starts_with(text: str, table: dict[str, str], order: list[str]) -> object:
        return next((table[k] for k in order if text == k or text.startswith(k + " ")), pd.NA)

    def find(text: str) -> object:
        return starts_with(text, lookup, keys)

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

    # Còn thiếu → thử đầu cột model ('freightliner cascadia', '*kenworth* t680', '86 chevorlet').
    # Cột model vốn mô tả xe nên không cần điều kiện năm như title; bỏ năm 2/4 chữ số ở đầu.
    from_model_mask = pd.Series(False, index=out.index)
    from_name_mask = pd.Series(False, index=out.index)
    if "model" in out.columns:
        still = out["manufacturer"].isna()
        model_text = (out["model"].fillna("").str.lower()
                      .str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
                      .str.replace(r"^(?:(?:19|20)\d{2}|\d{2})\s+", "", regex=True))
        out.loc[still, "manufacturer"] = model_text[still].map(find)
        from_model_mask = still & out["manufacturer"].notna()

        # Vẫn thiếu → suy từ tên dòng xe 1-1 ở đầu cột model, rồi đầu title có năm.
        def by_name(text: str) -> object:
            return starts_with(text, MODEL_NAME_BRANDS, name_keys)

        still = out["manufacturer"].isna()
        guess = model_text[still].map(by_name)
        title_words = text.str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
        title_guess = pd.Series(pd.NA, index=out.index, dtype="object")
        title_guess[still & has_year] = title_words[still & has_year].map(by_name)
        out.loc[still, "manufacturer"] = guess.where(guess.notna(), title_guess[still])
        from_name_mask = still & out["manufacturer"].notna()

    out["manufacturer_source"] = np.select(
        [original, rescued_mask, from_model_mask, from_name_mask],
        ["original", "title", "model", "model_name"], "none")
    out["manufacturer_source"] = out["manufacturer_source"].astype("string")

    both = original & from_title.notna()
    conflict = both & (_normalize_brand(out["manufacturer"]) != _normalize_brand(from_title.astype("string")))
    out["manufacturer_title_conflict"] = conflict.astype(bool)

    rescued, by_model = int(rescued_mask.sum()), int(from_model_mask.sum())
    by_name = int(from_name_mask.sum())
    log.record("02_missing", SHEET, "Cứu manufacturer từ title",
               f"Tiêu đề bắt đầu bằng năm + tên hãng (gồm hãng thương mại, hãng cổ, lỗi chính tả); "
               f"cứu {rescued}/{missing_before} dòng thiếu; lưu nguồn ở manufacturer_source",
               df, out, affected=rescued)
    log.record("02_missing", SHEET, "Cứu manufacturer từ model",
               f"Đầu cột model là tên hãng (gồm hãng xe tải thương mại); cứu thêm {by_model} dòng",
               out, out, affected=by_model)
    log.record("02_missing", SHEET, "Suy manufacturer từ tên dòng xe",
               f"Tên dòng xe chỉ thuộc một hãng (corvette → chevrolet, tacoma → toyota); "
               f"suy thêm {by_name} dòng, nguồn 'model_name'; tên dùng chung (sprinter, 1500) giữ unknown",
               out, out, affected=by_name)
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
