"""M2-02 / M2-08 · Dữ liệu cho Power BI và giá trị đối chiếu DAX.

Star Schema gọn: một bảng Fact (mỗi dòng một tin) và bốn Dim nhỏ dùng khóa tự
nhiên (hãng, bang, tình trạng, đời xe). Lý do chính là lọc rõ ràng một chiều và
thuộc tính mô tả nằm ở Dim, không phải hiệu năng (270.000 dòng là nhỏ với Power BI).
Không xuất ``model_clean`` (khoảng 27.000 nhãn) và mọi cột có thể chứa dữ liệu cá nhân.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

SENSITIVE_COLUMNS: frozenset[str] = frozenset({
    "VIN", "vin", "url", "region_url", "image_url", "image_urls", "description", "title", "lat", "long",
    "address_street", "address_city", "address_postal_code", "posting_date", "vehicle_group_id",
})
FACT_COLUMNS: tuple[str, ...] = (
    "listing_id", "manufacturer_clean", "state", "condition_clean", "year_era", "type", "fuel",
    "transmission", "drive", "seller_type", "title_status_clean", "condition_missing",
    "cylinders_was_missing", "split_set", "price", "log_price", "year", "car_age", "odometer",
    "log_odometer", "miles_per_year", "cylinders_num",
)
CONDITION_SORT: dict[str, int] = {"salvage": 0, "fair": 1, "good": 2, "excellent": 3, "like new": 4,
                                  "new": 5, "unknown": 6}
ERA_RANGES: dict[str, tuple[int | None, int | None, int]] = {
    "Classic": (None, 1999, 0), "Old": (2000, 2010, 1), "Mid": (2011, 2016, 2), "Modern": (2017, None, 3),
}
US_STATES: dict[str, str] = dict(zip(
    "al ak az ar ca co ct dc de fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny "
    "nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy".split(),
    ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
     "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana",
     "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
     "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
     "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
     "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
     "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"],
    strict=True,
))


def build_star_schema(df: pd.DataFrame, lookup: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fact_Listings + Dim_Manufacturer + Dim_State + Dim_Condition + Dim_YearEra."""
    fact = df.loc[:, [c for c in FACT_COLUMNS if c in df.columns]].copy()
    brands = pd.DataFrame({"manufacturer_clean": pd.Series(
        sorted(fact["manufacturer_clean"].astype(str).unique()), dtype="string")})
    dim_brand = brands.merge(lookup.astype("string"), on="manufacturer_clean", how="left").fillna(
        {"origin": "other", "segment": "unknown", "analysis_cluster": "other", "is_focus_brand": "false"})
    states = sorted(fact["state"].astype(str).unique())
    dim_state = pd.DataFrame({"state": states, "state_name": [US_STATES.get(s, "Unknown") for s in states],
                              "country": "United States"})
    conditions = sorted(set(fact["condition_clean"].astype(str)) | set(CONDITION_SORT),
                        key=lambda c: (CONDITION_SORT.get(c, 99), c))
    dim_condition = pd.DataFrame({"condition_clean": conditions,
                                  "sort_order": [CONDITION_SORT.get(c, 99) for c in conditions],
                                  "is_unknown": [c == "unknown" for c in conditions]})
    dim_era = pd.DataFrame([{"year_era": era, "year_from": low, "year_to": high, "sort_order": order}
                            for era, (low, high, order) in ERA_RANGES.items()])
    dim_era[["year_from", "year_to"]] = dim_era[["year_from", "year_to"]].astype("Int64")
    return {"Fact_Listings": fact, "Dim_Manufacturer": dim_brand, "Dim_State": dim_state,
            "Dim_Condition": dim_condition, "Dim_YearEra": dim_era}


DIM_KEYS: dict[str, str] = {"Dim_Manufacturer": "manufacturer_clean", "Dim_State": "state",
                            "Dim_Condition": "condition_clean", "Dim_YearEra": "year_era"}


def validate_star_schema(tables: Mapping[str, pd.DataFrame]) -> list[str]:
    """Danh sách vấn đề (rỗng = đạt): khóa Dim duy nhất, khóa Fact có trong Dim, không có cột nhạy cảm."""
    problems: list[str] = []
    fact = tables["Fact_Listings"]
    for name, table in tables.items():
        leaked = sorted(SENSITIVE_COLUMNS & set(table.columns))
        if leaked:
            problems.append(f"{name} chứa cột nhạy cảm: {leaked}")
    if not fact["listing_id"].is_unique:
        problems.append("Fact_Listings: listing_id trùng")
    for dim, key in DIM_KEYS.items():
        keys = tables[dim][key].astype(str)
        if not keys.is_unique:
            problems.append(f"{dim}: khóa {key} trùng")
        orphan = set(fact[key].astype(str)) - set(keys)
        if orphan:
            problems.append(f"Fact_Listings có {len(orphan)} giá trị {key} không có trong {dim}")
    return problems


DEFAULT_PARITY_FILTERS: tuple[tuple[str, dict[str, object]], ...] = (
    ("Toàn bộ", {}),
    ("Hãng = toyota", {"manufacturer_clean": "toyota"}),
    ("Hãng = bmw", {"manufacturer_clean": "bmw"}),
    ("Loại xe = suv", {"type": "suv"}),
    ("Đời xe = Modern", {"year_era": "Modern"}),
    ("Tuổi xe 1–20", {"car_age": (1, 20)}),
)
MIN_DAX_N = 100


def _two_pass(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson và độ dốc tính đúng như measure DAX hai lượt (trung bình trước, tổng độ lệch sau)."""
    dx, dy = x - x.mean(), y - y.mean()
    sxy, sxx, syy = float(dx @ dy), float(dx @ dx), float(dy @ dy)
    r = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else math.nan
    return r, (sxy / sxx if sxx > 0 else math.nan)


def dax_parity_expected(
    fact: pd.DataFrame, filters: Sequence[tuple[str, dict[str, object]]] = DEFAULT_PARITY_FILTERS
) -> pd.DataFrame:
    """Giá trị Python cho từng tổ hợp lọc; TV2 điền cột *_dax sau khi dựng measure trong Power BI."""
    rows = []
    for name, spec in filters:
        mask = np.ones(len(fact), dtype=bool)
        for col, value in spec.items():
            if isinstance(value, tuple):
                mask &= fact[col].between(*value).to_numpy(dtype=bool)
            else:
                mask &= fact[col].astype(str).eq(str(value)).to_numpy(dtype=bool)
        part = fact.loc[mask]
        age, odo = part["car_age"].to_numpy(dtype=float), part["odometer"].to_numpy(dtype=float)
        log_price = part["log_price"].to_numpy(dtype=float)
        enough = len(part) >= MIN_DAX_N
        r_age, slope = _two_pass(age, log_price) if enough else (math.nan, math.nan)
        r_odo, _ = _two_pass(odo, log_price) if enough else (math.nan, math.nan)
        rows.append({
            "filter_name": name, "filter_spec": repr(spec), "n_python": len(part),
            "median_price_python": float(part["price"].median()) if len(part) else math.nan,
            "pearson_age_logprice_python": r_age, "pearson_odometer_logprice_python": r_odo,
            "pct_per_year_python": 100.0 * (math.exp(slope) - 1.0) if np.isfinite(slope) else math.nan,
            "n_dax": math.nan, "median_price_dax": math.nan, "pearson_age_logprice_dax": math.nan,
            "pearson_odometer_logprice_dax": math.nan, "pct_per_year_dax": math.nan,
            "note": "" if enough else f"n < {MIN_DAX_N}: measure DAX trả BLANK",
        })
    return pd.DataFrame(rows)


def export_tables(tables: Mapping[str, pd.DataFrame], out_dir: Path, *, fmt: str = "both") -> list[Path]:
    """Ghi từng bảng ra CSV (UTF-8 có BOM) và/hoặc Parquet để nạp vào Power BI."""
    if fmt not in {"csv", "parquet", "both"}:
        raise ValueError("fmt phải là csv, parquet hoặc both")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, table in tables.items():
        if fmt in {"csv", "both"}:
            paths.append(out_dir / f"{name}.csv")
            table.to_csv(paths[-1], index=False, encoding="utf-8-sig")
        if fmt in {"parquet", "both"}:
            paths.append(out_dir / f"{name}.parquet")
            table.to_parquet(paths[-1], index=False, engine="pyarrow")
    return paths
