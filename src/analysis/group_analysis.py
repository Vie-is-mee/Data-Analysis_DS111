"""M2-06 · Khấu hao theo cụm hãng, hãng và đời xe.

Hai câu hỏi khác nhau, phải báo riêng:
1. Quy luật tuổi–giá chặt đến đâu: hệ số ρ (xem ``correlation_engine``).
2. Giá giảm nhanh đến đâu: hồi quy log-tuyến tính ln P = a + b·tuổi (+ kiểm soát),
   tốc độ %/năm = 100·(e^b − 1). Sai số chuẩn HC3 (chịu được phương sai thay đổi).

Dữ liệu cắt ngang (tin đăng 5 tuần năm 2026): "mất giá" là chênh lệch giữa các
đời xe tại cùng thời điểm, gồm cả hiệu ứng đời xe; không phải theo dõi một chiếc xe.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analysis.correlation_engine import z_critical

BRAND_LOOKUP_PATH = Path(__file__).with_name("brand_lookup.csv")
FOCUS_BRANDS: tuple[str, ...] = ("toyota", "honda", "ford", "chevrolet", "bmw", "mercedes-benz", "audi")
FOCUS_CLUSTERS: tuple[str, ...] = ("JP", "US", "DE")
AGE_WINDOW: tuple[int, int] = (1, 20)
MIN_GROUP_N: int = 100
MIN_TYPE_N: int = 30
CONTROL_SPECS: dict[str, tuple[str, ...]] = {
    "none": (),
    "odometer": ("odometer",),
    "odometer_type": ("odometer", "type"),
}
LOOKUP_COLUMNS: tuple[str, ...] = ("manufacturer_clean", "origin", "segment", "analysis_cluster")


def load_brand_lookup(path: Path | None = None) -> pd.DataFrame:
    """Đọc bảng tra hãng tĩnh (quyết định phân nhóm của TV2, không phải số đo)."""
    lookup = pd.read_csv(path or BRAND_LOOKUP_PATH, dtype="string", encoding="utf-8")
    missing = [col for col in LOOKUP_COLUMNS if col not in lookup.columns]
    if missing:
        raise KeyError(f"brand_lookup thiếu cột: {missing}")
    if lookup["manufacturer_clean"].duplicated().any():
        raise ValueError("brand_lookup có hãng trùng lặp")
    return lookup


def attach_brand_lookup(df: pd.DataFrame, lookup: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Thêm ``origin``, ``segment``, ``analysis_cluster``; trả về các hãng chưa có trong bảng tra."""
    keyed = lookup.set_index("manufacturer_clean")
    labels = df["manufacturer_clean"].astype("string")
    out = df.copy()
    for col, default in (("origin", "other"), ("segment", "unknown"), ("analysis_cluster", "other")):
        out[col] = labels.map(keyed[col]).fillna(default).astype("string")
    uncovered = sorted(set(labels.dropna().unique()) - set(keyed.index))
    return out, uncovered


def median_curve(
    df: pd.DataFrame,
    *,
    x: str = "car_age",
    y: str = "price",
    by: str | None = None,
    x_range: tuple[float, float] | None = None,
    min_n: int = 30,
) -> pd.DataFrame:
    """Trung vị và IQR của ``y`` theo từng giá trị ``x`` (tuỳ chọn theo nhóm ``by``).

    Đường trung vị theo nhóm thay cho scatter 270.000 điểm. Hệ số tương quan
    vẫn phải tính trên dữ liệu cấp tin, không tính trên các trung vị này.
    """
    cols = [x, y] + ([by] if by else [])
    frame = df[cols].dropna()
    if x_range is not None:
        frame = frame[frame[x].between(*x_range)]
    keys = [by, x] if by else [x]
    grouped = frame.groupby(keys, observed=True, sort=True)[y]
    table = pd.concat(
        {"n": grouped.size(), "median": grouped.median(),
         "q1": grouped.quantile(0.25), "q3": grouped.quantile(0.75)},
        axis=1,
    ).reset_index()
    table = table[table["n"] >= min_n].reset_index(drop=True)
    if by:
        table = table.rename(columns={by: "group_value"})
        table.insert(0, "group_col", by)
    else:
        table.insert(0, "group_value", "all")
        table.insert(0, "group_col", "all")
    table["group_value"] = table["group_value"].astype(str)
    return table


def _pct(log_coef: float) -> float:
    return 100.0 * (math.exp(log_coef) - 1.0)


def fit_log_depreciation(
    part: pd.DataFrame,
    controls: Sequence[str] = (),
    *,
    age_col: str = "car_age",
    target_col: str = "log_price",
    ci_level: float = 0.95,
) -> dict[str, float | str]:
    """OLS ln(giá) theo tuổi (+ số dặm/10.000, + biến giả loại xe), sai số HC3.

    Loại xe có < ``MIN_TYPE_N`` tin trong nhóm bị bỏ khi kiểm soát loại xe:
    nhãn quá hiếm tạo điểm đòn bẩy h ≈ 1 làm sai số HC3 không xác định.
    """
    use_odo, use_type = "odometer" in controls, "type" in controls
    cols = [age_col, target_col] + (["odometer"] if use_odo else []) + (["type"] if use_type else [])
    frame = part[cols].dropna()
    note = ""
    if use_type:
        counts = frame["type"].value_counts()
        keep = counts[counts >= MIN_TYPE_N].index
        dropped = int((~frame["type"].isin(keep)).sum())
        frame = frame[frame["type"].isin(keep)]
        note = f"bỏ {dropped} tin thuộc loại xe < {MIN_TYPE_N} tin" if dropped else ""
    design = pd.DataFrame({"const": 1.0, "age": frame[age_col].to_numpy(dtype=float)}, index=frame.index)
    if use_odo:
        design["odometer_10k"] = frame["odometer"].to_numpy(dtype=float) / 10_000.0
    if use_type:
        dummies = pd.get_dummies(frame["type"].astype(str), prefix="type", drop_first=True, dtype=float)
        design = pd.concat([design, dummies.loc[:, dummies.std(ddof=0) > 0]], axis=1)
    out: dict[str, float | str] = {"n": float(len(frame)), "n_params": float(design.shape[1]), "note": note}
    if len(frame) <= design.shape[1] + 2 or np.ptp(design["age"].to_numpy()) == 0:
        out["note"] = (note + "; " if note else "") + "không đủ dữ liệu để ước lượng"
        return out
    model = sm.OLS(frame[target_col].to_numpy(dtype=float), design.to_numpy(dtype=float)).fit(cov_type="HC3")
    params, bse, z = np.asarray(model.params), np.asarray(model.bse), z_critical(ci_level)
    out.update(slope_log=float(params[1]), slope_se=float(bse[1]), pct_per_year=_pct(params[1]),
               pct_ci_low=_pct(params[1] - z * bse[1]), pct_ci_high=_pct(params[1] + z * bse[1]),
               r2=float(model.rsquared))
    if use_odo:
        out.update(pct_per_10k_miles=_pct(params[2]), pct_10k_ci_low=_pct(params[2] - z * bse[2]),
                   pct_10k_ci_high=_pct(params[2] + z * bse[2]))
    return out


SLOPE_COLUMNS: tuple[str, ...] = (
    "dataset_version", "group_type", "group_value", "controls", "age_window", "n",
    "median_price", "slope_log", "slope_se", "pct_per_year", "pct_ci_low", "pct_ci_high",
    "pct_per_10k_miles", "pct_10k_ci_low", "pct_10k_ci_high", "r2", "n_params", "ci_level", "note",
)


def depreciation_table(
    df: pd.DataFrame,
    group_col: str | None,
    groups: Sequence[object] | None = None,
    *,
    specs: Sequence[str] = tuple(CONTROL_SPECS),
    age_window: tuple[int, int] = AGE_WINDOW,
    min_n: int = MIN_GROUP_N,
    group_type: str | None = None,
    dataset_version: str | None = None,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Tốc độ mất giá theo nhóm cho từng cấu hình kiểm soát trong ``specs``."""
    frame = df[df["car_age"].between(*age_window)]
    if group_col is None:
        parts = [("all", frame)]
    else:
        keys = frame[group_col]
        values = list(groups) if groups is not None else sorted(keys.dropna().unique().tolist(), key=str)
        parts = [(value, frame[keys.eq(value).fillna(False).to_numpy(dtype=bool)]) for value in values]
    rows = []
    for value, part in parts:
        for spec in specs:
            row: dict[str, object] = {
                "dataset_version": dataset_version, "group_type": group_type or group_col or "all",
                "group_value": str(value), "controls": spec, "age_window": f"{age_window[0]}-{age_window[1]}",
                "n": float(len(part)), "median_price": float(part["price"].median()) if len(part) else math.nan,
                "ci_level": ci_level, "note": "",
            }
            needed = [c for c in CONTROL_SPECS[spec] if c not in part.columns]
            if needed:
                row["note"] = f"Thiếu cột kiểm soát: {', '.join(needed)}"
            elif len(part) < min_n:
                row["note"] = f"n = {len(part)} < {min_n}: không báo độ dốc"
            else:
                fitted = fit_log_depreciation(part, CONTROL_SPECS[spec], ci_level=ci_level)
                row.update({k: v for k, v in fitted.items() if k != "note"})
                row["note"] = fitted.get("note", "")
            rows.append(row)
    return pd.DataFrame(rows).reindex(columns=list(SLOPE_COLUMNS))


def within_type_slopes(
    df: pd.DataFrame,
    group_col: str = "analysis_cluster",
    groups: Sequence[object] | None = FOCUS_CLUSTERS,
    *,
    types: Sequence[str] = ("sedan", "suv", "pickup"),
    spec: str = "odometer",
    dataset_version: str | None = None,
    min_n: int = MIN_GROUP_N,
) -> pd.DataFrame:
    """So tốc độ mất giá giữa các nhóm TRONG cùng một loại xe (kiểm soát cơ cấu loại xe)."""
    if "type" not in df.columns:
        return pd.DataFrame(columns=list(SLOPE_COLUMNS))
    frames = [
        depreciation_table(
            df[df["type"].eq(vehicle_type).fillna(False).to_numpy(dtype=bool)],
            group_col,
            groups,
            specs=(spec,),
            group_type=f"{group_col}|type={vehicle_type}",
            dataset_version=dataset_version,
            min_n=min_n,
        )
        for vehicle_type in types
    ]
    return pd.concat(frames, ignore_index=True)


def era_ranges(df: pd.DataFrame, *, era_col: str = "year_era") -> pd.DataFrame:
    """Số tin và miền năm/tuổi trong từng đời xe (để đọc đúng hiệu ứng thu hẹp miền)."""
    grouped = df.groupby(era_col, observed=True, sort=False)
    table = pd.DataFrame(
        {
            "n": grouped.size(),
            "year_min": grouped["year"].min(),
            "year_max": grouped["year"].max(),
            "age_min": grouped["car_age"].min(),
            "age_max": grouped["car_age"].max(),
            "median_price": grouped["price"].median(),
        }
    )
    order = {"Classic": 0, "Old": 1, "Mid": 2, "Modern": 3}
    table.index.name = era_col
    table = table.reset_index()
    table["sort_order"] = table[era_col].astype(str).map(order)
    return table.sort_values("sort_order").reset_index(drop=True)
