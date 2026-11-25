"""M2-04 · EDA đơn biến: số liệu – hình – nhận định cho từng biến trọng tâm.

Ưu tiên thống kê bền vững (trung vị, IQR) vì giá và số dặm lệch phải mạnh.
``cylinders_num`` chỉ mô tả trên giá trị quan sát khi ABT có cờ
``cylinders_was_missing`` (khoảng 27% giá trị đã được TV1 điền theo nhóm).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

NUMERIC_EDA: tuple[str, ...] = ("price", "car_age", "odometer", "miles_per_year", "cylinders_num")
CATEGORICAL_EDA: tuple[str, ...] = (
    "condition_clean", "title_status_clean", "manufacturer_clean", "type", "drive",
    "fuel", "transmission", "seller_type", "year_era", "state",
)
# Biến số → cờ "đã điền" trong ABT: chỉ mô tả các dòng có giá trị quan sát.
OBSERVED_ONLY: Mapping[str, str] = {"cylinders_num": "cylinders_was_missing"}


def observed_values(df: pd.DataFrame, column: str) -> tuple[pd.Series, str]:
    """Trả về giá trị quan sát của ``column`` và ghi chú cách lấy."""
    values = df[column]
    flag = OBSERVED_ONLY.get(column)
    if flag is None:
        return values, ""
    if flag not in df.columns:
        return values, f"ABT thiếu {flag}: gồm cả giá trị đã điền"
    was_missing = df[flag].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return values[~was_missing], f"chỉ giá trị quan sát (bỏ {int(was_missing.sum())} dòng đã điền)"


def _skew(values: pd.Series) -> float:
    return float(values.skew()) if values.nunique() > 2 else math.nan


def numeric_summary(
    df: pd.DataFrame, columns: Sequence[str] = NUMERIC_EDA, *, dataset_version: str | None = None
) -> pd.DataFrame:
    """Một dòng mỗi biến số: n, % thiếu, trung vị, IQR, P1/P99, skewness trước/sau log."""
    rows = []
    for col in columns:
        if col not in df.columns:
            rows.append({"dataset_version": dataset_version, "column": col, "note": "Thiếu cột"})
            continue
        values, note = observed_values(df, col)
        numeric = pd.to_numeric(values, errors="coerce").astype("float64")
        finite = numeric[np.isfinite(numeric.to_numpy())]
        n_total, n = len(numeric), len(finite)
        row: dict[str, object] = {"dataset_version": dataset_version, "column": col, "n": n,
                                  "missing_pct": 100.0 * (n_total - n) / n_total if n_total else math.nan,
                                  "note": note}
        if n:
            q = finite.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            if (finite > 0).all():
                log_kind, logged = "ln(x)", np.log(finite)
            elif (finite >= 0).all():
                log_kind, logged = "ln(1+x)", np.log1p(finite)
            else:
                log_kind, logged = "", None
            row.update(mean=float(finite.mean()), std=float(finite.std()), min=float(finite.min()),
                       p01=float(q[0.01]), q1=float(q[0.25]), median=float(q[0.5]), q3=float(q[0.75]),
                       p99=float(q[0.99]), max=float(finite.max()), iqr=float(q[0.75] - q[0.25]),
                       skew=_skew(finite), log_transform=log_kind,
                       skew_log=_skew(logged) if logged is not None else math.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def categorical_summary(
    df: pd.DataFrame,
    columns: Sequence[str] = CATEGORICAL_EDA,
    *,
    top: int = 12,
    rare_share_pct: float = 1.0,
    dataset_version: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tổng quan mỗi biến, tỷ trọng các nhãn phổ biến nhất) cho biến phân loại."""
    overview, levels = [], []
    for col in columns:
        if col not in df.columns:
            overview.append({"dataset_version": dataset_version, "column": col, "note": "Thiếu cột"})
            continue
        counts = df[col].astype("string").fillna("<NA>").value_counts()
        share = 100.0 * counts / counts.sum()
        overview.append({
            "dataset_version": dataset_version, "column": col, "n": int(counts.sum()),
            "n_levels": int(len(counts)), "top_level": str(counts.index[0]),
            "top_share_pct": float(share.iloc[0]), "unknown_share_pct": float(share.get("unknown", 0.0)),
            "n_rare_levels": int((share < rare_share_pct).sum()),
            "rare_share_pct": float(share[share < rare_share_pct].sum()), "note": "",
        })
        for rank, (level, n) in enumerate(counts.head(top).items(), start=1):
            levels.append({"dataset_version": dataset_version, "column": col, "rank": rank,
                           "level": str(level), "n": int(n), "share_pct": float(share[level])})
        if len(counts) > top:
            levels.append({"dataset_version": dataset_version, "column": col, "rank": top + 1,
                           "level": "(các nhãn còn lại)", "n": int(counts.iloc[top:].sum()),
                           "share_pct": float(share.iloc[top:].sum())})
    return pd.DataFrame(overview), pd.DataFrame(levels)


def cleaning_funnel(log: pd.DataFrame) -> pd.DataFrame:
    """Phễu làm sạch từ nhật ký biến đổi của TV1 (``m1_transform_log.csv``).

    Dùng nhật ký thay vì đọc lại dữ liệu thô: số liệu khớp đúng Workbook của TV1.
    """
    required = {"step", "action", "rows_before", "rows_after"}
    if not required <= set(log.columns):
        raise KeyError(f"Nhật ký biến đổi thiếu cột: {sorted(required - set(log.columns))}")
    keep = [c for c in ("step", "sheet", "action", "reason", "rows_before", "rows_after") if c in log.columns]
    table = log.loc[:, keep].copy()
    raw_rows = float(table["rows_before"].iloc[0])
    table["rows_removed"] = table["rows_before"] - table["rows_after"]
    table["removed_pct_of_raw"] = 100.0 * table["rows_removed"] / raw_rows
    table["kept_pct_of_raw"] = 100.0 * table["rows_after"] / raw_rows
    table["changes_rows"] = table["rows_removed"].ne(0)
    return table.reset_index(drop=True)


def vn_number(value: float, digits: int = 0) -> str:
    """Định dạng số kiểu Việt Nam: 12.995 và 0,25."""
    if value is None or not np.isfinite(value):
        return "không xác định"
    text = f"{value:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def insight_sentences(summary: pd.DataFrame) -> pd.DataFrame:
    """Câu nhận định định lượng (bản nháp) cho mỗi biến số; TV2 biên tập lại khi viết báo cáo."""
    templates = {
        "price": "50% tin có giá từ {q1} đến {q3} USD (trung vị {median} USD); độ lệch {skew} → {skew_log} sau {log}.",
        "car_age": "Tuổi xe trung vị {median} năm; 50% tin ở khoảng {q1}–{q3} năm; P99 = {p99} năm.",
        "odometer": "Số dặm trung vị {median}; IQR {q1}–{q3}; độ lệch {skew} → {skew_log} sau {log}.",
    }
    rows = []
    for record in summary.to_dict("records"):
        if "median" not in record or not np.isfinite(record.get("median", math.nan)):
            continue
        fields = {key: vn_number(record.get(key, math.nan)) for key in ("q1", "q3", "median", "p99")}
        fields.update(skew=vn_number(record.get("skew", math.nan), 2),
                      skew_log=vn_number(record.get("skew_log", math.nan), 2),
                      log=record.get("log_transform") or "log")
        template = templates.get(
            record["column"], record["column"] + ": trung vị {median}; IQR {q1}–{q3}; độ lệch {skew}."
        )
        sentence = template.format(**fields)
        if record.get("note"):
            sentence += f" ({record['note']})"
        rows.append({"column": record["column"], "sentence": sentence})
    return pd.DataFrame(rows, columns=["column", "sentence"])
