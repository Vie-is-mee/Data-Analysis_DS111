"""M1-10 · Bước 5 · Chọn biến → Workbook Sheet 06.

- ``to_contract``: giữ đúng 22 cột của REQUIRED_CLEAN_SCHEMA, ép đúng kiểu.
- ``vif_report``: VIF của các biến số (dùng hàm chung của TV3 để cùng cách tính).
"""

from __future__ import annotations

import pandas as pd

from configs.project_config import REQUIRED_CLEAN_SCHEMA
from src.data.transform_log import TransformLog
from src.models.feature_ranking import compute_vif_table

SHEET = "06_Selection"
VIF_FEATURES: list[str] = ["car_age", "log_odometer", "miles_per_year", "cylinders_num", "year"]


def to_contract(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    missing = [c for c in REQUIRED_CLEAN_SCHEMA if c not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột hợp đồng: {missing}")
    out = df[list(REQUIRED_CLEAN_SCHEMA)].astype(dict(REQUIRED_CLEAN_SCHEMA))
    dropped = sorted(set(df.columns) - set(out.columns))
    log.record("06_select", SHEET, "Giữ 22 cột hợp đồng",
               f"Bỏ cột trung gian/không thuộc hợp đồng: {', '.join(dropped)}", df, out)
    return out


def vif_report(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """TODO(M1-10): diễn giải VIF > 10 ở Sheet 06 (ví dụ car_age ~ year)."""
    return compute_vif_table(df, features or VIF_FEATURES)
