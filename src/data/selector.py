"""M1-10 · Bước 5 · Chọn biến → Workbook Sheet 06.

- ``to_contract``: giữ 22 cột của REQUIRED_CLEAN_SCHEMA (+ cột cờ bổ sung), ép đúng kiểu.
- ``vif_report``: VIF của các biến số (dùng hàm chung của TV3 để cùng cách tính).
"""

from __future__ import annotations

import pandas as pd

from configs.project_config import REQUIRED_CLEAN_SCHEMA
from src.data.transform_log import TransformLog
from src.models.feature_ranking import compute_vif_table

SHEET = "06_Selection"
# Cột bổ sung ngoài 22 cột hợp đồng (DATA_CONTRACT 1.1; validator cho phép cột thừa).
EXTRA_COLUMNS: dict[str, str] = {
    # Hợp đồng 1.1 (TV3): boolean không rỗng / string có miền cố định.
    "condition_missing": "bool",            # condition gốc rỗng, trước khi điền 'unknown'
    "cylinders_was_missing": "bool",        # xi-lanh khuyết sau tách số + xe điện = 0, trước khi điền
    "seller_type": "string",                # private / dealer / unknown
    "car_age_squared": "float64",           # M1-07: tuổi xe bậc 2 (đã lưu FormulaStore)
    # Cột truy vết/kiểm tra do TV1 thêm (TV3 đồng ý 24/11):
    "manufacturer_source": "string",        # original / title / none
    "manufacturer_title_conflict": "bool",  # hãng gốc khác hãng trong title
    "price_outlier_iqr": "bool",            # ngoài 1,5·IQR log(price) của hãng (ngưỡng từ train)
}
VIF_FEATURES: list[str] = ["car_age", "log_odometer", "miles_per_year", "cylinders_num", "year"]


def to_contract(df: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    missing = [c for c in REQUIRED_CLEAN_SCHEMA if c not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột hợp đồng: {missing}")
    schema = dict(REQUIRED_CLEAN_SCHEMA) | {c: t for c, t in EXTRA_COLUMNS.items() if c in df.columns}
    out = df[list(schema)].astype(schema)
    dropped = sorted(set(df.columns) - set(out.columns))
    log.record("06_select", SHEET, f"Giữ 22 cột hợp đồng + {len(out.columns) - 22} cột cờ bổ sung",
               f"Bỏ cột trung gian/không thuộc hợp đồng: {', '.join(dropped)}", df, out)
    return out


def vif_report(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """TODO(M1-10): diễn giải VIF > 10 ở Sheet 06 (ví dụ car_age ~ year)."""
    return compute_vif_table(df, features or VIF_FEATURES)
