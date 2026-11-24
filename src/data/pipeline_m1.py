"""Pipeline TV1: vehicles.csv → used_cars_clean_v1.parquet + formula_store_v1.joblib.

Chạy từ gốc repo:
    python -X utf8 -m src.data.pipeline_m1
    python -X utf8 -m src.core.bootstrap_and_validate validate ^
        --parquet data/processed/used_cars_clean_v1.parquet ^
        --formula-store data/processed/formula_store_v1.joblib
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from configs.project_config import CONFIG
from src.core.formula_store import FormulaStore
from src.data import feature_engineer, formatter, missing_handler, selector
from src.data.raw_profile import load_raw
from src.data.transform_log import TransformLog
from src.models.ablation_runner import assign_production_splits

DEFAULT_STORE = CONFIG.clean_parquet.with_name("formula_store_v1.joblib")
DEFAULT_LOG = CONFIG.clean_parquet.with_name("m1_transform_log.csv")
MIN_ROWS_TARGET = 200_000  # mục tiêu số dòng sạch (M1-06, rủi ro R1)


@dataclass(slots=True)
class M1Result:
    data: pd.DataFrame
    store: FormulaStore
    log: TransformLog


def build_clean_dataset(raw: pd.DataFrame, dedup_vin: bool = False) -> M1Result:
    log = TransformLog()
    store = FormulaStore()
    log.record("01_load", "00_Log", "Đọc dữ liệu gốc", "Điểm xuất phát", raw, raw)

    df = missing_handler.run(raw, log)                                       # Sheet 02
    df = formatter.run(df, log, dedup_vin=dedup_vin)                         # Sheet 03
    # Từ đây không lọc dòng nữa: FormulaStore khóa index.
    df = feature_engineer.run(df, store)                                     # Sheet 04
    # Sheet 05 (mã hóa) do TV3 làm trong từng fold theo DATA_CONTRACT.

    # split_set: dùng hàm chia chính thức của TV3 theo listing_id, seed 42
    # (không dựa vào giá). TODO: xác nhận với TV3 trước khi bàn giao V1.0.
    df["split_set"] = assign_production_splits(df["listing_id"].astype("int64"))
    df = formatter.flag_price_outliers(df, log)                              # Sheet 03 (cờ)

    df = selector.to_contract(df, log)                                       # Sheet 06
    if len(df) < MIN_ROWS_TARGET:
        print(f"[CẢNH BÁO] Chỉ còn {len(df):,} dòng (< {MIN_ROWS_TARGET:,}, rủi ro R1)")
    return M1Result(df, store, log)


def save(result: M1Result, parquet: Path, store_path: Path, log_path: Path) -> None:
    parquet.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_parquet(parquet, index=False, engine="pyarrow")
    result.store.save(store_path)
    result.log.to_dataframe().to_csv(log_path, index=False, encoding="utf-8-sig")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=CONFIG.raw_csv)
    p.add_argument("--out", type=Path, default=CONFIG.clean_parquet)
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--dedup-vin", action="store_true",
                   help="Bỏ tin đăng lại cùng VIN, giữ tin mới nhất")
    args = p.parse_args(argv)

    result = build_clean_dataset(load_raw(args.raw), dedup_vin=args.dedup_vin)
    save(result, args.out, args.store, args.log)
    raw_rows = result.log.to_dataframe()["rows_before"].iloc[0]
    print(f"Giữ {len(result.data):,}/{raw_rows:,} dòng "
          f"({len(result.data) / raw_rows:.1%}); {result.store.formula_count} công thức")
    print(f"→ {args.out}\n→ {args.store}\n→ {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
