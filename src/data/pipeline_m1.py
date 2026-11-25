"""Pipeline TV1: vehicles.csv → used_cars_clean_v1.parquet + formula_store_v1.joblib.

Quy trình 2 lượt theo DATA_CONTRACT 1.1 (chạy từ gốc repo):

1. ``python -X utf8 -m src.data.pipeline_m1``
   → ABT bản phát triển (chưa gộp tin đăng lại, split theo ID) +
     ``vin_candidates_local.parquet`` (listing_id, VIN, posting_date; CHỈ lưu nội bộ).
2. TV3: ``python -X utf8 -m src.models.split_manifest --vin-listings
   data/processed/vin_candidates_local.parquet --output data/processed/split_manifest_v1.parquet``
3. ``python -X utf8 -m src.data.pipeline_m1 --split-manifest data/processed/split_manifest_v1.parquet``
   → ABT chính thức: chỉ giữ ID trong manifest, split_set lấy từ manifest.
4. ``python -X utf8 -m src.core.bootstrap_and_validate validate --parquet ... --formula-store ...``
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
from src.models.split_manifest import assign_production_splits, validate_split_manifest

DEFAULT_STORE = CONFIG.clean_parquet.with_name("formula_store_v1.joblib")
DEFAULT_LOG = CONFIG.clean_parquet.with_name("m1_transform_log.csv")
DEFAULT_VIN = CONFIG.clean_parquet.with_name("vin_candidates_local.parquet")  # nội bộ, không lên Git
DEFAULT_VIN_REPORT = CONFIG.clean_parquet.with_name("m1_vin_report.csv")
MIN_ROWS_TARGET = 200_000  # mục tiêu số dòng sạch (M1-06, rủi ro R1)


@dataclass(slots=True)
class M1Result:
    data: pd.DataFrame
    store: FormulaStore
    log: TransformLog
    vin_candidates: pd.DataFrame            # đầu vào manifest TV3 (có VIN thô → chỉ nội bộ)
    vin_report: pd.DataFrame | None = None  # số dòng/giá trung vị trước–sau gộp theo seller_type


def vin_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Metadata VIN sau lọc nghiệp vụ, trước gộp tin (đúng đầu vào split_manifest của TV3)."""
    out = pd.DataFrame({"listing_id": df["listing_id"].astype("int64")})
    out["VIN"] = (df["VIN"].astype("string") if "VIN" in df
                  else pd.Series(pd.NA, index=df.index, dtype="string"))
    out["posting_date"] = df["posting_date"].astype("string")
    return out.sort_values("listing_id").reset_index(drop=True)


def vin_report(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Số dòng + giá trung vị trước/sau gộp tin đăng lại, theo loại người bán (hợp đồng 1.1)."""
    groups = [(k, before[before["seller_type"] == k], after[after["seller_type"] == k])
              for k in sorted(before["seller_type"].unique())]
    rows = []
    for name, b, a in [*groups, ("Tổng", before, after)]:
        removed = len(b) - len(a)
        rows.append({"seller_type": name, "Số dòng trước": len(b), "Số dòng sau": len(a),
                     "Bị gộp": removed, "% bị gộp": round(removed / max(len(b), 1) * 100, 2),
                     "Giá trung vị trước": float(b["price"].median()),
                     "Giá trung vị sau": float(a["price"].median())})
    return pd.DataFrame(rows)


def apply_manifest(df: pd.DataFrame, manifest: pd.DataFrame, log: TransformLog) -> pd.DataFrame:
    """Giữ đúng các listing_id trong manifest TV3 (tin đăng lại đã bị loại ở đó)."""
    keep = pd.Index(manifest["listing_id"].astype("int64"))
    stale = keep.difference(pd.Index(df["listing_id"]))
    if len(stale):
        raise ValueError(f"Manifest có {len(stale)} listing_id không còn trong ABT (manifest cũ). "
                         "Tạo lại manifest từ vin_candidates_local.parquet của lần chạy này.")
    out = df[df["listing_id"].isin(keep)]
    log.record("03_format", "03_Format", "Gộp tin đăng lại theo manifest VIN (TV3)",
               "Cùng VIN hợp lệ (qua chữ số kiểm tra) → giữ tin mới nhất; tránh lặp xe giữa train/test",
               df, out)
    return out


def build_clean_dataset(raw: pd.DataFrame, split_manifest: pd.DataFrame | None = None) -> M1Result:
    log = TransformLog()
    store = FormulaStore()
    log.record("01_load", "00_Log", "Đọc dữ liệu gốc", "Điểm xuất phát", raw, raw)

    df = missing_handler.run(raw, log)                                       # Sheet 02
    df = formatter.run(df, log)                                              # Sheet 03
    candidates = vin_candidates(df)
    report = None
    if split_manifest is not None:
        before = df
        df = apply_manifest(df, split_manifest, log)                         # Sheet 03
        report = vin_report(before, df)
    else:
        print("[LƯU Ý] Chưa có manifest: chưa gộp tin đăng lại, split theo ID (chỉ dùng phát triển).")
    df = df.reset_index(drop=True)
    # Từ đây không lọc dòng nữa: FormulaStore khóa index.
    df = feature_engineer.run(df, store)                                     # Sheet 04
    # Sheet 05 (mã hóa) do TV3 làm trong từng fold theo DATA_CONTRACT.

    # split_set do TV3 quyết định (không dựa vào giá).
    ids = df["listing_id"].astype("int64")
    if split_manifest is not None:
        df["split_set"] = validate_split_manifest(split_manifest, ids)["split_set"].to_numpy()
    else:
        df["split_set"] = assign_production_splits(ids)
    df = formatter.flag_price_outliers(df, log)                              # Sheet 03 (cờ)

    df = selector.to_contract(df, log)                                       # Sheet 06
    if len(df) < MIN_ROWS_TARGET:
        print(f"[CẢNH BÁO] Chỉ còn {len(df):,} dòng (< {MIN_ROWS_TARGET:,}, rủi ro R1)")
    return M1Result(df, store, log, candidates, report)


def save(result: M1Result, parquet: Path, store_path: Path, log_path: Path,
         vin_path: Path | None = None, vin_report_path: Path | None = None) -> None:
    parquet.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_parquet(parquet, index=False, engine="pyarrow")
    result.store.save(store_path)
    result.log.to_dataframe().to_csv(log_path, index=False, encoding="utf-8-sig")
    if vin_path is not None:
        result.vin_candidates.to_parquet(vin_path, index=False, engine="pyarrow")
    if vin_report_path is not None:
        if result.vin_report is not None:
            result.vin_report.to_csv(vin_report_path, index=False, encoding="utf-8-sig")
        else:
            Path(vin_report_path).unlink(missing_ok=True)  # không giữ báo cáo của lần chạy cũ


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=CONFIG.raw_csv)
    p.add_argument("--out", type=Path, default=CONFIG.clean_parquet)
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--vin-out", type=Path, default=DEFAULT_VIN,
                   help="Danh sách VIN nội bộ cho split_manifest của TV3 (không lên Git)")
    p.add_argument("--vin-report", type=Path, default=DEFAULT_VIN_REPORT)
    p.add_argument("--split-manifest", type=Path, default=None,
                   help="Manifest TV3: lọc tin đăng lại + lấy split_set")
    args = p.parse_args(argv)

    manifest = pd.read_parquet(args.split_manifest) if args.split_manifest else None
    result = build_clean_dataset(load_raw(args.raw), split_manifest=manifest)
    save(result, args.out, args.store, args.log, args.vin_out, args.vin_report)
    raw_rows = result.log.to_dataframe()["rows_before"].iloc[0]
    print(f"Giữ {len(result.data):,}/{raw_rows:,} dòng "
          f"({len(result.data) / raw_rows:.1%}); {result.store.formula_count} công thức")
    print(f"→ {args.out}\n→ {args.store}\n→ {args.log}\n→ {args.vin_out} (nội bộ)")
    if result.vin_report is not None:
        print(result.vin_report.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
