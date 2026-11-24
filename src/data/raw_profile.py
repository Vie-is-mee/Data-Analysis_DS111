"""M1-02 · Kiểm kê dữ liệu gốc → Workbook Sheet 01_Raw_Profile.

Đọc ``vehicles.csv`` (26 cột) và lập hồ sơ từng cột: kiểu, tỷ lệ khuyết,
số nhãn, ví dụ. Hai cột ``keep_drop`` và ``reason`` để TV1 tự điền quyết định.

Chạy:  python -X utf8 -m src.data.raw_profile
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from configs.project_config import CONFIG

# Cột gốc cần cho pipeline M1. Các cột còn lại (url, image_url, VIN,
# description, county, lat, long, ...) chỉ được kiểm kê ở Sheet 01.
PIPELINE_COLUMNS: tuple[str, ...] = (
    "id", "region", "price", "year", "manufacturer", "model", "condition",
    "cylinders", "fuel", "odometer", "title_status", "transmission", "drive",
    "type", "state", "posting_date",
)
NUMERIC_RAW: tuple[str, ...] = ("id", "price", "year", "odometer")


def load_raw(path: Path | None = None, usecols: tuple[str, ...] | None = PIPELINE_COLUMNS) -> pd.DataFrame:
    """Đọc CSV gốc; mọi cột đọc dạng chuỗi rồi ép số có kiểm soát ở bước sau."""
    path = Path(path or CONFIG.raw_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có dữ liệu gốc: {path} (tải vehicles.csv vào data/raw/)")
    cols = None if usecols is None else (lambda c: c in usecols)
    return pd.read_csv(path, usecols=cols, dtype="string", encoding="utf-8")


def profile_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Bảng hồ sơ một dòng cho mỗi cột gốc."""
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        numeric = pd.to_numeric(non_null, errors="coerce")
        rows.append({
            "column": col,
            "dtype_raw": str(s.dtype),
            "looks_numeric": bool(len(non_null)) and numeric.notna().mean() > 0.95,
            "n_rows": len(s),
            "missing_count": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "n_unique": int(s.nunique(dropna=True)),
            "example_values": " | ".join(map(str, non_null.unique()[:5])),
            "keep_drop": "",   # TODO(M1-02): điền "giữ" / "bỏ"
            "reason": "",      # TODO(M1-02): lý do (ví dụ: URL không mang thông tin giá)
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_raw(usecols=None)
    profile = profile_raw(df)
    out = CONFIG.root / "workbook" / "01_Raw_Profile.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{len(df):,} dòng × {df.shape[1]} cột → {out}")


if __name__ == "__main__":
    main()
