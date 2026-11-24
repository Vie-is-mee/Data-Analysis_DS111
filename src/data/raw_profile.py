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
    # Cột phụ trợ làm sạch (Sheet 01: "giữ (phụ trợ làm sạch)"), không vào file bàn giao
    "title", "VIN", "address_country", "odometer_broken", "odometer_rolled_over",
)
NUMERIC_RAW: tuple[str, ...] = ("id", "price", "year", "odometer")
# Cột có thể chứa dữ liệu cá nhân/định danh (SĐT, địa chỉ, VIN, tọa độ, link):
# không in ví dụ ra Sheet 01 vì file này được đưa lên Git (DATA_CONTRACT §4).
SENSITIVE_COLUMNS: frozenset[str] = frozenset({
    "url", "region_url", "image_url", "image_urls", "description", "VIN",
    "lat", "long", "address_street", "address_city", "address_postal_code",
})


def load_raw(path: Path | None = None, usecols: tuple[str, ...] | None = PIPELINE_COLUMNS) -> pd.DataFrame:
    """Đọc CSV gốc; mọi cột đọc dạng chuỗi rồi ép số có kiểm soát ở bước sau."""
    path = Path(path or CONFIG.raw_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có dữ liệu gốc: {path} (tải vehicles.csv vào data/raw/)")
    # Callable usecols: cột phụ trợ vắng mặt (CSV cũ/giả lập) không gây lỗi.
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
            "example_values": "[ẩn: có thể chứa dữ liệu cá nhân]" if col in SENSITIVE_COLUMNS
                              else " | ".join(map(str, non_null.unique()[:5])),
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
