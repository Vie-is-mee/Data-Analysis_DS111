"""M2-21 · Bảng số liệu chính của TV2 (``m2_key_numbers.csv``) cho Gate 3.

Mỗi số đưa lên slide, báo cáo hoặc dashboard phải lấy từ bảng này để TV3 so khớp
tự động giữa các sản phẩm. ``compare`` so hai phiên bản (ví dụ v0.9 với V1.0).

    python -X utf8 -m src.analysis.key_numbers compare --old a.csv --new b.csv
"""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLUMNS: tuple[str, ...] = ("key", "value", "unit", "dataset_version", "source_table", "computed_at")


class KeyNumbers:
    """Bộ gom số liệu chính; khóa ``snake_case`` cố định giữa các lần chạy."""

    def __init__(self, dataset_version: str | None) -> None:
        self.dataset_version = dataset_version
        self.computed_at = datetime.now(UTC).isoformat(timespec="seconds")
        self._rows: dict[str, dict[str, object]] = {}

    def add(self, key: str, value: object, unit: str, source: str) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.nan
        self._rows[key] = {"key": key, "value": number, "unit": unit, "dataset_version": self.dataset_version,
                           "source_table": source, "computed_at": self.computed_at}

    def add_from(self, table: pd.DataFrame, filters: dict[str, object], column: str, key: str,
                 unit: str, source: str) -> None:
        """Lấy đúng một ô từ ``table`` theo các điều kiện bằng; không có thì ghi NaN."""
        mask = np.ones(len(table), dtype=bool)
        for col, expected in filters.items():
            if col not in table.columns:
                mask[:] = False
                break
            mask &= table[col].astype(str).eq(str(expected)).to_numpy(dtype=bool)
        values = table.loc[mask, column] if column in table.columns else pd.Series(dtype=float)
        self.add(key, values.iloc[0] if len(values) == 1 else math.nan, unit, source)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._rows.values()), columns=list(KEY_COLUMNS))


def compare_key_numbers(old: pd.DataFrame, new: pd.DataFrame, *, rel_tol: float = 1e-9) -> pd.DataFrame:
    """So hai bảng key numbers theo ``key``: giá trị cũ/mới, chênh lệch, có đổi hay không."""
    merged = old[["key", "value", "dataset_version"]].merge(
        new[["key", "value", "dataset_version"]], on="key", how="outer",
        suffixes=("_old", "_new"), indicator=True,
    )
    merged["abs_diff"] = (merged["value_new"] - merged["value_old"]).abs()
    scale = merged[["value_old", "value_new"]].abs().max(axis=1).replace(0, np.nan)
    merged["rel_diff"] = merged["abs_diff"] / scale
    both = merged["_merge"].eq("both")
    one_nan = merged["value_old"].isna() ^ merged["value_new"].isna()
    merged["changed"] = ~both | one_nan | merged["rel_diff"].gt(rel_tol)
    merged["status"] = merged["_merge"].map({"both": "cả hai", "left_only": "chỉ bản cũ",
                                             "right_only": "chỉ bản mới"}).astype(str)
    return merged.drop(columns="_merge").sort_values(["changed", "key"], ascending=[False, True])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmp_parser = sub.add_parser("compare", help="So hai file m2_key_numbers.csv")
    cmp_parser.add_argument("--old", type=Path, required=True)
    cmp_parser.add_argument("--new", type=Path, required=True)
    cmp_parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    diff = compare_key_numbers(pd.read_csv(args.old), pd.read_csv(args.new))
    if args.out:
        diff.to_csv(args.out, index=False, encoding="utf-8-sig")
    changed = diff[diff["changed"]]
    print(f"{len(changed)} / {len(diff)} khóa thay đổi")
    print(changed.head(40).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
