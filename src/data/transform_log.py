"""Nhật ký biến đổi dữ liệu của TV1 (nguồn cho Workbook Sheet 00, 02, 03).

Mỗi thao tác làm thay đổi số dòng/cột phải được ghi lại với hành động,
lý do và tác động. Đây là bằng chứng cho tiêu chí "mỗi xử lý có hành động,
cơ sở và tác động số dòng/cột" của M1-12.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import pandas as pd


@dataclass(frozen=True, slots=True)
class TransformRecord:
    """Một dòng nhật ký: một thao tác cụ thể trên bảng dữ liệu."""

    step: str          # ví dụ "01_load", "02_missing", "03_format"
    sheet: str         # sheet Workbook nhận dòng này, ví dụ "02_Missing"
    action: str        # làm gì
    reason: str        # vì sao (cơ sở)
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    affected: int = 0  # số ô/dòng bị sửa giá trị mà không xóa dòng
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after


class TransformLog:
    """Danh sách TransformRecord theo thứ tự thực hiện."""

    def __init__(self) -> None:
        self._records: list[TransformRecord] = []

    def record(
        self,
        step: str,
        sheet: str,
        action: str,
        reason: str,
        before: pd.DataFrame,
        after: pd.DataFrame,
        affected: int = 0,
    ) -> None:
        """Ghi tác động của một thao tác dựa trên kích thước trước/sau."""
        self._records.append(
            TransformRecord(
                step=step,
                sheet=sheet,
                action=action,
                reason=reason,
                rows_before=len(before),
                rows_after=len(after),
                cols_before=before.shape[1],
                cols_after=after.shape[1],
                affected=int(affected),
            )
        )

    def __len__(self) -> int:
        return len(self._records)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [{**asdict(r), "rows_removed": r.rows_removed} for r in self._records]
        return pd.DataFrame(rows)
