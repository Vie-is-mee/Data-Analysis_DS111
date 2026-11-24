"""Kho công thức có hai vùng dữ liệu độc lập cho đồ án DS111.

``_series_memory`` chỉ lưu các chuỗi giá trị ``pandas.Series``;
``_metadata_registry`` chỉ lưu hồ sơ ``FormulaMetadata``. Không lưu lời
thuyết minh hoặc công thức vào thuộc tính của ``Series``.

Lưu ý an toàn: joblib sử dụng cơ chế pickle. Chỉ nạp các tệp do nhóm tin cậy
và tự tạo, không nạp tệp từ nguồn không xác định.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass(slots=True)
class FormulaMetadata:
    """Hồ sơ một phép biến đổi, không chứa dữ liệu đầu ra dạng ``Series``.

    ``formula_id`` định danh duy nhất một công thức. ``output_series_key`` là
    khóa trỏ vào vùng nhớ kết quả, còn ``feature_name`` là tên dễ đọc.
    ``created_at`` lưu thời gian dạng ISO 8601; mặc định dùng giờ UTC.
    """

    formula_id: str
    feature_name: str
    step_category: str
    expression_plain: str
    expression_latex: str
    input_columns: list[str]
    output_series_key: str
    theoretical_rationale: str
    suitable_algorithms: list[str]
    stats_before: dict[str, float] = field(default_factory=dict)
    stats_after: dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        """Chặn hồ sơ thiếu khóa hoặc sai kiểu dữ liệu ngay lúc khởi tạo."""
        required_text = (
            "formula_id",
            "feature_name",
            "step_category",
            "expression_plain",
            "expression_latex",
            "output_series_key",
            "theoretical_rationale",
            "created_at",
        )
        for attr in required_text:
            value = getattr(self, attr)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"FormulaMetadata.{attr} phải là chuỗi không rỗng")
        for attr in ("input_columns", "suitable_algorithms"):
            items = getattr(self, attr)
            if not isinstance(items, list) or any(
                not isinstance(item, str) or not item.strip() for item in items
            ):
                raise TypeError(f"FormulaMetadata.{attr} phải là list[str] hợp lệ")
            if len(set(items)) != len(items):
                raise ValueError(f"FormulaMetadata.{attr} có giá trị trùng")
        for attr in ("stats_before", "stats_after"):
            values = getattr(self, attr)
            if not isinstance(values, dict) or any(
                not isinstance(key, str)
                or not isinstance(value, (int, float, np.integer, np.floating))
                for key, value in values.items()
            ):
                raise TypeError(f"FormulaMetadata.{attr} phải là dict[str, float]")
        try:
            datetime.fromisoformat(self.created_at)
        except ValueError as exc:
            raise ValueError("created_at phải là ngày giờ ISO 8601") from exc


class FormulaStore:
    """Quản lý kết quả công thức và hồ sơ công thức bằng hai kho độc lập.

    Tất cả các chuỗi kết quả dùng cùng một ``pd.Index`` về giá trị và thứ tự.
    Không cho phép dùng cùng một khóa đầu ra cho hai công thức, nhằm tránh ghi
    đè bằng chứng trong Workbook.
    """

    _SERIALIZATION_VERSION: int = 1

    def __init__(self) -> None:
        """Khởi tạo hai vùng nhớ và chỉ mục đồng bộ."""
        self._series_memory: dict[str, pd.Series] = {}
        self._metadata_registry: dict[str, FormulaMetadata] = {}
        self._reference_index: pd.Index | None = None

    @staticmethod
    def _quick_stats(series: pd.Series) -> dict[str, float]:
        """Thống kê khuyết và thống kê số, không ép chuỗi phân loại thành số.

        Độ lệch (skewness) để ``NaN`` nếu không đủ ba giá trị hoặc chuỗi hằng.
        ``std`` dùng hiệu chỉnh mẫu (ddof=1), cùng định nghĩa với pandas.
        """
        stats: dict[str, float] = {
            "missing_count": float(series.isna().sum()),
        }
        if is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
            series.dtype
        ):
            values = pd.to_numeric(series, errors="coerce")
            finite = values.replace([np.inf, -np.inf], np.nan).dropna()
            stats["non_finite_count"] = float(
                np.isinf(np.asarray(values.dropna(), dtype="float64")).sum()
            )
            if finite.empty:
                stats.update(mean=float("nan"), std=float("nan"), skewness=float("nan"))
            else:
                stats["mean"] = float(finite.mean())
                stats["std"] = (
                    float(finite.std(ddof=1)) if len(finite) >= 2 else float("nan")
                )
                stats["skewness"] = (
                    float(finite.skew())
                    if len(finite) >= 3 and finite.nunique() > 1
                    else float("nan")
                )
        else:
            stats.update(mean=float("nan"), std=float("nan"), skewness=float("nan"))
        return stats

    @staticmethod
    def _clone_metadata(metadata: FormulaMetadata) -> FormulaMetadata:
        """Sao chép các cấu trúc lồng nhau để tránh sửa ngầm hồ sơ đã lưu."""
        return replace(
            metadata,
            input_columns=list(metadata.input_columns),
            suitable_algorithms=list(metadata.suitable_algorithms),
            stats_before=dict(metadata.stats_before),
            stats_after=dict(metadata.stats_after),
        )

    @property
    def formula_count(self) -> int:
        """Số công thức đã đăng ký."""
        return len(self._metadata_registry)

    @property
    def series_keys(self) -> tuple[str, ...]:
        """Danh sách khóa kết quả, dạng bất biến để có thể kiểm tra bàn giao."""
        return tuple(self._series_memory)

    def list_metadata(self) -> tuple[FormulaMetadata, ...]:
        """Trả về danh sách bản sao hồ sơ, không làm lộ kho nội bộ."""
        return tuple(self._clone_metadata(v) for v in self._metadata_registry.values())

    def register_formula(self, series: pd.Series, metadata: FormulaMetadata) -> None:
        """Đăng ký kết quả và hồ sơ theo nguyên tắc thành công cả hai hoặc không.

        Raises:
            TypeError: ``series``/``metadata`` không đúng kiểu.
            ValueError: trùng ID, trùng khóa, sai tên chuỗi hoặc lệch chỉ mục.
        """
        if not isinstance(series, pd.Series):
            raise TypeError("Kết quả công thức phải là pandas.Series")
        if not isinstance(metadata, FormulaMetadata):
            raise TypeError("metadata phải là FormulaMetadata")
        formula_id = metadata.formula_id
        key = metadata.output_series_key
        if formula_id in self._metadata_registry:
            raise ValueError(f"formula_id đã tồn tại: {formula_id}")
        if key in self._series_memory:
            raise ValueError(f"output_series_key đã tồn tại: {key}")
        if series.name is not None and series.name != key:
            raise ValueError(
                f"Series.name={series.name!r} phải khớp output_series_key={key!r}"
            )
        if not series.index.is_unique:
            raise ValueError("Chỉ mục Series có giá trị trùng, không thể đồng bộ")
        if self._reference_index is not None and not series.index.equals(
            self._reference_index
        ):
            raise ValueError("Chỉ mục Series khác các kết quả đã lưu (cả thứ tự)")

        stored_series = series.copy(deep=True).rename(key)
        stored_series.attrs.clear()
        stored_metadata = self._clone_metadata(metadata)
        if not stored_metadata.stats_after:
            stored_metadata.stats_after = self._quick_stats(stored_series)
        else:
            # Ghi thêm chỉ số còn thiếu, không ghi đè số liệu được kiểm chứng.
            computed = self._quick_stats(stored_series)
            stored_metadata.stats_after = {**computed, **stored_metadata.stats_after}

        if self._reference_index is None:
            self._reference_index = stored_series.index.copy(deep=True)
        self._series_memory[key] = stored_series
        self._metadata_registry[formula_id] = stored_metadata

    def get_series(self, key: str) -> pd.Series:
        """Lấy bản sao giá trị công thức, tránh vô tình sửa kho gốc."""
        if key not in self._series_memory:
            raise KeyError(f"Không có Series mang khóa {key!r}")
        return self._series_memory[key].copy(deep=True)

    def get_metadata(self, formula_id: str) -> FormulaMetadata:
        """Lấy bản sao hồ sơ, không cho sửa gián tiếp trong registry."""
        if formula_id not in self._metadata_registry:
            raise KeyError(f"Không có công thức {formula_id!r}")
        return self._clone_metadata(self._metadata_registry[formula_id])

    def apply_and_register(
        self,
        dataframe: pd.DataFrame,
        transform: Callable[[pd.DataFrame], pd.Series],
        metadata: FormulaMetadata,
    ) -> pd.Series:
        """Tính công thức, thu thập thống kê và đăng ký vào cả hai vùng nhớ.

        ``stats_before`` lấy từ cột số đầu tiên trong ``input_columns``.
        Chỉ dùng ``dataframe`` như nguồn đầu vào, không chỉnh trực tiếp bảng.
        Chỉ khi đăng ký thành công mới trả về chuỗi giá trị đã tính.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe phải là pandas.DataFrame")
        if not callable(transform):
            raise TypeError("transform phải là một hàm nhận DataFrame")
        if not isinstance(metadata, FormulaMetadata):
            raise TypeError("metadata phải là FormulaMetadata")
        missing_columns = [
            c for c in metadata.input_columns if c not in dataframe.columns
        ]
        if missing_columns:
            raise KeyError(f"Thiếu các cột đầu vào: {missing_columns}")
        result = transform(dataframe.copy(deep=True))
        if not isinstance(result, pd.Series):
            raise TypeError("Hàm transform phải trả về pandas.Series")
        if not result.index.equals(dataframe.index):
            raise ValueError("Index đầu ra khác index của DataFrame đầu vào")
        numeric_input = next(
            (c for c in metadata.input_columns if is_numeric_dtype(dataframe[c].dtype)),
            None,
        )
        updated = self._clone_metadata(metadata)
        if not updated.stats_before and numeric_input is not None:
            updated.stats_before = self._quick_stats(dataframe[numeric_input])
        result = result.rename(updated.output_series_key)
        self.register_formula(result, updated)
        return self.get_series(updated.output_series_key)

    def to_dataframe_report(self) -> pd.DataFrame:
        """Xuất bảng phẳng dùng cho Sheet 04 của Project_Workbook.xlsx.

        Giữ mỗi công thức một hàng; danh sách chuyển thành chuỗi dễ đọc,
        thống kê trước/sau thành các cột có tiền tố phân biệt rõ.
        """
        base_columns = [
            "formula_id",
            "feature_name",
            "step_category",
            "expression_plain",
            "expression_latex",
            "input_columns",
            "output_series_key",
            "theoretical_rationale",
            "suitable_algorithms",
            "created_at",
        ]
        stat_names = ["mean", "std", "skewness", "missing_count", "non_finite_count"]
        stat_columns = [
            f"{side}_{name}" for side in ("before", "after") for name in stat_names
        ]
        rows: list[dict[str, object]] = []
        for record in self._metadata_registry.values():
            row: dict[str, object] = {
                "formula_id": record.formula_id,
                "feature_name": record.feature_name,
                "step_category": record.step_category,
                "expression_plain": record.expression_plain,
                "expression_latex": record.expression_latex,
                "input_columns": ", ".join(record.input_columns),
                "output_series_key": record.output_series_key,
                "theoretical_rationale": record.theoretical_rationale,
                "suitable_algorithms": ", ".join(record.suitable_algorithms),
                "created_at": record.created_at,
            }
            for side, stats in (
                ("before", record.stats_before),
                ("after", record.stats_after),
            ):
                for name in stat_names:
                    row[f"{side}_{name}"] = stats.get(name, float("nan"))
            rows.append(row)
        return pd.DataFrame.from_records(rows, columns=base_columns + stat_columns)

    def save(self, filepath: Path) -> None:
        """Lưu nguyên vẹn hai vùng nhớ vào một tệp joblib (ghi tạm rồi đổi tên).

        Chỉ gọi với tệp nhóm tin cậy. Dữ liệu được giữ nguyên dtype và index.
        """
        path = Path(filepath)
        if path.suffix.lower() != ".joblib":
            raise ValueError("Tệp FormulaStore phải có đuôi .joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        payload: dict[str, object] = {
            "format_version": self._SERIALIZATION_VERSION,
            "series_memory": self._series_memory,
            "metadata_registry": self._metadata_registry,
            "reference_index": self._reference_index,
        }
        try:
            joblib.dump(payload, temporary, compress=3)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, filepath: Path) -> FormulaStore:
        """Nạp tệp joblib nội bộ, kiểm tra cấu trúc trước khi trả về đối tượng.

        **Không nạp tệp joblib từ người lạ:** cơ chế pickle có khả năng thực thi
        mã trong lúc nạp.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy FormulaStore: {path}")
        payload = joblib.load(path)
        if (
            not isinstance(payload, dict)
            or payload.get("format_version") != cls._SERIALIZATION_VERSION
        ):
            raise ValueError("Phiên bản hoặc định dạng FormulaStore không được hỗ trợ")
        series_memory = payload.get("series_memory")
        metadata_registry = payload.get("metadata_registry")
        if not isinstance(series_memory, dict) or not isinstance(
            metadata_registry, dict
        ):
            raise TypeError("Tệp FormulaStore thiếu một trong hai vùng nhớ")
        restored = cls()
        for formula_id, metadata in metadata_registry.items():
            if (
                not isinstance(metadata, FormulaMetadata)
                or formula_id != metadata.formula_id
            ):
                raise ValueError("Hồ sơ công thức trong joblib không hợp lệ")
            key = metadata.output_series_key
            series = series_memory.get(key)
            if not isinstance(series, pd.Series):
                raise TypeError(f"Thiếu Series của công thức {formula_id!r}")
            restored.register_formula(series, metadata)
        if set(series_memory) != set(restored._series_memory):
            raise ValueError("Vùng Series có khóa mồ côi không gắn với metadata")
        reference_index = payload.get("reference_index")
        if reference_index is not None and not isinstance(reference_index, pd.Index):
            raise ValueError("reference_index không hợp lệ")
        if reference_index is not None and (
            restored._reference_index is None
            or not reference_index.equals(restored._reference_index)
        ):
            raise ValueError("Index đã lưu không trùng index trong công thức")
        return restored
