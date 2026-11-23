"""Cấu hình duy nhất cho đồ án DS111 (Python 3.11, CPU-only).

Không nhập đường dẫn tuyệt đối theo máy cá nhân. Các phép biến đổi dựa trên nhãn
đích và các quy tắc chia tập dữ liệu phải được khớp CHỈ trên tập huấn luyện.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Mapping


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RANDOM_STATE: Final[int] = 42
CV_FOLDS: Final[int] = 5

# Tên cột và kiểu dữ liệu Parquet thống nhất theo Data Contract.
REQUIRED_CLEAN_SCHEMA: Final[Mapping[str, str]] = {
    "listing_id": "int64",
    "price": "float64",
    "log_price": "float64",
    "year": "int32",
    "posting_year": "int32",
    "car_age": "int32",
    "odometer": "float64",
    "log_odometer": "float64",
    "miles_per_year": "float64",
    "manufacturer_clean": "string",
    "model_clean": "string",
    "condition_clean": "string",
    "title_status_clean": "string",
    "cylinders_num": "float64",
    "fuel": "string",
    "transmission": "string",
    "drive": "string",
    "type": "string",
    "state": "string",
    "region": "string",
    "year_era": "string",
    "split_set": "string",
}

# 'unknown' là một nhãn riêng; KHÔNG tùy tiện coi là một mức chất lượng.
CONDITION_ORDINAL: Final[Mapping[str, int]] = {
    "salvage": 0,
    "fair": 1,
    "good": 2,
    "excellent": 3,
    "like new": 4,
    "new": 5,
}

# Chỉ là THỨ TỰ THỰC NGHIỆM, KHÔNG PHẢI quy luật có sẵn của Craigslist.
# Các trạng thái title có bản chất pháp lý khác nhau, không tạo thành thứ tự toàn phần.
# Cần đối chứng One-Hot với sơ đồ này trong Exp 3; 'unknown' mã hóa riêng.
TITLE_STATUS_ORDINAL_EXPERIMENTAL: Final[Mapping[str, int]] = {
    "parts only": 0,
    "missing": 1,
    "salvage": 2,
    "rebuilt": 3,
    "lien": 4,
    "clean": 5,
}

LOW_CARDINALITY_CATEGORICAL: Final[tuple[str, ...]] = (
    "fuel", "transmission", "drive", "type"
)
HIGH_CARDINALITY_CATEGORICAL: Final[tuple[str, ...]] = (
    "manufacturer_clean", "state"
)


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Cấu hình không thể gán lại; đường dẫn đều bắt đầu từ gốc repository.

    frozen=True chặn gán lại thuộc tính; MappingProxyType không cần thiết vì
    ánh xạ module-level không được đặt trực tiếp làm thuộc tính có thể biến đổi.
    """

    root: Path = PROJECT_ROOT
    random_state: int = RANDOM_STATE
    cv_folds: int = CV_FOLDS

    min_price_usd: float = 500.0
    max_price_usd: float = 150_000.0
    min_year: int = 1950
    max_year: int = 2022
    max_odometer_miles: float = 500_000.0
    max_car_age: int = 100
    mock_sample_rows: int = 5_000

    raw_csv: Path = field(init=False)
    mock_parquet: Path = field(init=False)
    clean_parquet: Path = field(init=False)
    predictions_parquet: Path = field(init=False)
    workbook_xlsx: Path = field(init=False)
    models_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Chuẩn hóa gốc và dựng đường dẫn ngay khi khởi tạo."""
        root = Path(self.root).resolve()
        if self.cv_folds < 2:
            raise ValueError("cv_folds phải >= 2")
        if self.min_price_usd <= 0 or self.max_price_usd <= self.min_price_usd:
            raise ValueError("Ngưỡng giá không hợp lệ")
        if self.min_year > self.max_year or self.max_car_age < 0:
            raise ValueError("Ngưỡng năm/tuổi xe không hợp lệ")
        if self.mock_sample_rows <= 0:
            raise ValueError("mock_sample_rows phải > 0")
        object.__setattr__(self, "root", root)
        for key, relative in {
            "raw_csv": "data/raw/vehicles.csv",
            "mock_parquet": "data/mock/mock_clean_sample.parquet",
            "clean_parquet": "data/processed/used_cars_clean_v1.parquet",
            "predictions_parquet": "outputs/predictions/predictions_test.parquet",
            "workbook_xlsx": "workbook/Project_Workbook.xlsx",
            "models_dir": "outputs/models",
            "reports_dir": "outputs/reports",
        }.items():
            object.__setattr__(self, key, root / relative)

    def ensure_output_directories(self) -> None:
        """Tạo thư mục đầu ra cục bộ; không đụng dữ liệu gốc."""
        for path in (
            self.mock_parquet.parent,
            self.clean_parquet.parent,
            self.predictions_parquet.parent,
            self.workbook_xlsx.parent,
            self.models_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


CONFIG: Final[ProjectConfig] = ProjectConfig()
