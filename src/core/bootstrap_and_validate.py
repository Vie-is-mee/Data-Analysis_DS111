"""Tạo dữ liệu mồi đúng hợp đồng và tự kiểm tra dữ liệu bàn giao DS111.

Chạy từ gốc repository (PowerShell, Python 3.11)::

    python -m src.core.bootstrap_and_validate bootstrap
    python -m src.core.bootstrap_and_validate validate --parquet data/mock/mock_clean_sample.parquet --formula-store data/mock/mock_formula_store.joblib

Dữ liệu giả lập chỉ phục vụ phát triển, thử giao diện và kiểm thử mã nguồn;
không được đưa các chỉ số chạy trên dữ liệu giả lập vào kết luận khoa học.
Bộ chia train/val/test ở đây chỉ dành cho dữ liệu mồi. Khi xử lý dữ liệu thật,
TV3 phải tạo danh sách chia ổn định, tránh dùng kết quả do TV1 tự chia.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

from configs.project_config import (
    CONDITION_ORDINAL,
    CONFIG,
    RANDOM_STATE,
    REQUIRED_CLEAN_SCHEMA,
    TITLE_STATUS_ORDINAL_EXPERIMENTAL,
)
from src.core.formula_store import FormulaMetadata, FormulaStore

# Các trường ngoài hợp đồng được cung cấp để thử nghiệm nhanh Bước 3.
# listing_id là khóa chính theo DATA_CONTRACT.md; row_id chỉ là bí danh.
MOCK_EXTRA_COLUMNS: tuple[str, ...] = ("row_id", "condition_ord", "title_status_ord")
VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})
STRING_COLUMNS: tuple[str, ...] = tuple(
    name for name, dtype in REQUIRED_CLEAN_SCHEMA.items() if dtype == "string"
)
NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    name for name, dtype in REQUIRED_CLEAN_SCHEMA.items() if dtype != "string"
)
RAW_USECOLS: frozenset[str] = frozenset(
    {
        "id",
        "price",
        "year",
        "odometer",
        "manufacturer",
        "model",
        "condition",
        "title_status",
        "cylinders",
        "fuel",
        "transmission",
        "drive",
        "type",
        "state",
        "region",
        "posting_date",
    }
)
MANUFACTURER_ALIASES: dict[str, str] = {
    "chevy": "chevrolet",
    "chev": "chevrolet",
    "vw": "volkswagen",
    "mercedes": "mercedes-benz",
    "mercedes benz": "mercedes-benz",
    "mercedes-benz": "mercedes-benz",
    "bmw": "bmw",
    "toyota": "toyota",
    "honda": "honda",
    "ford": "ford",
    "chevrolet": "chevrolet",
    "audi": "audi",
}
CONDITION_MULTIPLIER: dict[str, float] = {
    "salvage": 0.60,
    "fair": 0.78,
    "good": 1.00,
    "excellent": 1.10,
    "like new": 1.19,
    "new": 1.27,
}
BRAND_PARAMS: dict[str, tuple[float, float, tuple[str, ...]]] = {
    # (giá xe mới tham chiếu USD, tốc độ khấu hao/năm, danh sách dòng xe)
    "toyota": (36_000.0, 0.068, ("corolla", "camry", "rav4", "highlander")),
    "honda": (34_000.0, 0.074, ("civic", "accord", "cr-v", "pilot")),
    "ford": (42_000.0, 0.110, ("f-150", "escape", "explorer", "focus")),
    "chevrolet": (39_000.0, 0.115, ("silverado", "malibu", "equinox", "impala")),
    "bmw": (75_000.0, 0.155, ("320i", "530i", "x3", "x5")),
    "mercedes-benz": (82_000.0, 0.159, ("c300", "e350", "glc", "gle")),
    "audi": (72_000.0, 0.148, ("a4", "a6", "q5", "q7")),
}


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    """Một mục kiểm duyệt và hướng sửa tương ứng nếu chưa đạt."""

    name: str
    passed: bool
    detail: str
    remediation: str = ""


@dataclass(slots=True)
class ValidationReport:
    """Báo cáo đầy đủ của một lần kiểm tra, dùng được từ Python hoặc CLI."""

    parquet_path: Path
    checks: list[ValidationCheck] = field(default_factory=list)
    row_count: int | None = None

    @property
    def passed(self) -> bool:
        """Trạng thái đạt chung: tất cả hạng mục bắt buộc đều đạt."""
        return bool(self.checks) and all(item.passed for item in self.checks)

    @property
    def failed_count(self) -> int:
        """Số tiêu chí chưa đạt."""
        return sum(not item.passed for item in self.checks)

    def add(self, name: str, passed: bool, detail: str, remediation: str = "") -> None:
        """Ghi một tiêu chí; không làm mất tiêu chí thất bại trước đó."""
        self.checks.append(ValidationCheck(name, bool(passed), detail, remediation))

    def print_console(self) -> None:
        """In kết quả ASCII/Unicode dễ đọc trên Windows PowerShell."""
        line = "=" * 88
        print(
            f"\n{line}\nDS111 - KIỂM TRA BÀN GIAO DỮ LIỆU\nTệp: {self.parquet_path}\n{line}"
        )
        for index, item in enumerate(self.checks, start=1):
            status = "PASS" if item.passed else "FAIL"
            print(f"{index:02d}. [{status}] {item.name}: {item.detail}")
            if not item.passed and item.remediation:
                print(f"    Cách sửa: {item.remediation}")
        print("-" * 88)
        print(
            f"Tổng: {len(self.checks)} mục; {self.failed_count} lỗi; "
            f"số dòng: {self.row_count if self.row_count is not None else 'không đọc được'}"
        )
        print(
            "[PASS] READY FOR HOT-SWAP"
            if self.passed
            else "[FAIL] NOT READY FOR HOT-SWAP"
        )
        print(line)


def _normalize_text(values: pd.Series, *, default: str = "unknown") -> pd.Series:
    """Chuẩn hóa chuỗi về chữ thường và lấp giá trị trống bằng nhãn tường minh."""
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.replace("", pd.NA).fillna(default).astype("string")


def _make_year_era(years: pd.Series) -> pd.Series:
    """Phân nhóm đời xe đúng bốn nhãn trong hợp đồng."""
    bins = [-np.inf, 1999, 2010, 2016, np.inf]
    labels = ["Classic", "Old", "Mid", "Modern"]
    return pd.cut(years, bins=bins, labels=labels).astype("string")


def _synthetic_rows(n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sinh bản ghi giả lập theo hàm suy giảm giá có độ dốc riêng cho từng hãng.

    Hàm sử dụng quãng đường tính theo đơn vị 100.000 dặm; riêng
    ``log_odometer`` của hợp đồng vẫn là log(1 + odometer) tính theo dặm.
    Chỉ giữ mẫu có giá nằm trong miền thật, không ép giá bằng thao tác clip.
    """
    collected: list[pd.DataFrame] = []
    remaining = n_samples
    brand_names = list(BRAND_PARAMS)
    brand_weights = np.array([0.24, 0.20, 0.21, 0.15, 0.08, 0.07, 0.05])
    conditions = list(CONDITION_MULTIPLIER)
    condition_weights = [0.025, 0.13, 0.48, 0.235, 0.105, 0.025]
    titles = list(TITLE_STATUS_ORDINAL_EXPERIMENTAL)
    title_weights = [0.012, 0.018, 0.055, 0.045, 0.020, 0.850]
    states = ["ca", "tx", "fl", "ny", "wa", "il", "az", "co", "ga", "nc"]
    state_regions = {
        "ca": "los angeles",
        "tx": "dallas",
        "fl": "miami",
        "ny": "new york",
        "wa": "seattle",
        "il": "chicago",
        "az": "phoenix",
        "co": "denver",
        "ga": "atlanta",
        "nc": "charlotte",
    }
    attempts = 0
    while remaining > 0:
        attempts += 1
        if attempts > 50:
            raise RuntimeError("Không đủ mẫu giả lập hợp lệ sau 50 vòng sinh")
        batch_size = max(int(remaining * 1.7), 500)
        brands = rng.choice(brand_names, size=batch_size, p=brand_weights)
        years = rng.integers(1990, 2022, size=batch_size, dtype=np.int32)
        ages = 2021 - years
        mileage_per_year = np.maximum(
            rng.normal(10_000.0, 2_800.0, batch_size), 2_000.0
        )
        odometers = np.maximum(
            0.0, ages * mileage_per_year + rng.normal(1000, 3000, batch_size)
        )
        odometers = np.minimum(odometers, CONFIG.max_odometer_miles)
        cond = rng.choice(conditions, size=batch_size, p=condition_weights)
        title = rng.choice(titles, size=batch_size, p=title_weights)
        state = rng.choice(states, size=batch_size)
        base = np.array([BRAND_PARAMS[b][0] for b in brands], dtype=np.float64)
        slopes = np.array([BRAND_PARAMS[b][1] for b in brands], dtype=np.float64)
        condition_factor = np.array([CONDITION_MULTIPLIER[c] for c in cond])
        # P = P0 * exp(-k * age) * (1 + d/100000)^(-beta) * condition * epsilon
        # epsilon dùng nhiễu log-normal không lệch về âm trong không gian log.
        epsilon = rng.lognormal(mean=-0.02, sigma=0.19, size=batch_size)
        price = (
            base
            * np.exp(-slopes * ages)
            * np.power(1.0 + odometers / 100_000.0, -0.22)
            * condition_factor
            * epsilon
        )
        good = (price >= CONFIG.min_price_usd) & (price <= CONFIG.max_price_usd)
        if not np.any(good):
            continue
        take = np.flatnonzero(good)[:remaining]
        if take.size == 0:
            continue
        fuel = rng.choice(
            ["gas", "diesel", "hybrid", "electric"],
            size=batch_size,
            p=[0.86, 0.065, 0.055, 0.020],
        )
        transmission = rng.choice(
            ["automatic", "manual", "other"], size=batch_size, p=[0.82, 0.15, 0.03]
        )
        drive = rng.choice(["fwd", "rwd", "4wd"], size=batch_size, p=[0.44, 0.20, 0.36])
        vehicle_type = rng.choice(
            ["sedan", "suv", "truck", "pickup", "coupe", "wagon"],
            size=batch_size,
            p=[0.37, 0.31, 0.17, 0.09, 0.04, 0.02],
        )
        cylinders = rng.choice([4, 6, 8], size=batch_size, p=[0.53, 0.36, 0.11]).astype(
            float
        )
        cylinders[fuel == "electric"] = 0.0
        models = [
            BRAND_PARAMS[str(brand)][2][
                int(rng.integers(0, len(BRAND_PARAMS[str(brand)][2])))
            ]
            for brand in brands[take]
        ]
        collected.append(
            pd.DataFrame(
                {
                    "listing_id": np.arange(take.size, dtype=np.int64),
                    "price": price[take],
                    "year": years[take],
                    "posting_year": np.full(take.size, 2021, dtype=np.int32),
                    "odometer": odometers[take],
                    "manufacturer_clean": brands[take],
                    "model_clean": models,
                    "condition_clean": cond[take],
                    "title_status_clean": title[take],
                    "cylinders_num": cylinders[take],
                    "fuel": fuel[take],
                    "transmission": transmission[take],
                    "drive": drive[take],
                    "type": vehicle_type[take],
                    "state": state[take],
                    "region": [state_regions[s] for s in state[take]],
                }
            )
        )
        remaining -= len(take)
    output = pd.concat(collected, ignore_index=True)
    # Dải ID giả lập được cố định, không trùng ID thật trong Craigslist.
    output["listing_id"] = np.arange(
        9_000_000_000_000, 9_000_000_000_000 + len(output), dtype=np.int64
    )
    return output


def _select_raw_rows(
    raw_csv_path: Path, n_samples: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Lấy mẫu ngẫu nhiên đồng đều bằng ưu tiên ngẫu nhiên trong các khối CSV.

    Mỗi bản ghi hợp lệ nhận một khóa ngẫu nhiên độc lập; giữ lại những bản ghi
    có khóa nhỏ nhất. Bộ nhớ dùng cho mẫu không tăng theo kích thước CSV.
    Không thực hiện điền khuyết/ước lượng từ giá trên toàn dữ liệu trước chia tập.
    """
    if not raw_csv_path.is_file():
        raise FileNotFoundError(f"Không có dữ liệu gốc: {raw_csv_path}")
    reservoir = pd.DataFrame()
    try:
        reader = pd.read_csv(
            raw_csv_path,
            chunksize=20_000,
            usecols=lambda column: column in RAW_USECOLS,
            dtype="string",
            on_bad_lines="skip",
            encoding="utf-8",
        )
        for chunk in reader:
            required = {
                "id",
                "price",
                "year",
                "odometer",
                "posting_date",
                "condition",
                "title_status",
                "cylinders",
            }
            missing = required - set(chunk.columns)
            if missing:
                raise ValueError(f"CSV gốc thiếu cột quan trọng: {sorted(missing)}")
            data = chunk.copy()
            data["listing_id"] = pd.to_numeric(data["id"], errors="coerce")
            data["price"] = pd.to_numeric(data["price"], errors="coerce")
            data["year"] = pd.to_numeric(data["year"], errors="coerce")
            data["odometer"] = pd.to_numeric(data["odometer"], errors="coerce")
            timestamps = pd.to_datetime(data["posting_date"], utc=True, errors="coerce")
            data["posting_year"] = timestamps.dt.year
            data["condition_clean"] = _normalize_text(data["condition"])
            data["title_status_clean"] = _normalize_text(data["title_status"])
            data["cylinders_num"] = pd.to_numeric(
                data["cylinders"].astype("string").str.extract(r"(\d+)", expand=False),
                errors="coerce",
            )
            for column in RAW_USECOLS:
                if column not in data.columns:
                    data[column] = pd.Series(
                        "unknown", index=data.index, dtype="string"
                    )
            data["fuel"] = _normalize_text(data["fuel"])
            data.loc[data["fuel"] == "electric", "cylinders_num"] = 0.0
            mask = (
                data["price"].between(CONFIG.min_price_usd, CONFIG.max_price_usd)
                & data["year"].between(1990, 2021)
                & data["odometer"].between(0, CONFIG.max_odometer_miles)
                & data["posting_year"].notna()
                & data["listing_id"].between(
                    np.iinfo("int64").min, np.iinfo("int64").max
                )
                & data["listing_id"].mod(1).eq(0)
                & data["year"].mod(1).eq(0)
                & data["condition_clean"].isin(CONDITION_ORDINAL)
                & data["title_status_clean"].isin(TITLE_STATUS_ORDINAL_EXPERIMENTAL)
                & data["cylinders_num"].between(0, 16)
            ).fillna(False)
            accepted = data.loc[
                mask,
                [
                    "listing_id",
                    "price",
                    "year",
                    "posting_year",
                    "odometer",
                    "manufacturer",
                    "model",
                    "condition_clean",
                    "title_status_clean",
                    "cylinders_num",
                    "fuel",
                    "transmission",
                    "drive",
                    "type",
                    "state",
                    "region",
                ],
            ].copy()
            if accepted.empty:
                continue
            accepted["_sample_priority"] = rng.random(len(accepted))
            reservoir = (
                pd.concat([reservoir, accepted], ignore_index=True)
                .sort_values("_sample_priority")
                .drop_duplicates("listing_id")
                .head(n_samples)
            )
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Không đọc được CSV theo UTF-8: {raw_csv_path}") from exc

    if reservoir.empty:
        raise ValueError(
            "CSV không có bản ghi đáp ứng bộ lọc giá/năm/thông tin tối thiểu"
        )
    if len(reservoir) < n_samples:
        raise ValueError(
            f"CSV chỉ có {len(reservoir)} mẫu hợp lệ/không trùng ID. "
            f"Cần {n_samples}. Kiểm tra dữ liệu hoặc giảm --samples."
        )
    selected = reservoir.drop(columns="_sample_priority").reset_index(drop=True)
    selected["manufacturer_clean"] = _normalize_text(selected.pop("manufacturer"))
    selected["manufacturer_clean"] = selected["manufacturer_clean"].replace(
        MANUFACTURER_ALIASES
    )
    selected["model_clean"] = _normalize_text(selected.pop("model"))
    for column in ("transmission", "drive", "type", "state", "region"):
        selected[column] = _normalize_text(selected[column])
    # Giá trị year/posting_year thiếu đã được loại ở trên, nhưng ký hiệu
    # năm tương lai trong dữ liệu gốc được áp dụng quy tắc max(..., 0).
    return selected


def _assign_stratified_splits(price: pd.Series, rng_seed: int) -> pd.Series:
    """Chia 70/15/15 theo mười nhóm thứ hạng giá, giữ nguyên index.

    Việc chia theo giá chỉ phục vụ kiểm thử mock. Không truyền split này làm
    quyết định khoa học cho tập dữ liệu thật; TV3 quản lý phiên bản chia tập.
    """
    n_rows = len(price)
    if n_rows < 30:
        raise ValueError("Cần >= 30 dòng để chia phân tầng 70/15/15 đáng tin cậy")
    quantile_groups = min(10, max(2, n_rows // 20))
    price_ranks = price.rank(method="first")
    strata = pd.qcut(price_ranks, q=quantile_groups, labels=False)
    indices = np.arange(n_rows)
    train_idx, other_idx = train_test_split(
        indices, test_size=0.30, random_state=rng_seed, stratify=strata
    )
    remaining_strata = strata.iloc[other_idx]
    val_idx, test_idx = train_test_split(
        other_idx,
        test_size=0.50,
        random_state=rng_seed + 1,
        stratify=remaining_strata,
    )
    splits = pd.Series("", index=price.index, dtype="string")
    splits.iloc[train_idx] = "train"
    splits.iloc[val_idx] = "val"
    splits.iloc[test_idx] = "test"
    return splits


def _make_metadata(
    formula_id: str,
    category: str,
    expression: str,
    latex: str,
    inputs: list[str],
    rationale: str,
    algorithms: list[str],
) -> FormulaMetadata:
    """Dựng siêu dữ liệu hoàn chỉnh dùng cho công thức tạo mẫu."""
    return FormulaMetadata(
        formula_id=formula_id,
        feature_name=formula_id,
        step_category=category,
        expression_plain=expression,
        expression_latex=latex,
        input_columns=inputs,
        output_series_key=formula_id,
        theoretical_rationale=rationale,
        suitable_algorithms=algorithms,
    )


def _finalize_sample(prepared: pd.DataFrame) -> tuple[pd.DataFrame, FormulaStore]:
    """Tạo đủ cột hợp đồng, đăng ký công thức, ép kiểu và sắp xếp cột."""
    data = prepared.reset_index(drop=True).copy()
    if data.empty:
        raise ValueError("Tập dữ liệu chuẩn bị đang rỗng")
    numeric = (
        "listing_id",
        "price",
        "year",
        "posting_year",
        "odometer",
        "cylinders_num",
    )
    for name in numeric:
        if name not in data:
            raise KeyError(f"Thiếu cột số đầu vào: {name}")
        data[name] = pd.to_numeric(data[name], errors="raise")
    data["listing_id"] = data["listing_id"].astype("int64")
    data["price"] = data["price"].astype("float64")
    data["year"] = data["year"].astype("int32")
    data["posting_year"] = data["posting_year"].astype("int32")
    data["odometer"] = data["odometer"].astype("float64")
    data["cylinders_num"] = data["cylinders_num"].astype("float64")
    for name in STRING_COLUMNS:
        if name == "year_era":
            continue
        if name not in data and name != "split_set":
            raise KeyError(f"Thiếu cột chuỗi đầu vào: {name}")
        if name in data:
            data[name] = _normalize_text(data[name])

    store = FormulaStore()
    transformations: tuple[
        tuple[FormulaMetadata, Callable[[pd.DataFrame], pd.Series]], ...
    ] = (
        (
            _make_metadata(
                "car_age",
                "Step3_Creation",
                "max(posting_year - year, 0)",
                r"\max(\mathrm{posting\_year}-\mathrm{year},0)",
                ["posting_year", "year"],
                "Độ tuổi xe tính tại thời điểm đăng tin, không dùng năm chạy chương trình.",
                ["OLS", "Ridge", "DecisionTree", "RandomForest", "LightGBM"],
            ),
            lambda df: (df["posting_year"] - df["year"]).clip(lower=0).astype("int32"),
        ),
        (
            _make_metadata(
                "log_price",
                "Step3_Linearization",
                "ln(price)",
                r"\ln(P)",
                ["price"],
                "Biến đổi giá theo log giúp nghiên cứu khấu hao mũ; không dùng log_price làm biến dự báo đầu vào.",
                ["OLS", "Ridge", "DecisionTree", "RandomForest", "LightGBM"],
            ),
            lambda df: np.log(df["price"].astype("float64")),
        ),
        (
            _make_metadata(
                "log_odometer",
                "Step3_Linearization",
                "ln(1 + odometer)",
                r"\ln(1+\mathrm{odometer})",
                ["odometer"],
                "Biến số dặm lệch phải thành biến có dạng thuận tiện cho mô hình tuyến tính.",
                ["OLS", "Ridge", "DecisionTree", "RandomForest", "LightGBM"],
            ),
            lambda df: np.log1p(df["odometer"].astype("float64")),
        ),
        (
            _make_metadata(
                "miles_per_year",
                "Step3_Creation",
                "odometer / max(car_age, 1)",
                r"\frac{\mathrm{odometer}}{\max(\mathrm{car\_age},1)}",
                ["odometer", "car_age"],
                "Số dặm trung bình theo năm hoạt động; tránh chia cho 0 khi xe mới.",
                ["OLS", "Ridge", "DecisionTree", "RandomForest", "LightGBM"],
            ),
            lambda df: df["odometer"] / df["car_age"].clip(lower=1),
        ),
        (
            _make_metadata(
                "condition_ord",
                "Step4_Encoding",
                "condition_clean → {0,...,5}",
                r"\mathrm{Ordinal}(\mathrm{condition})\in\{0,1,2,3,4,5\}",
                ["condition_clean"],
                "Tình trạng xe có thứ bậc chất lượng; unknown không tự động coi là một thứ bậc.",
                ["OLS", "Ridge", "DecisionTree", "RandomForest", "LightGBM"],
            ),
            lambda df: df["condition_clean"].map(CONDITION_ORDINAL).astype("int8"),
        ),
        (
            _make_metadata(
                "title_status_ord",
                "Step4_Encoding",
                "title_status_clean → thứ bậc giả thuyết {0,...,5}",
                r"\mathrm{Ordinal}_{\mathrm{exp}}(\mathrm{title\_status})",
                ["title_status_clean"],
                "CHỈ LÀ GIẢ THUYẾT: trạng thái pháp lý không có thứ tự toàn phần; phải đối chứng One-Hot.",
                ["OLS", "Ridge", "DecisionTree", "RandomForest", "LightGBM"],
            ),
            lambda df: (
                df["title_status_clean"]
                .map(TITLE_STATUS_ORDINAL_EXPERIMENTAL)
                .astype("int8")
            ),
        ),
    )
    for metadata, transform in transformations:
        output = store.apply_and_register(data, transform, metadata)
        data[metadata.output_series_key] = output

    data["year_era"] = _make_year_era(data["year"])
    data["row_id"] = data["listing_id"].astype("int64")
    data["split_set"] = _assign_stratified_splits(data["price"], RANDOM_STATE)
    for name, dtype in REQUIRED_CLEAN_SCHEMA.items():
        if name not in data:
            raise AssertionError(f"Lỗi nội bộ: không tạo được {name}")
        data[name] = data[name].astype(dtype)
    data["condition_ord"] = data["condition_ord"].astype("int8")
    data["title_status_ord"] = data["title_status_ord"].astype("int8")
    final_columns = list(REQUIRED_CLEAN_SCHEMA) + list(MOCK_EXTRA_COLUMNS)
    return data.loc[:, final_columns], store


def generate_mock_clean_sample(
    raw_csv_path: Path | None,
    output_path: Path,
    n_samples: int = 5000,
    use_synthetic_if_missing: bool = True,
) -> pd.DataFrame:
    """Sinh mẫu 5.000 dòng từ CSV hoặc giả lập nếu CSV chưa được tải.

    Ngoài Parquet, ghi ``mock_formula_store.joblib`` trong cùng thư mục.
    Trả về DataFrame để TV1/TV2/TV3 có thể kiểm tra thêm mà không cần đọc lại.
    Nếu file CSV tồn tại nhưng sai định dạng, báo lỗi rõ thay vì âm thầm giả lập.
    """
    if n_samples < 30:
        raise ValueError("n_samples phải >= 30 để chia phân tầng đủ ba tập")
    path = Path(raw_csv_path) if raw_csv_path is not None else CONFIG.raw_csv
    output = Path(output_path)
    rng = np.random.default_rng(RANDOM_STATE)
    if path.is_file():
        print(f"Đang lấy mẫu thực từ CSV theo khối: {path}")
        original = _select_raw_rows(path, n_samples, rng)
        source = "DỮ LIỆU GỐC – mẫu thử, chưa phải tập sạch chính thức"
    else:
        if not use_synthetic_if_missing:
            raise FileNotFoundError(
                f"Thiếu {path}. Tải vehicles.csv vào data/raw hoặc bỏ --no-synthetic."
            )
        print(f"Chưa có {path}. Tạo dữ liệu giả lập theo quy luật khấu hao xe cũ.")
        original = _synthetic_rows(n_samples, rng)
        source = "GIẢ LẬP – KHÔNG DÙNG ĐỂ BÁO CÁO KẾT QUẢ KHOA HỌC"
    sample, store = _finalize_sample(original)
    if len(sample) != n_samples:
        raise RuntimeError(f"Số dòng sau tạo mẫu {len(sample)} khác {n_samples}")
    output.parent.mkdir(parents=True, exist_ok=True)
    formula_path = output.with_name("mock_formula_store.joblib")
    table = pa.Table.from_pandas(sample, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"ds111.mock_source"] = b"csv_sample" if path.is_file() else b"synthetic"
    pq.write_table(table.replace_schema_metadata(metadata), output)
    store.save(formula_path)
    print(
        f"[{source}]\nĐã ghi: {output} ({len(sample):,} dòng)\n"
        f"FormulaStore: {formula_path} ({store.formula_count} công thức)"
    )
    return sample


def _data_type_is_compatible(arrow_type: pa.DataType, expected_dtype: str) -> bool:
    """Đối chiếu đúng độ rộng và nhóm kiểu của hợp đồng."""
    if expected_dtype == "string":
        return pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type)
    return arrow_type == pa.type_for_alias(expected_dtype)


def validate_handoff_dataset(
    parquet_path: Path,
    formula_store_path: Path | None = None,
) -> ValidationReport:
    """Kiểm tra dữ liệu dựa trên cấu hình/hợp đồng GitHub và in báo cáo.

    Không sửa dữ liệu trong lúc kiểm định. Các cột phụ chỉ kiểm tra khi có mặt,
    bởi hợp đồng chính không yêu cầu ``row_id`` hoặc hai cột mã hóa thứ bậc.
    ``FormulaStore`` phải có ít nhất ba công thức nếu được cung cấp.
    """
    path = Path(parquet_path)
    report = ValidationReport(parquet_path=path)
    if not path.is_file():
        report.add(
            "Tệp Parquet",
            False,
            "Không tìm thấy tệp",
            "Kiểm tra đường dẫn hoặc chạy bootstrap trước.",
        )
        report.print_console()
        return report
    report.add("Tệp Parquet", True, "Đã tồn tại")
    try:
        parquet = pq.ParquetFile(path)
        arrow_schema = parquet.schema_arrow
        rows = parquet.metadata.num_rows
        report.row_count = rows
        report.add("Đọc bằng PyArrow", True, "Đọc được phần thông tin tệp Parquet")
    except (OSError, ValueError, pa.ArrowException) as exc:
        report.add(
            "Đọc bằng PyArrow",
            False,
            f"Không đọc được: {exc}",
            "Xuất lại Parquet bằng pyarrow; không đổi tên CSV thành .parquet.",
        )
        report.print_console()
        return report
    report.add(
        "Số dòng > 200",
        rows > 200,
        f"{rows:,} dòng",
        "Bàn giao dữ liệu ít nhất 201 dòng hợp lệ.",
    )
    missing = sorted(set(REQUIRED_CLEAN_SCHEMA) - set(arrow_schema.names))
    report.add(
        "Đầy đủ 22 cột theo hợp đồng",
        not missing,
        "Đủ cột" if not missing else f"Thiếu {missing}",
        "Xuất lại các cột với tên chính xác trong configs/project_config.py.",
    )
    type_errors = [
        f"{column}: {arrow_schema.field(column).type} (cần {dtype})"
        for column, dtype in REQUIRED_CLEAN_SCHEMA.items()
        if column in arrow_schema.names
        and not _data_type_is_compatible(arrow_schema.field(column).type, dtype)
    ]
    report.add(
        "Kiểu dữ liệu Parquet",
        not type_errors,
        "Đúng kiểu theo hợp đồng" if not type_errors else "; ".join(type_errors[:8]),
        "Ép kiểu số với pd.to_numeric và chuỗi với astype('string') trước khi xuất.",
    )
    if missing or type_errors:
        report.print_console()
        return report
    try:
        data = parquet.read().to_pandas()
        report.add("Đọc toàn bộ dữ liệu", True, "Có thể đọc nội dung các dòng")
    except (OSError, ValueError, pa.ArrowException) as exc:
        report.add(
            "Đọc toàn bộ dữ liệu",
            False,
            str(exc),
            "Sửa tệp Parquet hỏng, sau đó ghi lại bằng engine='pyarrow'.",
        )
        report.print_console()
        return report

    pandas_type_errors = {
        column: str(data[column].dtype)
        for column, expected in REQUIRED_CLEAN_SCHEMA.items()
        if str(data[column].dtype) != expected
    }
    report.add(
        "Kiểu dữ liệu pandas",
        not pandas_type_errors,
        str(pandas_type_errors) if pandas_type_errors else "Đúng kiểu theo hợp đồng",
        "Ép các cột theo REQUIRED_CLEAN_SCHEMA trước khi xuất.",
    )
    null_counts = data[list(REQUIRED_CLEAN_SCHEMA)].isna().sum()
    bad_nulls = null_counts[null_counts > 0].to_dict()
    report.add(
        "Không có NaN/giá trị khuyết",
        not bad_nulls,
        str(bad_nulls) if bad_nulls else "0 giá trị khuyết",
        "Bổ sung quy tắc điền khuyết có cơ sở hoặc loại dòng có giải trình.",
    )
    if pandas_type_errors or bad_nulls:
        report.print_console()
        return report

    def has(*names: str) -> bool:
        """Kiểm tra đủ cột trước khi thực hiện phép đo."""
        return all(name in data.columns for name in names)

    if has("listing_id"):
        key = data["listing_id"]
        valid_key = key.notna().all() and key.is_unique and is_numeric_dtype(key.dtype)
        report.add(
            "listing_id không trùng",
            valid_key,
            f"trùng: {int(key.duplicated().sum())}, rỗng: {int(key.isna().sum())}",
            "Giữ khóa tin đăng duy nhất hoặc dựng khóa ổn định có tài liệu.",
        )
    else:
        report.add(
            "listing_id không trùng", False, "Thiếu listing_id", "Bổ sung khóa chính."
        )
    if has("row_id", "listing_id"):
        equal_ids = data["row_id"].equals(data["listing_id"])
        report.add(
            "Bí danh row_id (nếu có)",
            equal_ids,
            "Khớp listing_id" if equal_ids else "row_id khác listing_id",
            "Đồng bộ row_id = listing_id hoặc bỏ bí danh này.",
        )
    if has("split_set"):
        actual = set(data["split_set"].dropna().astype(str).unique())
        complete = actual == VALID_SPLITS and data["split_set"].notna().all()
        counts = data["split_set"].value_counts().to_dict()
        report.add(
            "Phân chia train/val/test",
            complete,
            f"Các tập: {counts}",
            "TV3 chia cố định đúng ba tập, không để split_set rỗng hay nhãn lạ.",
        )
    else:
        report.add(
            "Phân chia train/val/test",
            False,
            "Thiếu split_set",
            "TV3 bổ sung split_set từ danh sách chia cố định.",
        )

    infinities: dict[str, int] = {}
    for col in NUMERIC_COLUMNS + ("condition_ord", "title_status_ord"):
        if col in data and is_numeric_dtype(data[col].dtype):
            n_inf = int(
                np.isinf(data[col].to_numpy(dtype="float64", na_value=np.nan)).sum()
            )
            if n_inf:
                infinities[col] = n_inf
    report.add(
        "Không có +inf/-inf",
        not infinities,
        "0 giá trị vô cực" if not infinities else str(infinities),
        "Kiểm tra phép chia cho 0 và phép log trên giá trị không hợp lệ.",
    )

    if has("price") and is_numeric_dtype(data["price"].dtype):
        p = data["price"]
        valid = bool(p.between(CONFIG.min_price_usd, CONFIG.max_price_usd).all())
        report.add(
            "Miền giá USD",
            valid,
            f"min={p.min()}, max={p.max()}",
            f"Lọc giá từ {CONFIG.min_price_usd} đến {CONFIG.max_price_usd}, ghi ngoại lệ vào Workbook.",
        )
    else:
        report.add(
            "Miền giá USD", False, "Thiếu cột số price", "Ép kiểu price thành float64."
        )
    if has("year", "posting_year", "car_age") and all(
        is_numeric_dtype(data[col].dtype) for col in ("year", "posting_year", "car_age")
    ):
        y, py, age = data["year"], data["posting_year"], data["car_age"]
        valid = bool(
            y.between(CONFIG.min_year, CONFIG.max_year).all()
            and age.between(0, CONFIG.max_car_age).all()
            and (np.maximum(py.astype("int64") - y, 0) == age).all()
        )
        report.add(
            "Năm sản xuất và tuổi xe",
            valid,
            f"year: [{y.min()}, {y.max()}], tuổi: [{age.min()}, {age.max()}]",
            "car_age phải bằng max(posting_year-year, 0), year trong phạm vi cấu hình.",
        )
    else:
        report.add(
            "Năm sản xuất và tuổi xe",
            False,
            "Thiếu cột năm/tuổi hợp lệ",
            "Bổ sung year, posting_year, car_age dạng số.",
        )
    if has("odometer") and is_numeric_dtype(data["odometer"].dtype):
        d = data["odometer"]
        valid = bool(d.between(0, CONFIG.max_odometer_miles).all())
        report.add(
            "Miền số dặm",
            valid,
            f"min={d.min()}, max={d.max()}",
            "Lọc/giải trình số dặm âm hoặc quá 500.000.",
        )
    else:
        report.add(
            "Miền số dặm",
            False,
            "Thiếu cột số odometer",
            "Ép kiểu odometer thành float64.",
        )
    if has("cylinders_num") and is_numeric_dtype(data["cylinders_num"].dtype):
        cylinders = data["cylinders_num"]
        valid_cylinders = bool(cylinders.ge(0).all())
        report.add(
            "Số xi-lanh không âm",
            valid_cylinders,
            f"min={cylinders.min()}, max={cylinders.max()}",
            "Chuẩn hóa cylinders_num dạng float64 không âm; xe điện có thể bằng 0 khi có cơ sở.",
        )
    else:
        report.add(
            "Số xi-lanh không âm",
            False,
            "Thiếu cylinders_num dạng số",
            "Bổ sung cylinders_num từ dữ liệu gốc, có quy tắc xử lý thiếu rõ ràng.",
        )
    for target, original, transform, label in (
        ("log_price", "price", np.log, "log_price = ln(price)"),
        ("log_odometer", "odometer", np.log1p, "log_odometer = ln(1+odometer)"),
    ):
        if has(target, original) and all(
            is_numeric_dtype(data[c].dtype) for c in (target, original)
        ):
            with np.errstate(divide="ignore", invalid="ignore"):
                expected = transform(data[original].to_numpy(dtype="float64"))
            actual = data[target].to_numpy(dtype="float64")
            valid = bool(
                np.allclose(actual, expected, rtol=0, atol=1e-8, equal_nan=False)
            )
            report.add(
                label,
                valid,
                "Sai số tối đa <=1e-8" if valid else "Giá trị log không nhất quán",
                f"Tính lại {target} đúng từ {original} bằng NumPy.",
            )
        else:
            report.add(
                label,
                False,
                "Thiếu cột số để đối chiếu",
                f"Tạo hai cột {target} và {original} dạng số.",
            )
    if has("miles_per_year", "odometer", "car_age") and all(
        is_numeric_dtype(data[c].dtype)
        for c in ("miles_per_year", "odometer", "car_age")
    ):
        expected = data["odometer"].to_numpy(dtype="float64") / np.maximum(
            data["car_age"].to_numpy(dtype="float64"), 1
        )
        valid = bool(
            np.allclose(
                data["miles_per_year"].to_numpy(dtype="float64"),
                expected,
                rtol=0,
                atol=1e-4,
                equal_nan=False,
            )
        )
        report.add(
            "miles_per_year",
            valid,
            "Công thức đúng" if valid else "Sai công thức",
            "Tính odometer / max(car_age, 1).",
        )
    else:
        report.add(
            "miles_per_year",
            False,
            "Thiếu cột dữ liệu",
            "Bổ sung miles_per_year theo hợp đồng.",
        )

    if has("condition_clean"):
        valid_conditions = set(CONDITION_ORDINAL) | {"unknown"}
        invalid_conditions = (
            set(data["condition_clean"].dropna().astype(str)) - valid_conditions
        )
        report.add(
            "Nhãn condition_clean",
            not invalid_conditions,
            "Nhãn hợp lệ" if not invalid_conditions else str(invalid_conditions),
            "Đưa tình trạng về các nhãn chuẩn; unknown giữ riêng.",
        )
    if has("title_status_clean"):
        valid_title = set(TITLE_STATUS_ORDINAL_EXPERIMENTAL) | {"unknown"}
        invalid_titles = (
            set(data["title_status_clean"].dropna().astype(str)) - valid_title
        )
        report.add(
            "Nhãn title_status_clean",
            not invalid_titles,
            "Nhãn hợp lệ" if not invalid_titles else str(invalid_titles),
            "Chuẩn hóa trạng thái pháp lý theo hợp đồng.",
        )
    if has("condition_ord"):
        ord_values = set(data["condition_ord"].dropna().tolist())
        valid = ord_values <= set(range(6)) and data["condition_ord"].notna().all()
        if has("condition_clean") and valid:
            expected = data["condition_clean"].map(CONDITION_ORDINAL)
            valid = bool((data["condition_ord"] == expected).all())
        report.add(
            "Mã condition_ord (nếu có)",
            valid,
            f"Các mã: {sorted(map(str, ord_values))}",
            "Chỉ mã hóa nhãn đã biết thành 0..5; unknown không được gán bừa.",
        )
    if has("title_status_ord"):
        ord_values = set(data["title_status_ord"].dropna().tolist())
        valid = ord_values <= set(range(6)) and data["title_status_ord"].notna().all()
        if has("title_status_clean") and valid:
            expected = data["title_status_clean"].map(TITLE_STATUS_ORDINAL_EXPERIMENTAL)
            valid = bool((data["title_status_ord"] == expected).all())
        report.add(
            "Mã title_status_ord (nếu có)",
            valid,
            f"Các mã: {sorted(map(str, ord_values))}",
            "Sửa ánh xạ thực nghiệm; không suy ra đây là thứ bậc tự nhiên.",
        )
    if has("manufacturer_clean"):
        values = data["manufacturer_clean"].astype("string")
        good = bool(
            values.notna().all()
            and values.eq(values.str.strip().str.lower()).all()
            and values.str.len().gt(0).all()
        )
        report.add(
            "Chuẩn hóa hãng xe",
            good,
            "Toàn chữ thường, không khoảng trắng thừa"
            if good
            else "Hãng còn tên sai chuẩn",
            "Dùng .astype('string').str.strip().str.lower() và gộp bí danh.",
        )
    for name in (
        "model_clean",
        "fuel",
        "transmission",
        "drive",
        "type",
        "state",
        "region",
    ):
        if name in data:
            values = data[name].astype("string")
            good = bool(
                values.notna().all()
                and values.eq(values.str.strip().str.lower()).all()
                and values.str.len().gt(0).all()
            )
            report.add(
                f"Chuẩn hóa {name}",
                good,
                "Đúng" if good else "Có chuỗi thiếu/chưa chuẩn hóa",
                f"Dùng .str.strip().str.lower() cho {name}, gán nhãn unknown hợp lệ.",
            )
    if has("year", "year_era"):
        expected = _make_year_era(data["year"])
        matched = bool(data["year_era"].astype("string").eq(expected).all())
        report.add(
            "Nhóm đời xe",
            matched,
            "Đúng bốn nhóm" if matched else "Nhãn không khớp năm sản xuất",
            "Dùng Classic(<2000), Old(2000–2010), Mid(2011–2016), Modern(>=2017).",
        )

    if formula_store_path is not None:
        store_path = Path(formula_store_path)
        if not store_path.is_file():
            report.add(
                "FormulaStore tồn tại",
                False,
                str(store_path),
                "Chạy bootstrap hoặc lưu FormulaStore sau bước tạo đặc trưng.",
            )
        else:
            try:
                # Chỉ nạp file joblib tin cậy do thành viên nhóm tạo.
                store = FormulaStore.load(store_path)
                report.add("Nạp FormulaStore", True, f"Đã nạp {store_path}")
                enough = store.formula_count >= 3
                report.add(
                    "Ít nhất 3 công thức",
                    enough,
                    f"Có {store.formula_count} công thức",
                    "Đăng ký bổ sung các bước biến đổi thực tế vào FormulaStore.",
                )
                keys = set(store.series_keys)
                absent = [
                    meta.output_series_key
                    for meta in store.list_metadata()
                    if meta.output_series_key not in keys
                ]
                valid = not absent and len(keys) == store.formula_count
                report.add(
                    "Đồng bộ metadata/Series",
                    valid,
                    "Mọi khóa tồn tại" if valid else f"Khóa thiếu: {absent}",
                    "Dùng register_formula để đăng ký đồng thời hai vùng nhớ.",
                )
                incompatible = [
                    key
                    for key in store.series_keys
                    if len(store.get_series(key)) != len(data)
                ]
                report.add(
                    "Số dòng FormulaStore",
                    not incompatible,
                    "Khớp dữ liệu"
                    if not incompatible
                    else f"Sai kích thước: {incompatible}",
                    "Lưu FormulaStore từ đúng phiên bản Parquet được bàn giao.",
                )
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                ImportError,
                EOFError,
                pickle.UnpicklingError,
            ) as exc:
                report.add(
                    "Nạp FormulaStore",
                    False,
                    f"Không hợp lệ: {exc}",
                    "Tạo lại joblib bằng FormulaStore.save (chỉ nạp nguồn tin cậy).",
                )
    report.print_console()
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Định nghĩa hai lệnh con và các tham số đường dẫn, cỡ mẫu."""
    parser = argparse.ArgumentParser(
        description="DS111: sinh dữ liệu mồi và tự kiểm tra dữ liệu bàn giao."
    )
    actions = parser.add_subparsers(dest="command", required=True)
    bootstrap = actions.add_parser(
        "bootstrap", help="Sinh mock từ CSV hoặc dữ liệu giả lập"
    )
    bootstrap.add_argument(
        "--raw-csv",
        type=Path,
        default=CONFIG.raw_csv,
        help="CSV gốc; mặc định data/raw/vehicles.csv",
    )
    bootstrap.add_argument(
        "--output",
        type=Path,
        default=CONFIG.mock_parquet,
        help="Đường dẫn Parquet dữ liệu mồi",
    )
    bootstrap.add_argument(
        "--samples",
        type=int,
        default=CONFIG.mock_sample_rows,
        help="Số mẫu (mặc định 5000)",
    )
    bootstrap.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Báo lỗi nếu thiếu CSV thay vì sinh giả lập",
    )
    validate = actions.add_parser(
        "validate", help="Kiểm tra Parquet trước khi bàn giao"
    )
    validate.add_argument(
        "--parquet",
        "--path",
        type=Path,
        default=CONFIG.mock_parquet,
        help="Parquet cần kiểm tra; có thể trỏ đến data/processed/",
    )
    validate.add_argument(
        "--formula-store",
        "--store",
        type=Path,
        default=None,
        help="Joblib công thức kèm theo (chỉ nạp tệp tin cậy)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy ứng dụng; trả 0 khi thành công và 1 khi có lỗi thực thi/kiểm định."""
    args = _parse_args(argv)
    try:
        if args.command == "bootstrap":
            generate_mock_clean_sample(
                raw_csv_path=args.raw_csv,
                output_path=args.output,
                n_samples=args.samples,
                use_synthetic_if_missing=not args.no_synthetic,
            )
            return 0
        if args.command == "validate":
            report = validate_handoff_dataset(
                parquet_path=args.parquet, formula_store_path=args.formula_store
            )
            return 0 if report.passed else 1
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
