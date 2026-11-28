"""M1-07 · Bước 3 · Đặc trưng → Workbook Sheet 04 (qua FormulaStore).

Mọi đặc trưng được tính bằng ``FormulaStore.apply_and_register`` để lưu
công thức + lý do + thống kê trước/sau tách khỏi Series.

QUAN TRỌNG: chạy SAU khi đã lọc xong dòng. FormulaStore yêu cầu mọi Series
dùng chung một index; lọc dòng sau bước này sẽ làm store lệch với Parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.formula_store import FormulaMetadata, FormulaStore

STEP = "03_feature_engineering"
LINEAR = ["OLS", "Ridge"]
ALL_MODELS = ["OLS", "Ridge", "DecisionTree", "RandomForest", "LightGBM"]


def make_year_era(years: pd.Series) -> pd.Series:
    """Classic < 2000 ≤ Old ≤ 2010 < Mid ≤ 2016 < Modern (khớp bộ kiểm định)."""
    bins = [-np.inf, 1999, 2010, 2016, np.inf]
    return pd.cut(years, bins=bins, labels=["Classic", "Old", "Mid", "Modern"]).astype("string")


FEATURES: tuple[tuple[FormulaMetadata, callable], ...] = (
    (
        FormulaMetadata(
            formula_id="car_age", feature_name="car_age", step_category=STEP,
            expression_plain="car_age = max(posting_year - year, 0)",
            expression_latex=r"\text{car\_age}=\max(\text{posting\_year}-\text{year},0)",
            input_columns=["year", "posting_year"], output_series_key="car_age",
            theoretical_rationale="Khấu hao theo thời gian: P = P0·e^(-k·car_age)",
            suitable_algorithms=ALL_MODELS,
        ),
        lambda d: (d["posting_year"] - d["year"]).clip(lower=0).astype("int32"),
    ),
    (
        FormulaMetadata(
            formula_id="log_price", feature_name="log_price", step_category=STEP,
            expression_plain="log_price = ln(price)",
            expression_latex=r"\ln(\text{price})",
            input_columns=["price"], output_series_key="log_price",
            theoretical_rationale="Giảm lệch phải; biến mô hình nhân thành cộng tính",
            suitable_algorithms=LINEAR,
        ),
        lambda d: np.log(d["price"].astype("float64")),
    ),
    (
        FormulaMetadata(
            formula_id="log_odometer", feature_name="log_odometer", step_category=STEP,
            expression_plain="log_odometer = ln(1 + odometer)",
            expression_latex=r"\ln(1+\text{odometer})",
            input_columns=["odometer"], output_series_key="log_odometer",
            theoretical_rationale="Hao mòn biên giảm dần theo số dặm: (1+odometer)^(-β)",
            suitable_algorithms=LINEAR,
        ),
        lambda d: np.log1p(d["odometer"].astype("float64")),
    ),
    (
        FormulaMetadata(
            formula_id="miles_per_year", feature_name="miles_per_year", step_category=STEP,
            expression_plain="miles_per_year = odometer / max(car_age, 1)",
            expression_latex=r"\frac{\text{odometer}}{\max(\text{car\_age},1)}",
            input_columns=["odometer", "car_age"], output_series_key="miles_per_year",
            theoretical_rationale="Cường độ sử dụng, tách khỏi tuổi xe",
            suitable_algorithms=ALL_MODELS,
        ),
        lambda d: d["odometer"].astype("float64") / d["car_age"].clip(lower=1).astype("float64"),
    ),
    (
        FormulaMetadata(
            formula_id="year_era", feature_name="year_era", step_category=STEP,
            expression_plain="Classic <2000, Old 2000–2010, Mid 2011–2016, Modern ≥2017",
            expression_latex=r"\text{cut}(\text{year})",
            input_columns=["year"], output_series_key="year_era",
            theoretical_rationale="Nhóm đời xe cho phân tích theo nhóm (TV2)",
            suitable_algorithms=["DecisionTree", "RandomForest", "LightGBM"],
        ),
        lambda d: make_year_era(d["year"]),
    ),
    (
        FormulaMetadata(
            formula_id="car_age_squared", feature_name="car_age_squared", step_category=STEP,
            expression_plain="car_age_squared = car_age^2",
            expression_latex=r"\text{car\_age}^2",
            input_columns=["car_age"], output_series_key="car_age_squared",
            theoretical_rationale=("Hồi quy đa thức bậc 2 (Bài 5): Sheet 06 cho |Spearman(car_age)| "
                                   "gần gấp đôi |Pearson| → quan hệ đơn điệu nhưng cong (xe cổ lại đắt)"),
            suitable_algorithms=LINEAR,
        ),
        # Cùng tên và công thức với ablation_runner của TV3 → không lệch giá trị.
        lambda d: d["car_age"].astype("float64").pow(2),
    ),
)


def run(df: pd.DataFrame, store: FormulaStore) -> pd.DataFrame:
    """Tính lần lượt từng đặc trưng (đặc trưng sau có thể dùng đặc trưng trước)."""
    out = df.copy()
    for metadata, transform in FEATURES:
        out[metadata.output_series_key] = store.apply_and_register(out, transform, metadata)
    return out
