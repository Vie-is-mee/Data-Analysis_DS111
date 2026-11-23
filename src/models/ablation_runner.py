"""Huấn luyện Exp0–Exp4 trên train, chọn bằng CV và mở test sau khi chốt."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from configs.project_config import (
    CONFIG,
    HIGH_CARDINALITY_CATEGORICAL,
    LOW_CARDINALITY_CATEGORICAL,
    RANDOM_STATE,
    REQUIRED_CLEAN_SCHEMA,
    ProjectConfig,
)
from src.models.metrics import (
    compute_duan_smearing_factor,
    evaluate_regression,
    inverse_log_transform,
)
from src.models.model_registry import ModelEntry, ModelRegistry
from src.models.preprocessing_bridge import build_feature_preprocessor

LINEARIZED_NUMERIC = ("car_age", "log_odometer", "miles_per_year", "cylinders_num")
_RAW_FEATURES = ("year", "odometer")
_CV_FIELDS = (
    "r2",
    "adj_r2",
    "ss_res",
    "ss_tot",
    "y_mean_baseline",
    "rmse",
    "mae",
    "mape_pct",
)


def assign_production_splits(
    listing_ids: pd.Series, random_state: int = RANDOM_STATE
) -> pd.Series:
    """Chia 70/15/15 theo ID duy nhất, độc lập với nhãn và thứ tự dòng."""
    ids = pd.Index(listing_ids, name="listing_id")
    if ids.dtype != np.dtype("int64") or not ids.is_unique or len(ids) < 10:
        raise ValueError("Split cần ít nhất 10 listing_id kiểu int64 và duy nhất.")
    train, other = train_test_split(
        ids.sort_values(), test_size=0.30, random_state=random_state
    )
    val, test = train_test_split(other, test_size=0.50, random_state=random_state + 1)
    splits = pd.Series(index=ids, dtype="string")
    for name, keys in (("train", train), ("val", val), ("test", test)):
        splits.loc[keys] = name
    return pd.Series(
        splits.to_numpy(), index=listing_ids.index, dtype="string", name="split_set"
    )


def read_raw_exp0_features(
    csv_path: Path, listing_ids: pd.Series
) -> pd.DataFrame | None:
    """Ghép hai biến CSV gốc theo khóa. Dùng proxy nếu thiếu hoặc mơ hồ ID."""
    if not csv_path.is_file():
        return None
    columns = ["id", *_RAW_FEATURES]
    matches = []
    with pd.read_csv(
        csv_path, usecols=columns, dtype={"id": "string"}, chunksize=20_000
    ) as reader:
        for chunk in reader:
            keys = pd.to_numeric(chunk["id"], errors="coerce")
            matched = keys.isin(listing_ids)
            features = chunk.loc[matched, list(_RAW_FEATURES)].apply(
                pd.to_numeric, errors="coerce"
            )
            features.index = pd.Index(keys[matched].astype("int64"), name="listing_id")
            matches.append(features.replace([np.inf, -np.inf], np.nan))
    if not matches:
        return None
    raw = pd.concat(matches).reset_index().drop_duplicates().set_index("listing_id")
    if not raw.index.is_unique or not pd.Index(listing_ids).difference(raw.index).empty:
        return None
    return raw


def _as_usd(predictions, factor):
    # OLS giá gốc có thể ngoại suy dưới 0. Áp sàn cố định $1 cho mọi tập.
    return (
        inverse_log_transform(predictions, factor)
        if factor is not None
        else np.maximum(predictions, 1.0)
    )


def _lightgbm_frame(frame: pd.DataFrame) -> pd.DataFrame:
    # LightGBM giới hạn ký tự trong tên cột; nhãn One-Hot có thể chứa ký tự đó.
    return frame.set_axis([f"feature_{i}" for i in range(frame.shape[1])], axis=1)


class AblationExperimentSuite:
    """Điều phối đối chứng và giữ test ngoài mọi phương thức huấn luyện."""

    def __init__(
        self,
        data: pd.DataFrame,
        fast_mode: bool = False,
        *,
        config: ProjectConfig = CONFIG,
        dataset_version: str = "mock-unspecified",
        raw_exp0_features: pd.DataFrame | None = None,
    ) -> None:
        self.config = config
        self.fast_mode = fast_mode
        self.dataset_version = dataset_version
        self.raw_exp0_features = raw_exp0_features
        self.registry = ModelRegistry()
        self._final_model_id: str | None = None
        if (
            not data["listing_id"].is_unique
            or data["listing_id"].isna().any()
            or not data["split_set"].isin(["train", "val", "test"]).all()
        ):
            raise ValueError("Cần listing_id duy nhất và split_set hợp lệ.")
        self._data = data.sort_values("listing_id").set_index("listing_id", drop=False)
        self._data["car_age_squared"] = self._data["car_age"].astype(float).pow(2)
        self._splits = {
            name: self._data.loc[self._data["split_set"].eq(name)]
            for name in ("train", "val", "test")
        }
        if len(self._splits["train"]) < max(config.cv_folds, 4) or any(
            len(self._splits[name]) < 2 for name in ("val", "test")
        ):
            raise ValueError("Thiếu số dòng theo split hoặc số fold CV.")
        self._cv = KFold(
            config.cv_folds, shuffle=True, random_state=config.random_state
        )

    @classmethod
    def from_parquet(
        cls, path: Path, *, mock: bool, fast_mode: bool, config: ProjectConfig
    ) -> "AblationExperimentSuite":
        """Đọc Parquet đã qua validator và áp dụng cách chia của TV3."""
        data = pd.read_parquet(
            path, engine="pyarrow", columns=list(REQUIRED_CLEAN_SCHEMA)
        )
        raw = None
        if mock:
            source = (pq.read_metadata(path).metadata or {}).get(b"ds111.mock_source")
            version = {
                b"synthetic": "mock-synthetic",
                b"csv_sample": "mock-csv-sample",
            }.get(source, "mock-unspecified")
        else:
            data["split_set"] = assign_production_splits(
                data["listing_id"], config.random_state
            )
            raw = read_raw_exp0_features(config.raw_csv, data["listing_id"])
            version = "v1.0"
        return cls(
            data,
            fast_mode,
            config=config,
            dataset_version=version,
            raw_exp0_features=raw,
        )

    @property
    def split_assignments(self) -> pd.DataFrame:
        """Danh sách chia tập theo ID để đối chiếu với dữ liệu TV1 và TV2."""
        return (
            self._data[["listing_id", "split_set"]]
            .reset_index(drop=True)
            .assign(dataset_version=self.dataset_version)
        )

    @property
    def test_data(self) -> pd.DataFrame:
        """Các dòng test đã được mở cho báo cáo cuối."""
        if self._final_model_id is None:
            raise RuntimeError("Cần chốt mô hình trước khi đọc test.")
        return self._splits["test"].copy()

    @property
    def train_data(self) -> pd.DataFrame:
        """Trả về bản sao train cho báo cáo VIF và ranking."""
        return self._splits["train"].copy()

    def _features(self, partition, columns, experiment_stage):
        frame = self._splits[partition]
        if (
            experiment_stage == "Exp0_Raw_Minimal"
            and self.raw_exp0_features is not None
        ):
            return self.raw_exp0_features.loc[frame.index, list(columns)]
        return frame.loc[:, list(columns)]

    def _cross_validate(self, prototype, X, target, price, log_target):
        usd_reports, log_reports = [], []
        oof = np.empty(len(X))
        for inner_train, holdout in self._cv.split(X):
            X_fold = X.iloc[inner_train]
            y_fold = target[inner_train]
            estimator = clone(prototype).fit(X_fold, y_fold)
            predictions = estimator.predict(X.iloc[holdout])
            oof[holdout] = predictions
            factor = (
                compute_duan_smearing_factor(y_fold, estimator.predict(X_fold))
                if log_target
                else None
            )
            count = len(estimator["preprocess"].get_feature_names_out())
            usd_reports.append(
                asdict(
                    evaluate_regression(
                        price[holdout], _as_usd(predictions, factor), count, factor
                    )
                )
            )
            if log_target:
                log_reports.append(
                    asdict(evaluate_regression(target[holdout], predictions, count))
                )
        usd = pd.DataFrame(usd_reports).loc[:, list(_CV_FIELDS)]
        if log_reports:
            log = pd.DataFrame(log_reports).loc[:, list(_CV_FIELDS)].add_prefix("log_")
            usd = pd.concat([usd, log], axis=1)
        # Chỉ công bố thang đo CV khi nó xác định trên tất cả fold.
        complete = usd.loc[:, usd.notna().all(axis=0)]
        return complete.mean().to_dict(), complete.std(ddof=0).to_dict(), oof

    def _fit(
        self,
        stage,
        name,
        estimator,
        numeric,
        *,
        strategy="numeric_only",
        log_target=True,
        ordinal=(),
        low=(),
        high=(),
        notes="",
        variant=None,
    ) -> ModelEntry:
        if self._final_model_id is not None:
            raise RuntimeError(
                "Đã chốt test. Kết thúc lần chạy trước khi huấn luyện thêm."
            )
        model_id = f"{stage.split('_', 1)[0]}_{variant or name}"
        if model_id in self.registry.list_ids():
            raise ValueError(f"Mô hình đã được chạy: {model_id}")
        columns = [*numeric, *ordinal, *low, *high]
        train, val = self._splits["train"], self._splits["val"]
        X = self._features("train", columns, stage)
        X_val = self._features("val", columns, stage)
        target_column = "log_price" if log_target else "price"
        target = train[target_column].to_numpy()
        preprocessor = build_feature_preprocessor(
            strategy,
            numeric,
            ordinal,
            low,
            high,
            random_state=self.config.random_state,
            cv_folds=self.config.cv_folds,
        ).set_output(transform="pandas")
        steps = [("preprocess", preprocessor)]
        if isinstance(estimator, LGBMRegressor):
            steps.append(("lightgbm_columns", FunctionTransformer(_lightgbm_frame)))
        steps.append(("regression", estimator))
        pipeline = Pipeline(steps)
        start = perf_counter()
        with threadpool_limits(limits=1, user_api="blas"):
            means, deviations, oof = self._cross_validate(
                pipeline, X, target, train["price"].to_numpy(), log_target
            )
            factor = compute_duan_smearing_factor(target, oof) if log_target else None
            pipeline.fit(X, target)
        elapsed = perf_counter() - start
        n_features = len(pipeline["preprocess"].get_feature_names_out())
        prediction = pipeline.predict(X_val)
        val_metrics = evaluate_regression(
            val["price"].to_numpy(), _as_usd(prediction, factor), n_features, factor
        )
        val_log = (
            evaluate_regression(val["log_price"].to_numpy(), prediction, n_features)
            if log_target
            else None
        )
        entry = ModelEntry(
            model_id=model_id,
            experiment_stage=stage,
            model_name=name,
            model_family="Traditional" if name in ("OLS", "Ridge") else "Tree_Ensemble",
            target_mode="log_price" if log_target else "raw_price",
            encoding_strategy=strategy,
            feature_names=columns,
            hyperparams=estimator.get_params(),
            cv_metrics_mean=means,
            cv_metrics_std=deviations,
            val_metrics=val_metrics,
            train_time_sec=elapsed,
            notes=notes,
            fitted_pipeline=pipeline,
            val_log_metrics=val_log,
            dataset_version=self.dataset_version,
            random_state=self.config.random_state,
        )
        self.registry.register(entry)
        return entry

    def run_exp0_raw_minimal_ols(self):
        notes = (
            "Biến year/odometer gốc theo listing_id. Dùng cùng giá tham chiếu và cùng mẫu sạch."
            if self.raw_exp0_features is not None
            else "Proxy raw-minimal. Dùng year/odometer từ mẫu đã lọc."
        )
        return self._fit(
            "Exp0_Raw_Minimal",
            "OLS",
            LinearRegression(),
            _RAW_FEATURES,
            strategy="raw_minimal",
            log_target=False,
            notes=notes,
        )

    def run_exp1_cleaned_age_ols(self):
        return self._fit(
            "Exp1_Cleaned_Age",
            "OLS",
            LinearRegression(),
            ("car_age", "odometer"),
            log_target=False,
            notes="Dùng tuổi tại thời điểm đăng tin và số dặm sạch.",
        )

    def run_exp2_linearized_log_price_ols(self):
        primary = self._fit(
            "Exp2_Linearized",
            "OLS",
            LinearRegression(),
            LINEARIZED_NUMERIC,
            notes="Tuổi và log(1+số dặm), dự báo log(giá).",
        )
        quadratic = self._fit(
            "Exp2_Linearized",
            "OLS",
            LinearRegression(),
            (*LINEARIZED_NUMERIC, "car_age_squared"),
            variant="OLS_AgeSquared",
            notes="Đối chứng bổ sung car_age², giữ mô hình tuyến tính theo hệ số.",
        )
        return primary, quadratic

    def run_exp3_encoding_comparison(self):
        ordinal = ("condition_clean",)
        nominal = (*LOW_CARDINALITY_CATEGORICAL, "title_status_clean")
        high = HIGH_CARDINALITY_CATEGORICAL
        naive = self._fit(
            "Exp3A_Naive_Label",
            "OLS",
            LinearRegression(),
            LINEARIZED_NUMERIC,
            strategy="naive_label",
            ordinal=ordinal,
            low=nominal,
            high=high,
            notes="Mã số từ thứ tự chữ cái, chỉ dùng làm đối chứng.",
        )
        principled = self._fit(
            "Exp3B_Principled_Encoding",
            "OLS",
            LinearRegression(),
            LINEARIZED_NUMERIC,
            strategy="principled_linear",
            ordinal=ordinal,
            low=nominal,
            high=high,
            notes="Title dùng One-Hot. Condition có thứ bậc và cờ unknown.",
        )
        title = self._fit(
            "Exp3B_Principled_Encoding",
            "OLS",
            LinearRegression(),
            LINEARIZED_NUMERIC,
            strategy="principled_linear",
            ordinal=(*ordinal, "title_status_clean"),
            low=LOW_CARDINALITY_CATEGORICAL,
            high=high,
            variant="OLS_TitleOrdinal",
            notes="Chỉ thay One-Hot của title bằng thứ bậc giả thuyết để đối chứng.",
        )
        return naive, principled, title

    def run_exp4_multimodel_benchmark(self):
        seed = self.config.random_state
        jobs = 1 if self.fast_mode else -1
        candidates = (
            ("OLS", LinearRegression()),
            ("Ridge", Ridge(alpha=1.0)),
            (
                "DecisionTree",
                DecisionTreeRegressor(
                    max_depth=12, min_samples_leaf=20, random_state=seed
                ),
            ),
            (
                "RandomForest",
                RandomForestRegressor(
                    n_estimators=12 if self.fast_mode else 80,
                    max_depth=14,
                    min_samples_leaf=4,
                    n_jobs=jobs,
                    random_state=seed,
                ),
            ),
            (
                "LightGBM",
                LGBMRegressor(
                    n_estimators=40 if self.fast_mode else 300,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=10,
                    min_child_samples=30,
                    n_jobs=jobs,
                    random_state=seed,
                    verbosity=-1,
                ),
            ),
        )
        entries = []
        for name, estimator in candidates:
            strategy = (
                "principled_linear" if name in ("OLS", "Ridge") else "principled_tree"
            )
            entries.append(
                self._fit(
                    "Exp4_Benchmark",
                    name,
                    estimator,
                    LINEARIZED_NUMERIC,
                    strategy=strategy,
                    ordinal=("condition_clean",),
                    low=(*LOW_CARDINALITY_CATEGORICAL, "title_status_clean"),
                    high=HIGH_CARDINALITY_CATEGORICAL,
                    notes="Dùng cùng cột nguồn và các fold train của benchmark.",
                )
            )
        return entries

    def run_all(self) -> ModelRegistry:
        """Chạy toàn bộ chuỗi và trả kết quả CV/validation."""
        self.run_exp0_raw_minimal_ols()
        self.run_exp1_cleaned_age_ols()
        self.run_exp2_linearized_log_price_ols()
        self.run_exp3_encoding_comparison()
        self.run_exp4_multimodel_benchmark()
        return self.registry

    def evaluate_final(self) -> ModelEntry:
        """Chốt đúng một mô hình bằng CV USD, sau đó mới dự báo trên test."""
        if self._final_model_id is not None:
            return self.registry.get(self._final_model_id)
        winner = self.registry.get_best_model()
        test = self._splits["test"]
        X = self._features("test", winner.feature_names, winner.experiment_stage)
        prediction = winner.fitted_pipeline.predict(X)
        factor = winner.val_metrics.smearing_factor
        count = winner.val_metrics.n_features
        usd = _as_usd(prediction, factor)
        winner.test_metrics = evaluate_regression(
            test["price"].to_numpy(), usd, count, factor
        )
        if winner.target_mode == "log_price":
            winner.test_log_metrics = evaluate_regression(
                test["log_price"].to_numpy(), prediction, count
            )
        winner.y_pred_test = pd.Series(
            usd, index=test.index, name="predicted_price_usd"
        )
        self._final_model_id = winner.model_id
        return winner
