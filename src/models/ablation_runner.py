"""Huấn luyện Exp0–Exp4 trên train, chọn bằng CV và mở test sau khi chốt."""

from dataclasses import asdict
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from configs.project_config import (
    CONFIG,
    HIGH_CARDINALITY_CATEGORICAL,
    LOW_CARDINALITY_CATEGORICAL,
    REQUIRED_CLEAN_SCHEMA,
    ProjectConfig,
)
from src.models.domain_features import (
    DOMAIN_HINGE_FEATURES,
    DOMAIN_TOKEN_FEATURES,
    FLAT_OOF_FEATURES,
    WEAR_ADJ_HIER_FEATURES,
    WearAdjustedHierarchicalEncoder,
    add_domain_depreciation_features,
)
from src.models.metrics import (
    compute_duan_smearing_factor,
    evaluate_regression,
    fit_binned_duan_smearing,
    inverse_log_transform,
    inverse_log_transform_binned,
)
from src.models.model_registry import ModelEntry, ModelRegistry
from src.models.preprocessing_bridge import (
    CylinderModeImputer,
    build_feature_preprocessor,
)
from src.models.split_manifest import (
    GROUP_COLUMN,
    assign_production_splits,
    validate_split_manifest,
)

LINEARIZED_NUMERIC = ("car_age", "log_odometer", "miles_per_year", "cylinders_num")
LINEARIZED_NUMERIC_QUADRATIC = (*LINEARIZED_NUMERIC, "car_age_squared")
_RAW_FEATURES = ("year", "odometer")
_CV_FIELDS = (
    "r2",
    "adj_r2",
    "ss_res",
    "ss_tot",
    "y_mean_baseline",
    "mse",
    "rmse",
    "mae",
    "mape_pct",
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


def _as_usd(predictions, factor, binned_smearing=None):
    # OLS giá gốc có thể ngoại suy dưới 0. Áp sàn cố định $1 cho mọi tập.
    if binned_smearing is not None:
        return inverse_log_transform_binned(predictions, binned_smearing)
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
        course_encoding: bool = False,
    ) -> None:
        self.config = config
        self.fast_mode = fast_mode
        self.dataset_version = dataset_version
        self.raw_exp0_features = raw_exp0_features
        self._course_encoding = course_encoding
        self._train_groups = None
        self.registry = ModelRegistry()
        self._final_model_id: str | None = None
        if (
            not data["listing_id"].is_unique
            or data["listing_id"].isna().any()
            or not data["split_set"].isin(["train", "val", "test"]).all()
        ):
            raise ValueError("Cần listing_id duy nhất và split_set hợp lệ.")
        self._data = add_domain_depreciation_features(
            data.sort_values("listing_id").set_index("listing_id", drop=False)
        )
        self._splits = {
            name: self._data.loc[self._data["split_set"].eq(name)]
            for name in ("train", "val", "test")
        }
        if len(self._splits["train"]) < max(config.cv_folds, 4) or any(
            len(self._splits[name]) < 2 for name in ("val", "test")
        ):
            raise ValueError("Thiếu số dòng theo split hoặc số fold CV.")
        if GROUP_COLUMN in self._data:
            manifest = validate_split_manifest(self._data, self._data["listing_id"])
            groups = manifest.loc[self._splits["train"].index, GROUP_COLUMN]
            if groups.nunique() < config.cv_folds:
                raise ValueError("Số nhóm xe trong train nhỏ hơn số fold.")
            if not course_encoding:
                raise ValueError("CV theo nhóm yêu cầu mã hóa không dùng nhãn giá.")
            self._train_groups = groups
            self._cv = GroupKFold(n_splits=config.cv_folds)
        else:
            self._cv = KFold(
                config.cv_folds, shuffle=True, random_state=config.random_state
            )

    @classmethod
    def from_parquet(
        cls,
        path: Path,
        *,
        mock: bool,
        fast_mode: bool,
        config: ProjectConfig,
        split_manifest: Path | None = None,
        raw_csv: Path | None = None,
        dataset_version: str | None = None,
    ) -> "AblationExperimentSuite":
        """Đọc Parquet đã qua validator và áp dụng cách chia của TV3."""
        columns = list(REQUIRED_CLEAN_SCHEMA)
        if "cylinders_was_missing" in pq.read_schema(path).names:
            columns.append("cylinders_was_missing")
        data = pd.read_parquet(path, engine="pyarrow", columns=columns)
        raw = None
        if mock:
            if split_manifest is not None:
                raise ValueError("Manifest cố định chỉ dùng với dữ liệu production.")
            source = (pq.read_metadata(path).metadata or {}).get(b"ds111.mock_source")
            version = {
                b"synthetic": "mock-synthetic",
                b"csv_sample": "mock-csv-sample",
            }.get(source, "mock-unspecified")
        else:
            requires_vin_groups = bool(data["posting_year"].ge(2026).any())
            if split_manifest is None:
                if requires_vin_groups:
                    raise ValueError("Dữ liệu từ 2026 cần manifest nhóm xe đã khóa.")
                data["split_set"] = assign_production_splits(
                    data["listing_id"], config.random_state
                )
            else:
                manifest = validate_split_manifest(
                    pd.read_parquet(split_manifest, engine="pyarrow"),
                    data["listing_id"],
                )
                if requires_vin_groups and GROUP_COLUMN not in manifest:
                    raise ValueError("Manifest từ 2026 phải có vehicle_group_id.")
                if requires_vin_groups and (
                    "cylinders_was_missing" not in data
                    or not pd.api.types.is_bool_dtype(data["cylinders_was_missing"])
                    or data["cylinders_was_missing"].isna().any()
                ):
                    raise ValueError("ABT từ 2026 cần cờ cylinders_was_missing hợp lệ.")
                if (
                    data["split_set"].astype("string").tolist()
                    != manifest["split_set"].tolist()
                ):
                    raise ValueError(
                        "split_set trong ABT chưa khớp manifest đã khóa theo listing_id."
                    )
                if GROUP_COLUMN in manifest:
                    data[GROUP_COLUMN] = manifest[GROUP_COLUMN].to_numpy()
            raw = read_raw_exp0_features(raw_csv or config.raw_csv, data["listing_id"])
            version = dataset_version or "v1.0"
        return cls(
            data,
            fast_mode,
            config=config,
            dataset_version=version,
            raw_exp0_features=raw,
            course_encoding=not mock,
        )

    @property
    def split_assignments(self) -> pd.DataFrame:
        """Danh sách chia tập theo ID để đối chiếu với dữ liệu TV1 và TV2."""
        columns = ["listing_id", "split_set"]
        if GROUP_COLUMN in self._data:
            columns.append(GROUP_COLUMN)
        return (
            self._data[columns]
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

    def _cross_validate(
        self,
        prototype,
        X,
        target,
        price,
        log_target,
        *,
        use_domain_groups: bool = False,
        use_binned_duan: bool = False,
    ):
        usd_reports, log_reports = [], []
        oof = np.empty(len(X))
        for inner_train, holdout in self._cv.split(X, groups=self._train_groups):
            X_fold = X.iloc[inner_train]
            y_fold = target[inner_train]
            fit_kwargs = (
                {"domain_encode__groups": self._train_groups.iloc[inner_train]}
                if (use_domain_groups and self._train_groups is not None)
                else {}
            )
            estimator = clone(prototype).fit(X_fold, y_fold, **fit_kwargs)
            predictions = estimator.predict(X.iloc[holdout])
            oof[holdout] = predictions
            train_fold_pred = estimator.predict(X_fold) if log_target else None
            binned = (
                fit_binned_duan_smearing(y_fold, train_fold_pred)
                if (log_target and use_binned_duan)
                else None
            )
            factor = (
                binned.global_factor
                if binned is not None
                else (
                    compute_duan_smearing_factor(y_fold, train_fold_pred)
                    if log_target
                    else None
                )
            )
            count = len(estimator["preprocess"].get_feature_names_out())
            usd_reports.append(
                asdict(
                    evaluate_regression(
                        price[holdout],
                        _as_usd(predictions, factor, binned),
                        count,
                        factor,
                    )
                )
            )
            if log_target:
                log_reports.append(
                    asdict(evaluate_regression(target[holdout], predictions, count))
                )
            del estimator, predictions, train_fold_pred
            gc.collect()
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
        domain_encoder_mode: str | None = None,
        use_binned_duan: bool = False,
        encoding_label: str | None = None,
    ) -> ModelEntry:
        if self._final_model_id is not None:
            raise RuntimeError(
                "Đã chốt test. Kết thúc lần chạy trước khi huấn luyện thêm."
            )
        model_id = f"{stage.split('_', 1)[0]}_{variant or name}"
        if model_id in self.registry.list_ids():
            raise ValueError(f"Mô hình đã được chạy: {model_id}")
        encoded_cols = (
            FLAT_OOF_FEATURES
            if domain_encoder_mode == "flat_oof"
            else (
                WEAR_ADJ_HIER_FEATURES
                if domain_encoder_mode == "wear_adjusted_hierarchical"
                else ()
            )
        )
        columns = [*numeric, *ordinal, *low, *high]
        if domain_encoder_mode is not None:
            columns = list(
                dict.fromkeys([*columns, *WearAdjustedHierarchicalEncoder.required_columns])
            )
        refit_cylinders = (
            "cylinders_num" in numeric and "cylinders_was_missing" in self._data
        )
        if refit_cylinders:
            columns = list(dict.fromkeys([*columns, *CylinderModeImputer.columns]))
        train, val = self._splits["train"], self._splits["val"]
        X = self._features("train", columns, stage)
        X_val = self._features("val", columns, stage)
        target_column = "log_price" if log_target else "price"
        target = train[target_column].to_numpy()
        preprocessor = build_feature_preprocessor(
            strategy,
            [*numeric, *encoded_cols],
            ordinal,
            low,
            high,
            random_state=self.config.random_state,
            cv_folds=self.config.cv_folds,
        ).set_output(transform="pandas")
        steps = [("cylinders_mode", CylinderModeImputer())] if refit_cylinders else []
        if domain_encoder_mode is not None:
            steps.append(
                (
                    "domain_encode",
                    WearAdjustedHierarchicalEncoder(
                        mode=domain_encoder_mode,
                        n_splits=self.config.cv_folds,
                        random_state=self.config.random_state,
                    ),
                )
            )
        steps.append(("preprocess", preprocessor))
        if isinstance(estimator, LGBMRegressor):
            steps.append(("lightgbm_columns", FunctionTransformer(_lightgbm_frame)))
        steps.append(("regression", estimator))
        pipeline = Pipeline(steps)
        use_domain_groups = domain_encoder_mode is not None
        start = perf_counter()
        with threadpool_limits(limits=1, user_api="blas"):
            means, deviations, oof = self._cross_validate(
                pipeline,
                X,
                target,
                train["price"].to_numpy(),
                log_target,
                use_domain_groups=use_domain_groups,
                use_binned_duan=use_binned_duan,
            )
            binned_smearing = (
                fit_binned_duan_smearing(target, oof)
                if (log_target and use_binned_duan)
                else None
            )
            factor = (
                binned_smearing.global_factor
                if binned_smearing is not None
                else (compute_duan_smearing_factor(target, oof) if log_target else None)
            )
            fit_kwargs = (
                {"domain_encode__groups": self._train_groups}
                if (use_domain_groups and self._train_groups is not None)
                else {}
            )
            pipeline.fit(X, target, **fit_kwargs)
        elapsed = perf_counter() - start
        n_features = len(pipeline["preprocess"].get_feature_names_out())
        train_prediction = pipeline.predict(X)
        train_metrics = evaluate_regression(
            train["price"].to_numpy(),
            _as_usd(train_prediction, factor, binned_smearing),
            n_features,
            factor,
        )
        train_log = (
            evaluate_regression(
                train["log_price"].to_numpy(), train_prediction, n_features
            )
            if log_target
            else None
        )
        prediction = pipeline.predict(X_val)
        val_metrics = evaluate_regression(
            val["price"].to_numpy(),
            _as_usd(prediction, factor, binned_smearing),
            n_features,
            factor,
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
            model_family=(
                "Traditional" if name.startswith(("OLS", "Ridge")) else "Tree_Ensemble"
            ),
            target_mode="log_price" if log_target else "raw_price",
            encoding_strategy=encoding_label or strategy,
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
            train_metrics=train_metrics,
            train_log_metrics=train_log,
            binned_smearing=binned_smearing,
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
            LINEARIZED_NUMERIC_QUADRATIC,
            variant="OLS_AgeSquared",
            notes="Đối chứng bổ sung car_age², giữ mô hình tuyến tính theo hệ số.",
        )
        return primary, quadratic

    def _nominal_columns(self):
        if self._course_encoding:
            return (*LOW_CARDINALITY_CATEGORICAL, *HIGH_CARDINALITY_CATEGORICAL), ()
        return LOW_CARDINALITY_CATEGORICAL, HIGH_CARDINALITY_CATEGORICAL

    def run_exp3_encoding_comparison(self):
        ordinal = ("condition_clean",)
        low, high = self._nominal_columns()
        nominal = (*low, "title_status_clean")
        naive = self._fit(
            "Exp3A_Naive_Label",
            "OLS",
            LinearRegression(),
            LINEARIZED_NUMERIC_QUADRATIC,
            strategy="naive_label",
            ordinal=ordinal,
            low=nominal,
            high=high,
            notes="Mã số từ thứ tự chữ cái (giữ cùng cụm biến số kèm car_age² làm đối chứng).",
        )
        principled = self._fit(
            "Exp3B_Principled_Encoding",
            "OLS",
            LinearRegression(),
            LINEARIZED_NUMERIC_QUADRATIC,
            strategy="principled_linear",
            ordinal=ordinal,
            low=nominal,
            high=high,
            notes="Title dùng One-Hot, Condition có thứ bậc + cờ unknown, giữ car_age².",
        )
        title = self._fit(
            "Exp3B_Principled_Encoding",
            "OLS",
            LinearRegression(),
            LINEARIZED_NUMERIC_QUADRATIC,
            strategy="principled_linear",
            ordinal=(*ordinal, "title_status_clean"),
            low=low,
            high=high,
            variant="OLS_TitleOrdinal",
            notes="Chỉ thay One-Hot của title bằng thứ bậc giả thuyết để đối chứng (giữ car_age²).",
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
        low, high = self._nominal_columns()
        for name, estimator in candidates:
            is_linear = name in ("OLS", "Ridge")
            strategy = "principled_linear" if is_linear else "principled_tree"
            numeric = LINEARIZED_NUMERIC_QUADRATIC if is_linear else LINEARIZED_NUMERIC
            note = (
                "Tuyến tính kèm car_age² trên cùng cột nguồn và các fold train của benchmark."
                if is_linear
                else "Dùng cùng cột nguồn và các fold train của benchmark."
            )
            entries.append(
                self._fit(
                    "Exp4_Benchmark",
                    name,
                    estimator,
                    numeric,
                    strategy=strategy,
                    ordinal=("condition_clean",),
                    low=(*low, "title_status_clean"),
                    high=high,
                    notes=note,
                )
            )
        return entries

    def run_exp5_domain_innovation(self):
        """Đối chứng 2×2: Target Encoding phẳng vs Phần dư hao mòn phân cấp (Group-OOF + Binned Duan)."""
        seed = self.config.random_state
        jobs = 1 if self.fast_mode else -1
        low, high = self._nominal_columns()
        nominal = (*low, "title_status_clean")
        lin_numeric = (
            *LINEARIZED_NUMERIC_QUADRATIC,
            *DOMAIN_HINGE_FEATURES,
            *DOMAIN_TOKEN_FEATURES,
        )
        tree_numeric = (
            *LINEARIZED_NUMERIC,
            *DOMAIN_HINGE_FEATURES,
            *DOMAIN_TOKEN_FEATURES,
        )
        lgb_base = dict(
            n_estimators=40 if self.fast_mode else 300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=10,
            min_child_samples=30,
            n_jobs=jobs,
            random_state=seed,
            verbosity=-1,
        )
        lgb_huber = dict(
            objective="huber",
            alpha=0.8,
            n_estimators=50 if self.fast_mode else 500,
            learning_rate=0.05,
            num_leaves=31 if self.fast_mode else 63,
            max_depth=12,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            n_jobs=jobs,
            random_state=seed,
            verbosity=-1,
        )
        specs = (
            (
                "Ridge_FlatOOF",
                Ridge(alpha=10.0),
                lin_numeric,
                "principled_linear",
                "flat_oof",
                False,
                "flat_oof_model_region",
                "Đối chứng: Ridge (α=10) + khấu hao gãy khúc/cờ phiên bản + Target Encoding phẳng (model, region).",
            ),
            (
                "Ridge_WearAdjHier",
                Ridge(alpha=10.0),
                lin_numeric,
                "principled_linear",
                "wear_adjusted_hierarchical",
                True,
                "wear_adj_hierarchical",
                "Đề xuất TV3: Ridge (α=10) + phần dư hao mòn phân cấp 5 tầng theo đời xe (Group-OOF) + Binned Duan.",
            ),
            (
                "LightGBM_FlatOOF",
                LGBMRegressor(**lgb_base),
                tree_numeric,
                "principled_tree",
                "flat_oof",
                False,
                "flat_oof_model_region",
                "Đối chứng: LightGBM mặc định (31/300) + Target Encoding phẳng (model, region).",
            ),
            (
                "LightGBM_WearAdjHier",
                LGBMRegressor(**lgb_base),
                tree_numeric,
                "principled_tree",
                "wear_adjusted_hierarchical",
                True,
                "wear_adj_hierarchical",
                "Đề xuất TV3 (cùng tham số 31/300 để tách thuần hiệu quả đặc trưng): phân cấp 5 tầng theo đời xe + Binned Duan.",
            ),
            (
                "LightGBM_Huber_Reg",
                LGBMRegressor(**lgb_huber),
                tree_numeric,
                "principled_tree",
                "wear_adjusted_hierarchical",
                True,
                "wear_adj_hierarchical",
                "Đề xuất TV3 (Huber bền ngoại lai + điều chuẩn Bài 05): chống lệch từ giá ảo trong train, phân cấp 5 tầng + Binned Duan.",
            ),
        )
        return [
            self._fit(
                "Exp5_Domain_Innovation",
                name,
                estimator,
                numeric,
                strategy=strategy,
                ordinal=("condition_clean",),
                low=nominal,
                high=high,
                notes=note,
                domain_encoder_mode=enc_mode,
                use_binned_duan=binned,
                encoding_label=enc_label,
            )
            for name, estimator, numeric, strategy, enc_mode, binned, enc_label, note in specs
        ]

    def run_all(self, *, include_exp5: bool = True) -> ModelRegistry:
        """Chạy toàn bộ chuỗi và trả kết quả CV/validation."""
        self.run_exp0_raw_minimal_ols()
        self.run_exp1_cleaned_age_ols()
        self.run_exp2_linearized_log_price_ols()
        self.run_exp3_encoding_comparison()
        self.run_exp4_multimodel_benchmark()
        if include_exp5:
            self.run_exp5_domain_innovation()
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
        usd = _as_usd(prediction, factor, winner.binned_smearing)
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
