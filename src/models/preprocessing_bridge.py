"""Các biến đổi sklearn được fit cục bộ trong từng fold huấn luyện."""

from collections.abc import Sequence
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from configs.project_config import (
    CONDITION_ORDINAL,
    CV_FOLDS,
    RANDOM_STATE,
    TITLE_STATUS_ORDINAL_EXPERIMENTAL,
)


def _categories(X) -> pd.DataFrame:
    return pd.DataFrame(X).astype("string").fillna("unknown")


class OutOfFoldTargetEncoder(TransformerMixin, BaseEstimator):
    """Làm mượt theo trung bình train, mã hóa train với ánh xạ ngoài fold."""

    def __init__(self, smoothing=10.0, n_splits=CV_FOLDS, random_state=RANDOM_STATE):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

    def _mapping(self, frame, target):
        prior = float(target.mean())
        mappings = {}
        for column in frame:
            stats = pd.Series(target).groupby(frame[column].array).agg(["count", "sum"])
            mappings[column] = (stats["sum"] + self.smoothing * prior) / (
                stats["count"] + self.smoothing
            )
        return prior, mappings

    @staticmethod
    def _encode(frame, prior, mappings):
        return np.column_stack(
            [
                frame[column].map(mapping).fillna(prior)
                for column, mapping in mappings.items()
            ]
        )

    def _training_data(self, X, y):
        frame = _categories(X)
        target = np.asarray(y, dtype=float)
        if (
            target.shape != (len(frame),)
            or not len(frame)
            or not np.isfinite(target).all()
        ):
            raise ValueError("Target Encoding cần một nhãn hữu hạn cho mỗi dòng train.")
        if not np.isfinite(self.smoothing) or self.smoothing < 0:
            raise ValueError("smoothing phải hữu hạn và không âm.")
        return frame, target

    def fit(self, X, y):
        frame, target = self._training_data(X, y)
        self.global_mean_, self.mappings_ = self._mapping(frame, target)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        frame, target = self._training_data(X, y)
        if (
            not isinstance(self.n_splits, Integral)
            or self.n_splits < 2
            or len(frame) < 2
        ):
            raise ValueError(
                "OOF Target Encoding cần ít nhất hai fold và hai dòng train."
            )
        cv = KFold(
            min(self.n_splits, len(frame)), shuffle=True, random_state=self.random_state
        )
        encoded = np.empty(frame.shape)
        for train_idx, holdout_idx in cv.split(frame):
            prior, mappings = self._mapping(frame.iloc[train_idx], target[train_idx])
            encoded[holdout_idx] = self._encode(
                frame.iloc[holdout_idx], prior, mappings
            )
        self.global_mean_, self.mappings_ = self._mapping(frame, target)
        return encoded

    def transform(self, X):
        check_is_fitted(self, "mappings_")
        return self._encode(_categories(X), self.global_mean_, self.mappings_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "mappings_")
        columns = self.mappings_ if input_features is None else input_features
        return np.asarray([f"{col}_target_oof" for col in columns], dtype=object)


class NaiveLabelEncodingTransformer(TransformerMixin, BaseEstimator):
    """Gán số theo bảng chữ cái để làm đối chứng Exp3A."""

    def fit(self, X, y=None):
        frame = _categories(X)
        self.mappings_ = {
            column: {value: i for i, value in enumerate(sorted(frame[column].unique()))}
            for column in frame
        }
        return self

    def transform(self, X):
        check_is_fitted(self, "mappings_")
        frame = _categories(X)
        return np.column_stack(
            [
                frame[column].map(mapping).fillna(-1)
                for column, mapping in self.mappings_.items()
            ]
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "mappings_")
        columns = self.mappings_ if input_features is None else input_features
        return np.asarray([f"{col}_label" for col in columns], dtype=object)


class CylinderModeImputer(TransformerMixin, BaseEstimator):
    """Khôi phục vị trí đã điền ở ABT rồi học mode từ train của fold."""

    columns = ("manufacturer_clean", "model_clean", "cylinders_was_missing")

    @staticmethod
    def _group_modes(observed, keys):
        counts = (
            observed.groupby([*keys, "cylinders_num"])["cylinders_num"]
            .size()
            .rename("observations")
            .reset_index()
        )
        return (
            counts.sort_values(
                ["observations", "cylinders_num"], ascending=[False, True]
            )
            .drop_duplicates(keys)
            .set_index(keys)["cylinders_num"]
        )

    def fit(self, X, y=None):
        observed = X.loc[
            ~X["cylinders_was_missing"],
            ["manufacturer_clean", "model_clean", "cylinders_num"],
        ]
        if observed.empty:
            raise ValueError("Train không có số xi-lanh gốc để điền.")
        self.model_modes_ = self._group_modes(
            observed, ["manufacturer_clean", "model_clean"]
        )
        self.brand_modes_ = self._group_modes(observed, ["manufacturer_clean"])
        self.global_mode_ = float(observed["cylinders_num"].mode().iloc[0])
        return self

    def transform(self, X):
        check_is_fitted(self, "global_mode_")
        result = X.copy()
        missing = result["cylinders_was_missing"]
        rows = result.loc[missing]
        keys = pd.MultiIndex.from_frame(rows[["manufacturer_clean", "model_clean"]])
        estimates = self.model_modes_.reindex(keys).to_numpy()
        brand = (
            rows["manufacturer_clean"]
            .map(self.brand_modes_)
            .fillna(self.global_mode_)
            .to_numpy()
        )
        result.loc[missing, "cylinders_num"] = np.where(
            np.isnan(estimates), brand, estimates
        )
        return result


class _KnownOrdinalEncoder(TransformerMixin, BaseEstimator):
    """Giữ hiệu ứng riêng cho unknown qua cờ, với mốc ordinal bằng 0."""

    def fit(self, X, y=None):
        frame = _categories(X)
        sources = {
            "condition_clean": CONDITION_ORDINAL,
            "title_status_clean": TITLE_STATUS_ORDINAL_EXPERIMENTAL,
        }
        self.mappings_ = {column: sources[column] for column in frame}
        return self

    def transform(self, X):
        check_is_fitted(self, "mappings_")
        frame = _categories(X)
        output = []
        for column, mapping in self.mappings_.items():
            values = frame[column].map(mapping)
            output.extend((values.fillna(0), values.isna()))
        return np.column_stack(output).astype(float)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "mappings_")
        columns = self.mappings_ if input_features is None else input_features
        return np.asarray(
            [
                f"{column}_{suffix}"
                for column in columns
                for suffix in ("ordinal", "unknown")
            ],
            dtype=object,
        )


def build_feature_preprocessor(
    strategy: str,
    numeric_cols: Sequence[str],
    ordinal_cols: Sequence[str],
    low_card_nominal_cols: Sequence[str],
    high_card_nominal_cols: Sequence[str],
    scale_numeric: bool = True,
    *,
    random_state: int = RANDOM_STATE,
    cv_folds: int = CV_FOLDS,
) -> ColumnTransformer:
    """Dựng các nhánh biến đổi trên danh sách đặc trưng được phép."""
    columns = [
        *numeric_cols,
        *ordinal_cols,
        *low_card_nominal_cols,
        *high_card_nominal_cols,
    ]
    forbidden = {"price", "log_price", "listing_id", "row_id", "split_set"}
    if (
        not columns
        or len(columns) != len(set(columns))
        or forbidden.intersection(columns)
    ):
        raise ValueError(
            "Danh sách đặc trưng rỗng, trùng hoặc chứa nhãn/khóa/cột chia tập."
        )
    if strategy not in (
        "raw_minimal",
        "numeric_only",
        "naive_label",
        "principled_linear",
        "principled_tree",
    ):
        raise ValueError(f"Chiến lược mã hóa không hợp lệ: {strategy}")

    numeric_steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
    if scale_numeric and strategy != "principled_tree":
        numeric_steps.append(StandardScaler())
    transformers = [("numeric", make_pipeline(*numeric_steps), list(numeric_cols))]
    if strategy == "naive_label":
        nominal = [*ordinal_cols, *low_card_nominal_cols, *high_card_nominal_cols]
        transformers.append(("naive", NaiveLabelEncodingTransformer(), nominal))
    elif strategy.startswith("principled_"):
        drop = "first" if strategy == "principled_linear" else None
        one_hot = make_pipeline(
            FunctionTransformer(_categories, feature_names_out="one-to-one"),
            OneHotEncoder(
                drop=drop,
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            ),
        )
        target = OutOfFoldTargetEncoder(n_splits=cv_folds, random_state=random_state)
        transformers.extend(
            [
                ("ordinal", _KnownOrdinalEncoder(), list(ordinal_cols)),
                ("nominal", one_hot, list(low_card_nominal_cols)),
                ("target", target, list(high_card_nominal_cols)),
            ]
        )
    return ColumnTransformer(transformers, remainder="drop")
