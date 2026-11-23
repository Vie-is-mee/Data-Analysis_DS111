"""VIF và trọng số chuẩn hóa trên tập train có cùng biến và cùng hàng."""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _complete_rows(df, columns):
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Danh sách cột phải khác rỗng và không trùng.")
    data = df.loc[:, columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2:
        raise ValueError("Cần ít nhất hai dòng hữu hạn trên cùng tập biến.")
    return data


def _vif_table(features, standardized):
    constant = np.ptp(standardized, axis=0) == 0
    matrix = np.column_stack([np.ones(len(standardized)), standardized[:, ~constant]])
    scores = np.full(len(features), np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="divide by zero encountered", category=RuntimeWarning
        )
        # Các cảnh báo số của hồi quy phụ được biểu diễn bằng VIF và cờ ở bảng.
        warnings.filterwarnings(
            "ignore", message="The design matrix is (poorly conditioned|rank-deficient)"
        )
        for index, feature_index in enumerate(np.flatnonzero(~constant), start=1):
            vif = variance_inflation_factor(matrix, index)
            # Với cộng tuyến hoàn hảo, làm tròn R² có thể cho giá trị hữu hạn cỡ 10^15.
            scores[feature_index] = (
                np.inf if vif >= 1 / (10 * np.finfo(float).eps) else vif
            )
    return pd.DataFrame(
        {
            "feature": features,
            "vif": scores,
            "vif_flag": np.select(
                [constant, scores > 10, scores > 5],
                ["Không xác định (cột hằng)", "Nghiêm trọng", "Cảnh báo"],
                default="Trong ngưỡng",
            ),
            "n_samples": len(standardized),
        }
    )


def compute_vif_table(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    """VIF với hệ số chặn, ghi vô hạn cho cộng tuyến hoàn hảo."""
    data = _complete_rows(df, numeric_features)
    return _vif_table(numeric_features, StandardScaler().fit_transform(data))


def rank_features_on_common_support(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "log_price",
    alpha_ridge: float = 1.0,
) -> pd.DataFrame:
    """Fit OLS và Ridge trên cùng hàng, cùng biến và cùng thang chuẩn hóa."""
    if {"price", "log_price", "listing_id", "row_id", "split_set"}.intersection(
        feature_cols
    ):
        raise ValueError(
            "Xếp hạng không nhận nhãn, khóa hoặc cột chia tập làm đặc trưng."
        )
    data = _complete_rows(train_df, [*feature_cols, target_col])
    scaled = StandardScaler().fit_transform(data[feature_cols])
    target = data[target_col]
    ols = LinearRegression().fit(scaled, target).coef_
    ridge = Ridge(alpha=alpha_ridge).fit(scaled, target).coef_
    report = _vif_table(feature_cols, scaled)
    for name, weights in (("ols", ols), ("ridge", ridge)):
        report[f"beta_{name}"] = weights
        report[f"abs_beta_{name}"] = np.abs(weights)
        report[f"sign_{name}"] = np.select(
            [weights > 0, weights < 0], ["+", "-"], default="0"
        )
        report[f"rank_{name}"] = (
            report[f"abs_beta_{name}"].rank(method="min", ascending=False).astype(int)
        )
    return (
        report.assign(rank=report["rank_ols"])
        .sort_values(["rank", "feature"])
        .reset_index(drop=True)
    )
