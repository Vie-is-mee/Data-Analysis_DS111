"""Thang đo hồi quy và nghịch đảo log có hiệu chỉnh Duan."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RegressionMetricsReport:
    r2: float
    adj_r2: float
    ss_res: float
    ss_tot: float
    y_mean_baseline: float
    mse: float
    rmse: float
    mae: float
    mape_pct: float
    n_samples: int
    n_features: int
    smearing_factor: float | None


def _paired_arrays(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.ndim != 1 or actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Cần hai mảng một chiều khác rỗng và cùng kích thước.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Giá trị quan sát và dự báo phải hữu hạn.")
    return actual, predicted


def compute_duan_smearing_factor(
    y_train_log_true: np.ndarray, y_train_log_pred: np.ndarray
) -> float:
    """Tính trung bình exp(phần dư log) chỉ từ các hàng huấn luyện."""
    actual, predicted = _paired_arrays(y_train_log_true, y_train_log_pred)
    return float(np.exp(np.clip(actual - predicted, -50, 50)).mean())


def inverse_log_transform(
    y_log_pred: np.ndarray, smearing_factor: float = 1.0
) -> np.ndarray:
    """Nghịch đảo log trong miền số hữu hạn, với hệ số Duan dương."""
    predictions = np.asarray(y_log_pred, dtype=np.float64)
    if not np.isfinite(smearing_factor) or smearing_factor <= 0:
        raise ValueError("Hệ số Duan phải hữu hạn và dương.")
    if not np.isfinite(predictions).all():
        raise ValueError("Dự báo log phải hữu hạn.")
    return np.exp(np.clip(predictions + np.log(smearing_factor), -50, 50))


def evaluate_regression(
    y_true_usd: np.ndarray,
    y_pred_usd: np.ndarray,
    n_features: int,
    smearing_factor: float | None = None,
) -> RegressionMetricsReport:
    """R² so sánh SS_res với SS_tot quanh trung bình của tập đánh giá.

    R² âm khi mô hình dự báo tệ hơn đường trung bình. R² không xác định
    khi nhãn hằng. Adjusted R² cần n > p + 1. MAPE dùng các nhãn khác 0.
    Hàm cũng dùng được cho thang log, nhưng MAPE log không mang nghĩa USD.
    """
    actual, predicted = _paired_arrays(y_true_usd, y_pred_usd)
    if n_features < 0:
        raise ValueError("Số đặc trưng phải không âm.")
    baseline = float(actual.mean())
    errors = actual - predicted
    ss_res = float(errors @ errors)
    centered = actual - baseline
    ss_tot = float(centered @ centered) if np.any(actual != actual[0]) else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n_samples = actual.size
    adj_r2 = (
        1.0 - (1.0 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        if n_samples > n_features + 1
        else float("nan")
    )
    nonzero = actual != 0
    mape = (
        float(100 * np.abs(errors[nonzero] / actual[nonzero]).mean())
        if nonzero.any()
        else float("nan")
    )
    mse = ss_res / n_samples
    return RegressionMetricsReport(
        r2=r2,
        adj_r2=adj_r2,
        ss_res=ss_res,
        ss_tot=ss_tot,
        y_mean_baseline=baseline,
        mse=float(mse),
        rmse=float(np.sqrt(mse)),
        mae=float(np.abs(errors).mean()),
        mape_pct=mape,
        n_samples=n_samples,
        n_features=n_features,
        smearing_factor=smearing_factor,
    )
