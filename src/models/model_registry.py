"""Danh mục kết quả đối chứng và đầu ra bàn giao Power BI."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from configs.project_config import RANDOM_STATE
from src.models.metrics import BinnedDuanSmearing, RegressionMetricsReport


@dataclass(slots=True)
class ModelEntry:
    model_id: str
    experiment_stage: str
    model_name: str
    model_family: str
    target_mode: str
    encoding_strategy: str
    feature_names: list[str]
    hyperparams: dict[str, Any]
    cv_metrics_mean: dict[str, float]
    cv_metrics_std: dict[str, float]
    val_metrics: RegressionMetricsReport
    train_time_sec: float
    notes: str
    fitted_pipeline: Any = None
    test_metrics: RegressionMetricsReport | None = None
    y_pred_test: pd.Series | None = None
    val_log_metrics: RegressionMetricsReport | None = None
    test_log_metrics: RegressionMetricsReport | None = None
    dataset_version: str = "v1.0"
    random_state: int = RANDOM_STATE
    train_metrics: RegressionMetricsReport | None = None
    train_log_metrics: RegressionMetricsReport | None = None
    binned_smearing: BinnedDuanSmearing | None = None


class ModelRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, ModelEntry] = {}

    def register(self, entry: ModelEntry) -> None:
        if not entry.model_id or entry.model_id in self._entries:
            raise ValueError(f"model_id rỗng hoặc trùng: {entry.model_id!r}")
        self._entries[entry.model_id] = entry

    def get(self, model_id: str) -> ModelEntry:
        return self._entries[model_id]

    def list_ids(self) -> list[str]:
        return list(self._entries)

    def get_best_model(
        self, metric: str = "rmse", ascending: bool = True
    ) -> ModelEntry:
        """Xếp theo CV; validation phá hòa, tiếp theo là thứ tự model_id."""
        candidates = [
            entry
            for entry in self._entries.values()
            if metric in entry.cv_metrics_mean
            and np.isfinite(entry.cv_metrics_mean[metric])
        ]
        if not candidates:
            raise ValueError(f"Không có kết quả CV hữu hạn cho {metric!r}.")
        direction = 1 if ascending else -1

        def rank_key(entry):
            val = getattr(entry.val_metrics, metric, float("nan"))
            return (
                direction * entry.cv_metrics_mean[metric],
                direction * val if np.isfinite(val) else float("inf"),
                entry.model_id,
            )

        return min(candidates, key=rank_key)

    def to_summary_dataframe(self) -> pd.DataFrame:
        rows = []
        for entry in self._entries.values():
            row = {
                "model_id": entry.model_id,
                "experiment_stage": entry.experiment_stage,
                "model_name": entry.model_name,
                "model_family": entry.model_family,
                "target_mode": entry.target_mode,
                "encoding_strategy": entry.encoding_strategy,
                "dataset_version": entry.dataset_version,
                "random_state": entry.random_state,
                "train_time_sec": entry.train_time_sec,
                "test_evaluated": entry.test_metrics is not None,
                "notes": entry.notes,
            }
            row.update(
                {
                    f"cv_{key}_mean": value
                    for key, value in entry.cv_metrics_mean.items()
                }
            )
            row.update(
                {f"cv_{key}_std": value for key, value in entry.cv_metrics_std.items()}
            )
            for prefix, report in (
                ("train", entry.train_metrics),
                ("val", entry.val_metrics),
                ("test", entry.test_metrics),
                ("train_log", entry.train_log_metrics),
                ("val_log", entry.val_log_metrics),
                ("test_log", entry.test_log_metrics),
            ):
                if report is not None:
                    row.update(
                        {
                            f"{prefix}_{key}": value
                            for key, value in asdict(report).items()
                        }
                    )
            if (
                entry.train_metrics is not None
                and "r2" in entry.cv_metrics_mean
                and np.isfinite(entry.train_metrics.r2)
                and np.isfinite(entry.cv_metrics_mean["r2"])
            ):
                row["train_cv_r2_gap"] = (
                    entry.train_metrics.r2 - entry.cv_metrics_mean["r2"]
                )
            if (
                entry.train_metrics is not None
                and "rmse" in entry.cv_metrics_mean
                and np.isfinite(entry.train_metrics.rmse)
                and np.isfinite(entry.cv_metrics_mean["rmse"])
            ):
                row["cv_train_rmse_gap"] = (
                    entry.cv_metrics_mean["rmse"] - entry.train_metrics.rmse
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def export_predictions_for_dashboard(
        self, model_id: str, test_df: pd.DataFrame, output_path: Path
    ) -> Path:
        """Xuất theo khóa listing_id, giữ thứ tự dự báo độc lập thứ tự truyền vào."""
        entry = self.get(model_id)
        predictions = entry.y_pred_test
        if entry.test_metrics is None or predictions is None:
            raise ValueError("Cần đánh giá test của mô hình đã chọn trước khi xuất.")
        ids = pd.Index(test_df["listing_id"], name="listing_id")
        if ids.dtype != np.dtype("int64") or not ids.is_unique or not len(ids):
            raise ValueError(
                "listing_id của test phải là int64, duy nhất và khác rỗng."
            )
        if not test_df["split_set"].eq("test").fillna(False).all():
            raise ValueError("Đầu ra Power BI chỉ được chứa test.")
        if not predictions.index.is_unique or len(predictions) != len(ids):
            raise ValueError("Số dự báo lệch hoặc listing_id bị trùng.")
        if not ids.difference(predictions.index).empty:
            raise ValueError("Khóa listing_id của dữ liệu khác dự báo đã ghi nhận.")
        actual = test_df["price"].to_numpy(dtype=float)
        predicted = predictions.reindex(ids).to_numpy(dtype=float)
        if not (
            np.isfinite(actual).all()
            and np.isfinite(predicted).all()
            and (actual > 0).all()
            and (predicted > 0).all()
        ):
            raise ValueError("Giá thực tế và dự báo phải dương, hữu hạn.")
        residual = actual - predicted
        output = pd.DataFrame(
            {
                "listing_id": ids.to_numpy(),
                "split_set": pd.Series("test", index=range(len(ids)), dtype="string"),
                "experiment_id": pd.Series(
                    entry.experiment_stage.split("_", 1)[0],
                    index=range(len(ids)),
                    dtype="string",
                ),
                "model_name": pd.Series(
                    entry.model_name, index=range(len(ids)), dtype="string"
                ),
                "actual_price_usd": actual,
                "predicted_price_usd": predicted,
                "residual_usd": residual,
                "absolute_error_usd": np.abs(residual),
                "dataset_version": pd.Series(
                    entry.dataset_version, index=range(len(ids)), dtype="string"
                ),
            }
        )
        for column in ("manufacturer_clean", "car_age", "year_era"):
            if column in test_df:
                output[column] = test_df[column].reset_index(drop=True)
        output["abs_pct_error"] = 100 * np.abs(residual / actual)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(output_path, engine="pyarrow", index=False)
        return output_path

    def save_registry(self, filepath: Path, *, include_models: bool = False) -> Path:
        """Lưu kết quả bằng joblib; chỉ nạp lại từ nguồn nội bộ tin cậy."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            self._entries
            if include_models
            else {
                key: replace(entry, fitted_pipeline=None)
                for key, entry in self._entries.items()
            }
        )
        joblib.dump(payload, path, compress=3)
        return path
