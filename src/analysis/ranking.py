"""M2-07 · Xếp hạng đặc trưng: dùng lại hàm của TV3, TV2 bổ sung liên hệ biên.

- |β_std| (OLS/Ridge trên biến chuẩn hóa, cùng tập dòng) là liên hệ CÓ ĐIỀU KIỆN:
  giữ các biến khác cố định. Tính bằng ``src.models.feature_ranking`` của TV3.
- |ρ| là liên hệ BIÊN của riêng từng biến với ``log_price``.
Hai bảng có thể xếp khác nhau vì tuổi xe và số dặm tương quan với nhau; cả hai
đều đúng theo nghĩa riêng. Chạy trên tập train để khớp với số của TV3.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.analysis.correlation_engine import compute_series_correlation
from src.models.feature_ranking import rank_features_on_common_support

# Trùng LINEARIZED_NUMERIC của TV3 (src/models/ablation_runner.py) để so được với Sheet 08b.
RANKING_FEATURES: tuple[str, ...] = ("car_age", "log_odometer", "miles_per_year", "cylinders_num")


def ranking_with_marginal(
    train_df: pd.DataFrame,
    features: Sequence[str] = RANKING_FEATURES,
    *,
    target: str = "log_price",
    dataset_version: str | None = None,
) -> pd.DataFrame:
    """Bảng ranking của TV3 cộng thêm Pearson/Spearman biên trên đúng các dòng đó."""
    features = list(features)
    ranking = rank_features_on_common_support(train_df, features, target_col=target)
    rows = train_df.loc[:, [*features, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(rows) != int(ranking["n_samples"].iloc[0]):
        raise AssertionError("Tập dòng của ranking TV3 và liên hệ biên không trùng nhau")
    marginal = {f: compute_series_correlation(rows[f], rows[target], f, target, min_n=4) for f in features}
    ranking["pearson_r_marginal"] = ranking["feature"].map(lambda f: marginal[f]["pearson_r"])
    ranking["spearman_rho_marginal"] = ranking["feature"].map(lambda f: marginal[f]["spearman_rho"])
    ranking["abs_rho_marginal"] = ranking["spearman_rho_marginal"].abs()
    ranking["rank_marginal_rho"] = ranking["abs_rho_marginal"].rank(ascending=False, method="min")
    ranking["note"] = np.where(
        ranking["feature"].eq("cylinders_num"), "cylinders_num gồm giá trị TV1 đã điền", ""
    )
    ranking.insert(0, "scope", "train")
    ranking.insert(0, "dataset_version", dataset_version)
    return ranking
