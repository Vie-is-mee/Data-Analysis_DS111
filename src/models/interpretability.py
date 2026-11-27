"""Diễn giải khoa học cho TV3 (chỉ dùng tập train & mô hình đã fit, không chạm test):
1. Hệ số góc khấu hao 3 chế độ tuổi (0–15, 15–25, >25) và phụ trội phiên bản.
2. Bảng xếp hạng Giá trị nội tại dòng xe sau khi trừ hao mòn (n >= min_n trên train).
3. Tầm quan trọng đặc trưng gom theo nhóm ngữ nghĩa cho mô hình cây chốt.
4. Đường cong điều chuẩn Ridge alpha (Bài 05: Underfit -> Good Fit -> Overfit).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold

from src.models.domain_features import (
    DOMAIN_TOKEN_FEATURES,
    WearAdjustedHierarchicalEncoder,
    add_domain_depreciation_features,
)
from src.models.metrics import (
    evaluate_regression,
    fit_binned_duan_smearing,
    inverse_log_transform_binned,
)
from src.models.model_registry import ModelEntry
from src.models.preprocessing_bridge import CylinderModeImputer

plt.switch_backend("Agg")

RIDGE_ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 500.0)


def compute_depreciation_slopes(
    train_df: pd.DataFrame,
    *,
    groups: pd.Series | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Khớp Ridge trên thang gốc các biến diễn giải để đọc tốc độ khấu hao 3 vùng tuổi."""
    enriched = add_domain_depreciation_features(train_df)
    if "cylinders_was_missing" in enriched:
        enriched = CylinderModeImputer().fit_transform(enriched)
    y_log = enriched["log_price"].to_numpy(dtype=float)
    encoder = WearAdjustedHierarchicalEncoder(
        mode="wear_adjusted_hierarchical", n_splits=n_splits, random_state=random_state
    )
    oof = encoder.fit_transform(enriched, y_log, groups=groups)
    cols = [
        "car_age",
        "age_hinge_15",
        "age_hinge_25",
        "log_odometer",
        "rel_log_odometer_by_age",
        "cylinders_num",
        "wear_adj_model_premium",
        *DOMAIN_TOKEN_FEATURES,
    ]
    X = oof[cols].fillna(0.0)
    model = Ridge(alpha=1.0).fit(X, y_log)
    coefs = dict(zip(cols, model.coef_))

    s0_15 = float(coefs["car_age"])
    s15_25 = float(coefs["car_age"] + coefs["age_hinge_15"])
    s25_plus = float(coefs["car_age"] + coefs["age_hinge_15"] + coefs["age_hinge_25"])
    rows = [
        ("Khấu hao tuổi 0–15 năm", "slope_age_0_15", s0_15, (np.exp(s0_15) - 1.0) * 100.0, "%/năm"),
        ("Khấu hao tuổi 15–25 năm", "slope_age_15_25", s15_25, (np.exp(s15_25) - 1.0) * 100.0, "%/năm"),
        ("Tăng giá xe cổ >25 năm", "slope_age_25_plus", s25_plus, (np.exp(s25_plus) - 1.0) * 100.0, "%/năm"),
        ("Độ lệch số dặm so với xe cùng tuổi", "rel_log_odometer_by_age", float(coefs["rel_log_odometer_by_age"]), np.nan, "log/đơn vị"),
        ("Số xi-lanh động cơ", "cylinders_num", float(coefs["cylinders_num"]), (np.exp(coefs["cylinders_num"]) - 1.0) * 100.0, "%/xi-lanh"),
        ("Giá trị nội tại dòng xe (trừ hao mòn)", "wear_adj_model_premium", float(coefs["wear_adj_model_premium"]), np.nan, "hệ số co giãn"),
    ]
    token_labels = {
        "tok_performance": "Phụ trội bản thể thao (AMG/GT/911/Raptor...)",
        "tok_heavy_duty": "Phụ trội tải nặng (2500/3500/HD/Cummins...)",
        "tok_luxury_trim": "Phụ trội bản cao cấp (Limited/Denali/Platinum...)",
        "tok_hybrid_ev": "Phụ trội Hybrid / EV trong tên dòng xe",
        "tok_awd_4wd": "Phụ trội AWD / 4WD trong tên dòng xe",
        "tok_commercial_van": "Phụ trội van thương mại (Sprinter/Transit...)",
    }
    for col, label in token_labels.items():
        val = float(coefs[col])
        rows.append((label, col, val, (np.exp(val) - 1.0) * 100.0, "% giá"))
    return pd.DataFrame(
        rows,
        columns=["factor", "feature_key", "log_coef", "approx_pct_effect", "unit"],
    )


def compute_intrinsic_model_ranking(
    train_df: pd.DataFrame, *, min_n: int = 50
) -> pd.DataFrame:
    """Xếp hạng dòng xe (`make|model_stem`) theo Giá trị nội tại sau khi trừ hao mòn (chỉ trên train)."""
    enriched = add_domain_depreciation_features(train_df)
    if "cylinders_was_missing" in enriched:
        enriched = CylinderModeImputer().fit_transform(enriched)
    y_log = enriched["log_price"].to_numpy(dtype=float)
    encoder = WearAdjustedHierarchicalEncoder(mode="wear_adjusted_hierarchical")
    encoder.fit(enriched, y_log)
    keys = (
        enriched["manufacturer_clean"].astype("string").fillna("unknown").astype(str)
        + "|"
        + enriched["model_stem"].astype("string").fillna("unknown").astype(str)
    )
    summary = (
        enriched.assign(make_stem=keys)
        .groupby("make_stem", sort=False)
        .agg(
            n_train=("price", "size"),
            median_price_usd=("price", "median"),
            median_age_yr=("car_age", "median"),
        )
        .reset_index()
    )
    summary = summary.loc[summary["n_train"] >= min_n].copy()
    summary["wear_adj_log_premium"] = (
        summary["make_stem"].map(encoder.state_["ms"]).astype(float)
    )
    summary["pct_vs_wear_peer"] = (np.exp(summary["wear_adj_log_premium"]) - 1.0) * 100.0
    return (
        summary.sort_values("wear_adj_log_premium", ascending=False)
        .reset_index(drop=True)
    )


def _pillar_for_feature(name: str) -> str:
    clean = name.split("__", 1)[-1]
    if clean.startswith(("wear_adj_", "flat_", "stem_", "age_x_model")):
        return "Mã hóa phân cấp phần dư dòng xe & vùng (Exp5)"
    if clean in ("car_age", "car_age_squared", "age_hinge_15", "age_hinge_25", "rel_age_within_stem"):
        return "Tuổi xe & Khấu hao 3 chế độ"
    if clean in ("log_odometer", "miles_per_year", "rel_log_odometer_by_age", "age_x_log_odometer"):
        return "Số dặm & Độ lệch cùng tuổi"
    if clean.startswith("tok_"):
        return "Tín hiệu phiên bản / tải nặng từ tên dòng xe"
    if clean == "cylinders_num" or clean.startswith(("fuel_", "transmission_", "drive_", "type_")):
        return "Thông số kỹ thuật (động cơ, kiểu dáng, dẫn động)"
    if clean.startswith(("manufacturer_clean_", "state_")):
        return "Hãng xe & Bang (One-Hot)"
    if clean.startswith(("condition_clean_", "title_status_clean_")):
        return "Tình trạng xe & Pháp lý (Title)"
    return "Khác"


def compute_grouped_feature_importance(entry: ModelEntry) -> pd.DataFrame | None:
    """Gom mức đóng góp Gain/Importance của mô hình cây theo nhóm ngữ nghĩa."""
    pipe = entry.fitted_pipeline
    if pipe is None or "regression" not in pipe.named_steps or "preprocess" not in pipe.named_steps:
        return None
    reg = pipe["regression"]
    names = list(pipe["preprocess"].get_feature_names_out())
    if hasattr(reg, "booster_"):
        raw = np.asarray(reg.booster_.feature_importance(importance_type="gain"), dtype=float)
    elif hasattr(reg, "feature_importances_"):
        raw = np.asarray(reg.feature_importances_, dtype=float)
    else:
        return None
    if len(raw) != len(names) or raw.sum() <= 0:
        return None
    frame = pd.DataFrame(
        {
            "feature": names,
            "pillar": [_pillar_for_feature(n) for n in names],
            "importance_raw": raw,
        }
    )
    grouped = (
        frame.groupby("pillar", as_index=False)["importance_raw"]
        .sum()
        .sort_values("importance_raw", ascending=False)
        .reset_index(drop=True)
    )
    grouped["importance_pct"] = grouped["importance_raw"] / grouped["importance_raw"].sum() * 100.0
    grouped["model_id"] = entry.model_id
    return grouped


def compute_ridge_validation_curve(
    ridge_entry: ModelEntry,
    train_df: pd.DataFrame,
    *,
    groups: pd.Series | None = None,
    alphas: Sequence[float] = RIDGE_ALPHAS,
    cv_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Đường cong kiểm định Ridge alpha trên train CV (Bài 05: Underfit -> Good Fit -> Overfit)."""
    pipe = ridge_entry.fitted_pipeline
    if pipe is None:
        raise ValueError("Cần pipeline đã khởi tạo của Ridge để chạy đường cong alpha.")
    X = train_df.loc[:, list(ridge_entry.feature_names)]
    y_log = train_df["log_price"].to_numpy(dtype=float)
    y_usd = train_df["price"].to_numpy(dtype=float)
    has_domain = "domain_encode" in pipe.named_steps
    use_binned = ridge_entry.binned_smearing is not None

    if groups is not None:
        splitter = list(GroupKFold(n_splits=cv_folds).split(X, y_log, groups=groups))
    else:
        splitter = list(
            KFold(cv_folds, shuffle=True, random_state=random_state).split(X)
        )

    # Biến đổi đặc trưng trước một lần mỗi fold (tránh tính lại OOF encoder 5 lần cho 5 alpha).
    prep_steps = [(name, step) for name, step in pipe.steps if name != "regression"]
    prep_proto = clone(pipe.__class__(prep_steps))
    fold_matrices = []
    for tr_i, va_i in splitter:
        fit_kw = (
            {"domain_encode__groups": groups.iloc[tr_i]}
            if (has_domain and groups is not None)
            else {}
        )
        prep = clone(prep_proto)
        X_tr = prep.fit_transform(X.iloc[tr_i], y_log[tr_i], **fit_kw)
        X_va = prep.transform(X.iloc[va_i])
        fold_matrices.append((tr_i, va_i, X_tr, X_va))

    rows = []
    for alpha in alphas:
        tr_r2s, cv_r2s, cv_rmses, cv_lr2s = [], [], [], []
        for tr_i, va_i, X_tr, X_va in fold_matrices:
            reg = Ridge(alpha=float(alpha)).fit(X_tr, y_log[tr_i])
            p_tr = reg.predict(X_tr)
            p_va = reg.predict(X_va)
            if use_binned:
                sm = fit_binned_duan_smearing(y_log[tr_i], p_tr)
                usd_tr = inverse_log_transform_binned(p_tr, sm)
                usd_va = inverse_log_transform_binned(p_va, sm)
            else:
                fac = float(np.exp(np.clip(y_log[tr_i] - p_tr, -50, 50)).mean())
                usd_tr = np.exp(np.clip(p_tr + np.log(fac), -50, 50))
                usd_va = np.exp(np.clip(p_va + np.log(fac), -50, 50))
            tr_r2s.append(evaluate_regression(y_usd[tr_i], usd_tr, 1).r2)
            va_rep = evaluate_regression(y_usd[va_i], usd_va, 1)
            cv_r2s.append(va_rep.r2)
            cv_rmses.append(va_rep.rmse)
            cv_lr2s.append(evaluate_regression(y_log[va_i], p_va, 1).r2)
        rows.append(
            {
                "model_id": ridge_entry.model_id,
                "alpha": float(alpha),
                "train_r2": float(np.mean(tr_r2s)),
                "cv_r2": float(np.mean(cv_r2s)),
                "cv_rmse_usd": float(np.mean(cv_rmses)),
                "cv_log_r2": float(np.mean(cv_lr2s)),
            }
        )
    return pd.DataFrame(rows)


def plot_grouped_feature_importance(
    grouped_df: pd.DataFrame, output_path: Path, version: str
) -> Path:
    """Vẽ biểu đồ đóng góp theo nhóm ngữ nghĩa (300 DPI)."""
    fig, ax = plt.subplots(figsize=(9.2, 4.8), layout="constrained")
    try:
        labels = grouped_df["pillar"].tolist()
        values = grouped_df["importance_pct"].to_numpy(dtype=float)
        y_pos = np.arange(len(labels))
        colors = ["#1b7837" if i == 0 else "#2b6cb0" for i in range(len(labels))]
        bars = ax.barh(y_pos, values, color=colors, height=0.62)
        ax.set_yticks(y_pos, labels)
        ax.invert_yaxis()
        ax.set(
            xlabel="Tỷ trọng đóng góp Gain (%)",
            xlim=(0, float(values.max()) * 1.18),
        )
        ax.set_title(
            f"Đóng góp theo nhóm đặc trưng (Gain)\n{grouped_df['model_id'].iloc[0]} · {version}",
            fontsize=11,
        )
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        for bar, value in zip(bars, values):
            ax.text(
                value + float(values.max()) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                fontsize=9,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        return output_path
    finally:
        plt.close(fig)


def plot_ridge_validation_curve(
    curve_df: pd.DataFrame, output_path: Path, version: str
) -> Path:
    """Vẽ đường cong kiểm định Ridge alpha (Bài 05: Train vs CV)."""
    fig, (ax_r2, ax_rmse) = plt.subplots(1, 2, figsize=(11.2, 4.4), layout="constrained")
    try:
        alphas = curve_df["alpha"].to_numpy(dtype=float)
        best_idx = int(curve_df["cv_rmse_usd"].idxmin())
        best_alpha = float(curve_df.loc[best_idx, "alpha"])

        ax_r2.semilogx(alphas, curve_df["train_r2"], "o-", color="#64748b", linewidth=2, label="Train R² (USD)")
        ax_r2.semilogx(alphas, curve_df["cv_r2"], "s-", color="#0f766e", linewidth=2.2, label="CV R² (USD)")
        ax_r2.axvline(best_alpha, color="#1b7837", linestyle="--", linewidth=1.2, label=f"Tối ưu α={best_alpha:g}")
        ax_r2.set(xlabel="Hệ số điều chuẩn Ridge α (thang log)", ylabel="R² (USD)", title=f"Train vs CV R² · {version}")
        ax_r2.grid(True, linestyle=":", alpha=0.35)
        ax_r2.legend(frameon=True, facecolor="white", framealpha=0.9)

        ax_rmse.semilogx(alphas, curve_df["cv_rmse_usd"], "s-", color="#b45309", linewidth=2.2, label="CV RMSE (USD)")
        ax_rmse.axvline(best_alpha, color="#1b7837", linestyle="--", linewidth=1.2, label=f"Min RMSE @ α={best_alpha:g}")
        ax_rmse.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax_rmse.set(xlabel="Hệ số điều chuẩn Ridge α (thang log)", ylabel="CV RMSE (USD)", title="Sai số CV theo hệ số điều chuẩn (Bài 05)")
        ax_rmse.grid(True, linestyle=":", alpha=0.35)
        ax_rmse.legend(frameon=True, facecolor="white", framealpha=0.9)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        return output_path
    finally:
        plt.close(fig)


def export_tv3_interpretability_pack(
    train_df: pd.DataFrame,
    winner_entry: ModelEntry,
    ridge_entry: ModelEntry | None,
    output_dir: Path,
    *,
    dataset_version: str,
    groups: pd.Series | None = None,
    min_model_n: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Path]:
    """Xuất trọn bộ bảng & biểu đồ diễn giải khoa học của TV3 (chỉ dùng train)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    slopes = compute_depreciation_slopes(
        train_df, groups=groups, n_splits=cv_folds, random_state=random_state
    ).assign(dataset_version=dataset_version)
    p_slopes = output_dir / "m3_depreciation_slopes.csv"
    slopes.to_csv(p_slopes, index=False, encoding="utf-8-sig")
    written["depreciation_slopes"] = p_slopes

    ranking = compute_intrinsic_model_ranking(train_df, min_n=min_model_n).assign(
        dataset_version=dataset_version
    )
    p_rank = output_dir / "m3_intrinsic_model_ranking.csv"
    ranking.to_csv(p_rank, index=False, encoding="utf-8-sig")
    written["intrinsic_model_ranking"] = p_rank

    imp = compute_grouped_feature_importance(winner_entry)
    if imp is not None:
        p_imp = output_dir / "m3_feature_importance_grouped.csv"
        imp.assign(dataset_version=dataset_version).to_csv(
            p_imp, index=False, encoding="utf-8-sig"
        )
        written["feature_importance_csv"] = p_imp
        written["feature_importance_png"] = plot_grouped_feature_importance(
            imp, output_dir / "model_feature_importance.png", dataset_version
        )

    if ridge_entry is not None and ridge_entry.fitted_pipeline is not None:
        curve = compute_ridge_validation_curve(
            ridge_entry,
            train_df,
            groups=groups,
            cv_folds=cv_folds,
            random_state=random_state,
        ).assign(dataset_version=dataset_version)
        p_curve = output_dir / "m3_validation_curve_ridge.csv"
        curve.to_csv(p_curve, index=False, encoding="utf-8-sig")
        written["validation_curve_csv"] = p_curve
        written["validation_curve_png"] = plot_ridge_validation_curve(
            curve, output_dir / "model_validation_curve_ridge.png", dataset_version
        )

    return written
