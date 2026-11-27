"""Báo cáo sai số của mô hình đã khóa trên tập test.

Chạy từ gốc repo:
python -m scripts.report_m3_residuals --abt <parquet> --splits <parquet>
    --predictions <parquet> --summary <csv> --dataset-version <phiên bản>
    --out <thư mục mới>
"""

from __future__ import annotations

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from src.models.metrics import evaluate_regression
from src.models.split_manifest import GROUP_COLUMN, validate_split_manifest

plt.switch_backend("Agg")

ACTUAL = "actual_price_usd"
PREDICTED = "predicted_price_usd"
RESIDUAL = "residual_usd"
VERSION = "dataset_version"
SEGMENTS = ("seller_type", "year_era", "type", "manufacturer_clean")
METRICS = {"rmse": "USD", "mae": "USD", "mape_pct": "%", "r2": "", "n_samples": "tin"}


def validate_test_predictions(abt, splits, predictions, summary, version):
    """Đối chiếu phiên bản, ID, mô hình đã chốt, giá thật và số liệu test."""
    for name, frame in (
        ("manifest", splits), ("dự báo", predictions), ("bảng mô hình", summary)
    ):
        if frame.empty or not frame[VERSION].eq(version).fillna(False).all():
            raise ValueError(f"Sai hoặc thiếu phiên bản dữ liệu trong {name}.")

    ids = abt["listing_id"]
    manifest = validate_split_manifest(splits, ids)
    if abt["posting_year"].ge(2026).any() and GROUP_COLUMN not in manifest:
        raise ValueError("ABT từ 2026 cần manifest nhóm xe.")
    if not abt["split_set"].astype("string").reset_index(drop=True).equals(
        manifest["split_set"].reset_index(drop=True).astype("string")
    ):
        raise ValueError("split_set của ABT không khớp manifest.")

    selected = summary.loc[summary["test_evaluated"].eq(True)]
    if len(selected) != 1:
        raise ValueError("Bảng mô hình phải có đúng một mô hình được chấm test.")
    winner = selected.iloc[0]
    if not (
        predictions["model_name"].eq(winner["model_name"]).fillna(False).all()
        and predictions["experiment_id"].eq(
            winner["experiment_stage"].split("_", 1)[0]
        ).fillna(False).all()
        and predictions["split_set"].eq("test").fillna(False).all()
    ):
        raise ValueError("Dự báo phải thuộc mô hình đã chốt và chỉ chứa tập test.")

    cohort = abt.loc[abt["split_set"].eq("test")].set_index("listing_id")
    observed_ids = pd.Index(predictions["listing_id"])
    if (
        observed_ids.dtype != np.dtype("int64")
        or not observed_ids.is_unique
        or len(observed_ids) != len(cohort)
        or not observed_ids.difference(cohort.index).empty
    ):
        raise ValueError("ID dự báo không khớp toàn bộ ID test của ABT.")
    data = predictions.set_index("listing_id").loc[cohort.index, [ACTUAL, PREDICTED]]
    if not np.array_equal(data[ACTUAL].to_numpy(), cohort["price"].to_numpy()):
        raise ValueError("Giá thật trong dự báo không khớp ABT theo listing_id.")
    if data.le(0).any().any():
        raise ValueError("Giá thật và giá dự báo phải dương.")

    metrics = evaluate_regression(data[ACTUAL], data[PREDICTED], int(winner["test_n_features"]))
    for metric in METRICS:
        if not np.isclose(
            getattr(metrics, metric), winner[f"test_{metric}"], rtol=1e-10,
            atol=1e-8, equal_nan=True
        ):
            raise ValueError(f"Số test_{metric} trong bảng mô hình lệch với dự báo.")

    data = data.join(cohort.loc[:, cohort.columns.intersection(SEGMENTS)])
    data[RESIDUAL] = data[ACTUAL] - data[PREDICTED]
    return data, winner, metrics


def segment_errors(data, min_group_n):
    """Nhóm nhỏ vẫn có số dòng và không công bố chỉ số sai số."""
    rows = []
    for column in SEGMENTS:
        if column not in data:
            continue
        for label, group in data.groupby(column, dropna=False, observed=True, sort=True):
            count = len(group)
            row = {"segment": column, "level": str(label), "n": count}
            if count >= min_group_n:
                measured = evaluate_regression(group[ACTUAL], group[PREDICTED], 0)
                row.update(
                    mae_usd=measured.mae,
                    rmse_usd=measured.rmse,
                    mean_residual_usd=group[RESIDUAL].mean(),
                    median_residual_usd=group[RESIDUAL].median(),
                )
            rows.append(row)
    return pd.DataFrame(
        rows, columns=["segment", "level", "n", "mae_usd", "rmse_usd",
                       "mean_residual_usd", "median_residual_usd"]
    )


COMPACT_SUMMARY_COLUMNS = (
    "model_id",
    "experiment_stage",
    "model_name",
    "target_mode",
    "train_r2",
    "cv_r2_mean",
    "train_cv_r2_gap",
    "cv_rmse_mean",
    "cv_mae_mean",
    "cv_log_r2_mean",
    "val_r2",
    "val_rmse",
    "test_r2",
    "test_rmse",
    "test_mae",
    "train_time_sec",
)


def _plot_distribution(data, out, subtitle):
    """Biểu đồ phân phối mật độ Thực tế vs Dự báo (Distribution Plot, Bài 05)."""
    actual = data[ACTUAL].to_numpy(dtype=float)
    predicted = data[PREDICTED].to_numpy(dtype=float)
    upper = (
        float(max(np.quantile(actual, 0.995), np.quantile(predicted, 0.995)))
        if len(data) >= 50
        else float(max(actual.max(), predicted.max()))
    )
    lower = 0.0
    if upper <= lower:
        upper = lower + 1.0
    grid = np.linspace(lower, upper, 320)
    med_actual = float(np.median(actual))
    med_pred = float(np.median(predicted))

    fig, ax = plt.subplots(figsize=(8.8, 5.2), layout="constrained")
    try:
        used_kde = False
        if (
            len(data) >= 5
            and np.std(actual) > 0
            and np.std(predicted) > 0
        ):
            try:
                kde_actual = gaussian_kde(actual)(grid)
                kde_pred = gaussian_kde(predicted)(grid)
                ax.plot(
                    grid,
                    kde_actual,
                    color="#1f5a85",
                    linewidth=2.2,
                    label=f"Giá rao thực tế (trung vị {med_actual:,.0f} USD)",
                )
                ax.fill_between(grid, kde_actual, color="#1f5a85", alpha=0.14)
                ax.plot(
                    grid,
                    kde_pred,
                    color="#d95f02",
                    linewidth=2.2,
                    linestyle="--",
                    label=f"Giá dự báo (trung vị {med_pred:,.0f} USD)",
                )
                ax.fill_between(grid, kde_pred, color="#d95f02", alpha=0.12)
                used_kde = True
            except np.linalg.LinAlgError:
                used_kde = False
        if not used_kde:
            bins = min(30, max(5, len(data)))
            ax.hist(
                actual,
                bins=bins,
                range=(lower, upper),
                density=True,
                histtype="step",
                linewidth=2.0,
                color="#1f5a85",
                label=f"Giá rao thực tế (trung vị {med_actual:,.0f} USD)",
            )
            ax.hist(
                predicted,
                bins=bins,
                range=(lower, upper),
                density=True,
                histtype="step",
                linewidth=2.0,
                linestyle="--",
                color="#d95f02",
                label=f"Giá dự báo (trung vị {med_pred:,.0f} USD)",
            )
        ax.axvline(med_actual, color="#1f5a85", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.axvline(med_pred, color="#d95f02", linestyle=":", linewidth=1.2, alpha=0.8)
        clipped = int((actual > upper).sum())
        note = f" · hiển thị tới P99.5 ({clipped:,} tin đuôi trên)" if clipped > 0 else ""
        ax.set(
            xlabel="Giá xe (USD)",
            ylabel="Mật độ phân phối (KDE)" if used_kde else "Mật độ phân phối",
            xlim=(lower, upper),
        )
        ax.set_title(
            f"Phân phối giá thực tế và giá dự báo\n{subtitle}{note}",
            fontsize=11,
        )
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.legend(frameon=True, facecolor="white", framealpha=0.9)
        fig.savefig(out / "model_distribution_actual_vs_predicted.png", dpi=300)
    finally:
        plt.close(fig)


def plot_ablation_summary(summary, out, version):
    """Vẽ đối chứng CV RMSE và Train vs CV R² (kiểm tra Underfit/Overfit)."""
    if len(summary) < 2 or "cv_rmse_mean" not in summary:
        return None
    frame = summary.loc[summary["cv_rmse_mean"].notna()].copy()
    if len(frame) < 2:
        return None
    labels = frame["model_id"].astype(str).tolist()
    y_pos = np.arange(len(labels))
    has_train_r2 = "train_r2" in frame and frame["train_r2"].notna().any()
    has_cv_r2 = "cv_r2_mean" in frame and frame["cv_r2_mean"].notna().any()

    ncols = 2 if has_cv_r2 else 1
    fig, axes = plt.subplots(
        1, ncols, figsize=(13.2 if ncols == 2 else 7.6, max(4.6, 0.45 * len(labels) + 1.4)),
        layout="constrained",
    )
    ax_rmse = axes[0] if ncols == 2 else axes
    try:
        colors = [
            "#1b7837"
            if bool(row.get("test_evaluated", False))
            else ("#2b6cb0" if row.get("model_family") == "Traditional" else "#d97706")
            for _, row in frame.iterrows()
        ]
        rmses = frame["cv_rmse_mean"].to_numpy(dtype=float)
        bars = ax_rmse.barh(y_pos, rmses, color=colors, height=0.62)
        ax_rmse.set_yticks(y_pos, labels)
        ax_rmse.invert_yaxis()
        stages = [str(s).split("_", 1)[0] for s in frame.get("experiment_stage", []) if pd.notna(s)]
        stage_span = f"{stages[0]}–{stages[-1]}" if len(stages) >= 2 else "các mô hình"
        ax_rmse.set(
            xlabel="CV RMSE (USD) — càng thấp càng tốt",
            title=f"Đối chứng {stage_span} ({version})",
        )
        ax_rmse.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax_rmse.set_xlim(0, float(rmses.max()) * 1.18)
        ax_rmse.grid(axis="x", linestyle=":", alpha=0.35)
        ax_rmse.legend(
            handles=[
                Patch(facecolor="#2b6cb0", label="Truyền thống (OLS/Ridge)"),
                Patch(facecolor="#d97706", label="Cây / Ensemble"),
                Patch(facecolor="#1b7837", label="Mô hình chốt theo CV"),
            ],
            loc="lower right",
            fontsize=8.2,
            frameon=True,
            facecolor="white",
            framealpha=0.9,
        )
        for bar, value in zip(bars, rmses):
            ax_rmse.text(
                value + float(rmses.max()) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{value:,.0f}",
                va="center",
                fontsize=8.5,
            )

        if ncols == 2:
            ax_r2 = axes[1]
            cv_r2 = frame["cv_r2_mean"].to_numpy(dtype=float)
            if has_train_r2:
                tr_r2 = frame["train_r2"].to_numpy(dtype=float)
                h = 0.34
                ax_r2.barh(y_pos - h / 2, tr_r2, height=h, color="#94a3b8", label="Train R² (USD)")
                ax_r2.barh(y_pos + h / 2, cv_r2, height=h, color="#0f766e", label="CV R² (USD)")
                ax_r2.set_title("Chẩn đoán Underfit / Overfit (Train vs CV R²)")
            else:
                ax_r2.barh(y_pos, cv_r2, height=0.58, color="#0f766e", label="CV R² (USD)")
                ax_r2.set_title("Hệ số giải thích CV R² (USD)")
            ax_r2.axvline(0, color="#aa3439", linestyle="--", linewidth=1)
            ax_r2.set_yticks(y_pos, [""] * len(labels))
            ax_r2.invert_yaxis()
            ax_r2.set_xlabel("R² trên thang giá USD")
            ax_r2.grid(axis="x", linestyle=":", alpha=0.35)
            ax_r2.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="white", framealpha=0.9)

        output_path = out / "model_ablation_comparison.png"
        fig.savefig(output_path, dpi=300)
        return output_path
    finally:
        plt.close(fig)


def plot_errors(data, out, subtitle):
    """Vẽ phân tán thực tế–dự báo, phần dư và phân phối mật độ (Bài 05)."""
    upper = float(data[[ACTUAL, PREDICTED]].max().max())
    for name, x, y, x_label, y_label in (
        ("model_actual_vs_predicted.png", ACTUAL, PREDICTED, "Giá rao thực tế (USD)", "Giá dự báo (USD)"),
        ("model_residuals.png", PREDICTED, RESIDUAL, "Giá dự báo (USD)", "Phần dư thực tế − dự báo (USD)"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 5.4), layout="constrained")
        try:
            points = ax.hexbin(data[x], data[y], gridsize=48, mincnt=1, bins="log", cmap="viridis")
            if y == RESIDUAL:
                ax.axhline(0, color="#aa3439", linestyle="--", linewidth=1)
            else:
                ax.plot([0, upper], [0, upper], color="#aa3439", linestyle="--", linewidth=1)
                ax.set_aspect("equal", adjustable="box")
            ax.set(xlabel=x_label, ylabel=y_label, title=subtitle)
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            fig.colorbar(points, ax=ax, label="Số tin trong ô (thang log)")
            fig.savefig(out / name, dpi=300)
        finally:
            plt.close(fig)
    _plot_distribution(data, out, subtitle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("abt", "splits", "predictions", "summary", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--min-group-n", default=30, type=int)
    args = parser.parse_args(argv)
    version = args.dataset_version.strip()
    if args.min_group_n < 2 or not version:
        parser.error("Cần nhãn phiên bản và ngưỡng nhóm tối thiểu từ hai tin.")
    if args.out.exists():
        parser.error("Chọn thư mục đầu ra mới.")

    try:
        abt = pd.read_parquet(args.abt)
        summary = pd.read_csv(args.summary)
        data, winner, metrics = validate_test_predictions(
            abt, pd.read_parquet(args.splits), pd.read_parquet(args.predictions),
            summary, version,
        )
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))

    args.out.mkdir(parents=True, exist_ok=False)
    model = winner["model_id"]
    identity = {"dataset_version": version, "model_id": model}
    key_numbers = pd.DataFrame([
        {**identity, "key": f"test_{name}", "value": getattr(metrics, name), "unit": unit}
        for name, unit in METRICS.items()
    ])
    key_numbers.to_csv(args.out / "m3_key_numbers.csv", index=False, encoding="utf-8-sig")
    segment_errors(data, args.min_group_n).assign(**identity).to_csv(
        args.out / "m3_residual_segments.csv", index=False, encoding="utf-8-sig"
    )
    compact_cols = [col for col in COMPACT_SUMMARY_COLUMNS if col in summary.columns]
    if compact_cols:
        summary[compact_cols].to_csv(
            args.out / "m3_ablation_compact.csv", index=False, encoding="utf-8-sig"
        )
    plot_errors(data, args.out, f"{model} · {version} · test n={len(data):,}")
    plot_ablation_summary(summary, args.out, version)
    hashes = {}
    for name in ("abt", "splits", "predictions", "summary"):
        with getattr(args, name).open("rb") as source:
            hashes[name] = file_digest(source, "sha256").hexdigest()
    test_metrics = {key: getattr(metrics, key) for key in METRICS}
    report = {
        **identity,
        "cohort": "test", "min_group_n": args.min_group_n,
        "input_sha256": hashes,
        "test_metrics": {
            key: value if np.isfinite(value) else None for key, value in test_metrics.items()
        },
    }
    (args.out / "residual_run_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] {model}, {len(data):,} dự báo test · {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
