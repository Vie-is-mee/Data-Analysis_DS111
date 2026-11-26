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
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd

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


def plot_errors(data, out, subtitle):
    """Dùng bin mật độ để biểu diễn tất cả quan sát, kể cả tập test lớn."""
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
        data, winner, metrics = validate_test_predictions(
            abt, pd.read_parquet(args.splits), pd.read_parquet(args.predictions),
            pd.read_csv(args.summary), version,
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
    plot_errors(data, args.out, f"{model} · {version} · test n={len(data):,}")
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
