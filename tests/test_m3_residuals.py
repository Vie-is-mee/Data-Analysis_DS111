"""Regression checks for the TV3 diagnostics handoff."""

from dataclasses import asdict

import pandas as pd
import pytest

from scripts.report_m3_residuals import (
    main as report_main,
    segment_errors,
    validate_test_predictions,
)
from src.models.metrics import evaluate_regression


@pytest.fixture
def handoff():
    abt = pd.DataFrame([
        (4, "train", 2000.0, "dealer"),
        (1, "test", 1000.0, "dealer"),
        (5, "val", 4000.0, "private"),
        (2, "test", 3000.0, "dealer"),
        (3, "test", 5000.0, "private"),
        *((key, "train", 2000.0, "dealer") for key in range(6, 11)),
    ], columns=["listing_id", "split_set", "price", "seller_type"]).assign(posting_year=2026)
    splits = abt[["listing_id", "split_set"]].assign(
        dataset_version="dev-test", vehicle_group_id="g:" + abt["listing_id"].astype(str)
    ).sample(frac=1, random_state=3)
    pred = pd.DataFrame({
        "listing_id": [3, 1, 2],
        "dataset_version": "dev-test",
        "split_set": "test",
        "model_name": "OLS",
        "experiment_id": "Exp0",
        "actual_price_usd": [5000.0, 1000.0, 3000.0],
        "predicted_price_usd": [4000.0, 1100.0, 3200.0],
    })
    report = evaluate_regression(pred["actual_price_usd"], pred["predicted_price_usd"], 1)
    summary = pd.DataFrame([{
        "dataset_version": "dev-test", "model_id": "Exp0_OLS", "model_name": "OLS",
        "experiment_stage": "Exp0_Raw_Minimal", "test_evaluated": True,
        **{f"test_{key}": value for key, value in asdict(report).items()},
    }])
    return abt, splits, pred, summary


def test_residuals_align_by_id_and_keep_group_counts(handoff):
    rows, winner, metrics = validate_test_predictions(*handoff, "dev-test")
    assert winner["model_id"] == "Exp0_OLS"
    assert metrics.n_samples == 3
    assert rows["residual_usd"].to_dict() == {1: -100.0, 2: -200.0, 3: 1000.0}
    groups = segment_errors(rows, 2).set_index("level")
    assert groups.loc["dealer", "mae_usd"] == 150.0
    assert groups.loc["private", "n"] == 1
    assert pd.isna(groups.loc["private", "mae_usd"])


@pytest.mark.parametrize(("table", "column", "value", "error"), [
    (1, "vehicle_group_id", "g:4", "nhiều tập"),
    (2, "dataset_version", "other", "phiên bản"),
    (2, "listing_id", 4, "khớp toàn bộ ID test"),
    (2, "actual_price_usd", 5100.0, "khớp ABT"),
    (2, "model_name", "Ridge", "mô hình đã chốt"),
    (3, "test_rmse", 1.0, "test_rmse"),
])
def test_residuals_reject_mixed_handoffs(handoff, table, column, value, error):
    frame = handoff[table]
    index = frame.index[frame["listing_id"].eq(1)][0] if table == 1 else frame.index[0]
    frame.loc[index, column] = value
    with pytest.raises(ValueError, match=error):
        validate_test_predictions(*handoff, "dev-test")


def test_report_cli_exports_distribution_ablation_and_compact_summary(handoff, tmp_path):
    abt, splits, pred, summary = handoff
    summary = pd.concat(
        [
            summary.assign(
                model_family="Traditional",
                target_mode="raw_price",
                train_r2=0.80,
                cv_r2_mean=0.72,
                train_cv_r2_gap=0.08,
                cv_rmse_mean=620.0,
                cv_mae_mean=450.0,
                val_r2=0.71,
                val_rmse=630.0,
                train_time_sec=0.5,
            ),
            summary.assign(
                model_id="Exp4_Ridge",
                model_name="Ridge",
                model_family="Traditional",
                experiment_stage="Exp4_Benchmark",
                target_mode="log_price",
                test_evaluated=False,
                train_r2=0.84,
                cv_r2_mean=0.78,
                train_cv_r2_gap=0.06,
                cv_rmse_mean=510.0,
                cv_mae_mean=390.0,
                val_r2=0.77,
                val_rmse=520.0,
                train_time_sec=0.6,
            ),
        ],
        ignore_index=True,
    )
    abt_path = tmp_path / "abt.parquet"
    splits_path = tmp_path / "splits.parquet"
    pred_path = tmp_path / "pred.parquet"
    summary_path = tmp_path / "summary.csv"
    out_dir = tmp_path / "q5_report"
    abt.to_parquet(abt_path, index=False)
    splits.to_parquet(splits_path, index=False)
    pred.to_parquet(pred_path, index=False)
    summary.to_csv(summary_path, index=False)

    assert (
        report_main(
            [
                "--abt", str(abt_path),
                "--splits", str(splits_path),
                "--predictions", str(pred_path),
                "--summary", str(summary_path),
                "--dataset-version", "dev-test",
                "--min-group-n", "2",
                "--out", str(out_dir),
            ]
        )
        == 0
    )
    for filename in (
        "m3_key_numbers.csv",
        "m3_residual_segments.csv",
        "m3_ablation_compact.csv",
        "residual_run_manifest.json",
        "model_actual_vs_predicted.png",
        "model_residuals.png",
        "model_distribution_actual_vs_predicted.png",
        "model_ablation_comparison.png",
    ):
        target = out_dir / filename
        assert target.is_file() and target.stat().st_size > 0
    compact = pd.read_csv(out_dir / "m3_ablation_compact.csv")
    assert len(compact) == 2 and "train_r2" in compact.columns and "cv_r2_mean" in compact.columns
