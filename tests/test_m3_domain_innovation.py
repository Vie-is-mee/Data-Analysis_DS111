"""Kiểm thử tính đúng đắn và chống rò rỉ cho phương pháp sáng tạo Exp5 TV3."""

import numpy as np
import pandas as pd
import pytest

from src.models.ablation_runner import AblationExperimentSuite
from src.models.domain_features import (
    WEAR_ADJ_HIER_FEATURES,
    WearAdjustedHierarchicalEncoder,
    add_domain_depreciation_features,
    extract_model_stem,
)
from src.models.metrics import (
    compute_duan_smearing_factor,
    evaluate_regression,
    fit_binned_duan_smearing,
    inverse_log_transform,
    inverse_log_transform_binned,
)


def test_extract_model_stem_and_domain_tokens():
    frame = pd.DataFrame(
        {
            "car_age": [2, 18, 35],
            "model_clean": [
                "f-150 xlt 4x4 supercrew",
                "cr-v ex-l hybrid",
                "911 carrera cabriolet",
            ],
        }
    )
    enriched = add_domain_depreciation_features(frame)
    assert extract_model_stem(frame["model_clean"]).tolist() == ["f150", "crv", "911"]
    assert enriched["age_hinge_15"].tolist() == [0.0, 3.0, 20.0]
    assert enriched["age_hinge_25"].tolist() == [0.0, 0.0, 10.0]
    assert enriched["tok_awd_4wd"].tolist() == [1.0, 0.0, 0.0]
    assert enriched["tok_hybrid_ev"].tolist() == [0.0, 1.0, 0.0]
    assert enriched["tok_performance"].tolist() == [0.0, 0.0, 1.0]


def test_wear_adjusted_hierarchical_encoder_backoff_and_zero_group_leakage():
    rng = np.random.default_rng(7)
    rows = []
    # Two brands with distinct wear-adjusted premiums: porsche (+0.9) vs nissan (-0.5)
    for i in range(60):
        is_porsche = i < 30
        age = int(rng.integers(1, 20))
        odo = float(np.log1p(age * 10_000 + rng.integers(1_000, 15_000)))
        model = "911 carrera" if is_porsche else "versa s"
        if i in (0, 1):
            model = "911 rare_trim"
        premium = 0.9 if is_porsche else -0.5
        log_p = 10.2 - 0.06 * age - 0.1 * odo + premium
        rows.append(
            {
                "manufacturer_clean": "porsche" if is_porsche else "nissan",
                "type": "coupe" if is_porsche else "sedan",
                "model_clean": model,
                "state": "ca" if i % 2 == 0 else "tx",
                "region": "sf bay" if i % 2 == 0 else "dallas",
                "car_age": age,
                "log_odometer": odo,
                "cylinders_num": 6.0 if i % 2 == 0 else 4.0,
                "group": f"g:{i // 2}",
                "y": log_p,
            }
        )
    df = add_domain_depreciation_features(pd.DataFrame(rows))
    y = df["y"].to_numpy()
    groups = df["group"]

    enc = WearAdjustedHierarchicalEncoder(
        mode="wear_adjusted_hierarchical", n_splits=5, random_state=42
    )
    oof_a = enc.fit_transform(df.drop(columns=["group", "y"]), y, groups=groups)
    assert set(WEAR_ADJ_HIER_FEATURES).issubset(oof_a.columns)

    # Perturbing target of group g:0 (rows 0 and 1) MUST NOT change rows 0 and 1 OOF features!
    y_perturbed = y.copy()
    y_perturbed[0] += 5.0
    y_perturbed[1] += 5.0
    enc_b = WearAdjustedHierarchicalEncoder(
        mode="wear_adjusted_hierarchical", n_splits=5, random_state=42
    )
    oof_b = enc_b.fit_transform(df.drop(columns=["group", "y"]), y_perturbed, groups=groups)
    np.testing.assert_allclose(
        oof_a.loc[[0, 1], list(WEAR_ADJ_HIER_FEATURES)].to_numpy(),
        oof_b.loc[[0, 1], list(WEAR_ADJ_HIER_FEATURES)].to_numpy(),
    )

    # Hierarchical backoff: unseen porsche 911 trim inherits strong positive porsche/911 premium,
    # NOT collapsing to global mean (~0) like flat TE would!
    unseen = add_domain_depreciation_features(
        pd.DataFrame(
            [
                {
                    "manufacturer_clean": "porsche",
                    "type": "coupe",
                    "model_clean": "911 gt3 touring unseen",
                    "state": "ca",
                    "region": "unknown_city",
                    "car_age": 5,
                    "log_odometer": np.log1p(25_000),
                    "cylinders_num": 6.0,
                },
                {
                    "manufacturer_clean": "nissan",
                    "type": "sedan",
                    "model_clean": "versa unseen trim",
                    "state": "tx",
                    "region": "unknown_city",
                    "car_age": 5,
                    "log_odometer": np.log1p(25_000),
                    "cylinders_num": 4.0,
                },
            ]
        )
    )
    out = enc.transform(unseen)
    assert out.loc[0, "wear_adj_model_premium"] > 0.35
    assert out.loc[1, "wear_adj_model_premium"] < -0.25
    assert out.loc[0, "wear_adj_model_premium"] - out.loc[1, "wear_adj_model_premium"] > 0.70


def test_binned_duan_smearing_beats_global_under_heteroscedasticity():
    rng = np.random.default_rng(42)
    n = 4_000
    pred = rng.uniform(7.5, 11.2, size=n)
    # High noise at low prices (sigma=0.8), low noise at high prices (sigma=0.08)
    sigma = np.where(pred < 9.0, 0.85, 0.08)
    eps = rng.normal(0.0, sigma)
    actual_log = pred + eps
    actual_usd = np.exp(actual_log)

    global_fac = compute_duan_smearing_factor(actual_log[:2000], pred[:2000])
    binned = fit_binned_duan_smearing(actual_log[:2000], pred[:2000], n_bins=8)

    rep_global = evaluate_regression(
        actual_usd[2000:], inverse_log_transform(pred[2000:], global_fac), 1
    )
    rep_binned = evaluate_regression(
        actual_usd[2000:], inverse_log_transform_binned(pred[2000:], binned), 1
    )
    assert rep_binned.rmse < rep_global.rmse * 0.92


def test_exp5_runs_with_groups_and_isolates_test(tmp_path):
    from src.core.bootstrap_and_validate import generate_mock_clean_sample

    path = tmp_path / "mock.parquet"
    data = generate_mock_clean_sample(tmp_path / "absent.csv", path, n_samples=360)
    data["vehicle_group_id"] = "g:" + (data["listing_id"] // 2).astype(str)
    # Re-assign clean group splits
    from src.models.split_manifest import make_split_manifest

    manifest = make_split_manifest(data[["listing_id", "vehicle_group_id"]])
    data = data.drop(columns="split_set").merge(
        manifest[["listing_id", "split_set"]], on="listing_id"
    )
    suite = AblationExperimentSuite(
        data, fast_mode=True, dataset_version="mock-exp5", course_encoding=True
    )
    entries = suite.run_exp5_domain_innovation()
    assert [e.model_id for e in entries] == [
        "Exp5_Ridge_FlatOOF",
        "Exp5_Ridge_WearAdjHier",
        "Exp5_LightGBM_FlatOOF",
        "Exp5_LightGBM_WearAdjHier",
    ]
    assert entries[1].binned_smearing is not None
    assert entries[3].binned_smearing is not None
    winner = suite.evaluate_final()
    assert winner.test_metrics is not None and np.isfinite(winner.test_metrics.rmse)
