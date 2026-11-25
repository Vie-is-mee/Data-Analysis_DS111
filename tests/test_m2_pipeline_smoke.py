"""Smoke test M2 · run_m2 trên dữ liệu giả lập đúng hợp đồng (không dùng dữ liệu thật)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import bi_export, run_m2
from src.analysis.key_numbers import compare_key_numbers
from src.core.bootstrap_and_validate import generate_mock_clean_sample


@pytest.fixture(scope="module")
def abt_path(tmp_path_factory):
    root = tmp_path_factory.mktemp("m2_smoke")
    frame = generate_mock_clean_sample(root / "missing.csv", root / "mock.parquet", 1500)
    rng = np.random.default_rng(7)
    frame["condition_missing"] = rng.random(len(frame)) < 0.4
    frame.loc[frame["condition_missing"], "condition_clean"] = "unknown"
    frame["cylinders_was_missing"] = rng.random(len(frame)) < 0.25
    frame["seller_type"] = pd.Series(rng.choice(["dealer", "private"], len(frame), p=[0.7, 0.3]), dtype="string")
    frame["VIN"] = pd.Series("khong-phai-vin-that", index=frame.index, dtype="string")
    path = root / "abt.parquet"
    frame.to_parquet(path, index=False)
    pd.DataFrame({
        "step": ["01_load", "02_missing"], "sheet": ["00_Log", "02_Missing"], "action": ["Đọc", "Lọc"],
        "reason": ["", ""], "rows_before": [2000, 2000], "rows_after": [2000, 1500],
    }).to_csv(root / "log.csv", index=False)
    return path


def _run(abt_path, out, *extra):
    code = run_m2.main(["--abt", str(abt_path), "--dataset-version", "smoke", "--out", str(out),
                        "--transform-log", str(abt_path.parent / "log.csv"), *extra])
    assert code == 0
    return out


def test_run_m2_writes_all_outputs(abt_path, tmp_path):
    out = _run(abt_path, tmp_path / "full", "--bi")
    for name in ("correlation_results", "partial_correlations", "group_depreciation", "median_curves",
                 "categorical_association", "level_summaries", "common_scale_ranking", "eda_numeric",
                 "eda_categorical", "cleaning_funnel", "m2_key_numbers", "feature_ranking_m2",
                 "confounding_condition_by_seller", "confounding_cylinders_price"):
        assert (out / f"{name}.csv").is_file(), name
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == "smoke" and "VIN" not in manifest["abt_columns"]
    assert not [step for step in manifest["steps"] if step["status"] == "FAIL"]
    assert len(list((out / "figures").glob("eda_*.png"))) >= 10
    key = pd.read_csv(out / "m2_key_numbers.csv")
    assert set(key["dataset_version"]) == {"smoke"} and key["key"].is_unique
    assert np.isfinite(key.set_index("key").loc["rho_car_age_log_price", "value"])
    tables = {name: pd.read_parquet(out / "bi" / f"{name}.parquet")
              for name in ("Fact_Listings", *bi_export.DIM_KEYS)}
    assert bi_export.validate_star_schema(tables) == []
    assert "VIN" not in tables["Fact_Listings"].columns
    parity = pd.read_csv(out / "bi" / "dax_parity_check.csv")
    assert parity.loc[parity["filter_name"] == "Toàn bộ", "n_python"].iloc[0] == len(tables["Fact_Listings"])


def test_run_m2_is_reproducible(abt_path, tmp_path):
    first = _run(abt_path, tmp_path / "a", "--no-figures")
    second = _run(abt_path, tmp_path / "b", "--no-figures")
    diff = compare_key_numbers(pd.read_csv(first / "m2_key_numbers.csv"), pd.read_csv(second / "m2_key_numbers.csv"))
    assert not diff["changed"].any()


def test_parity_values_match_scipy(abt_path):
    from scipy import stats

    fact = pd.read_parquet(abt_path)
    expected = bi_export.dax_parity_expected(fact, (("Toàn bộ", {}),)).iloc[0]
    r = stats.pearsonr(fact["car_age"], fact["log_price"])[0]
    assert expected["pearson_age_logprice_python"] == pytest.approx(r, abs=1e-12)
