"""Kiểm thử M2-03 · correlation_engine (tiêu chí Gate 1)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.analysis.correlation_engine import (
    RESULT_COLUMNS,
    compute_series_correlation,
    correlate_by_group,
    correlate_pairs,
    fisher_ci,
    partial_correlation,
)

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "src" / "analysis"


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_perfect_linear_relation_gives_one_for_both():
    x = pd.Series(np.arange(50, dtype=float))
    res = compute_series_correlation(x, 2 * x + 1, "x", "y")
    assert res["pearson_r"] == pytest.approx(1.0)
    assert res["spearman_rho"] == pytest.approx(1.0)
    assert res["n_samples"] == 50
    assert res["recommended_method"] == "pearson"


def test_monotonic_nonlinear_relation_has_positive_delta():
    x = pd.Series(np.linspace(0, 10, 200))
    res = compute_series_correlation(x, np.exp(x), "x", "exp_x")
    assert res["spearman_rho"] == pytest.approx(1.0)
    assert res["pearson_r"] < 0.9
    assert res["delta_abs"] > 0.1
    assert res["recommended_method"] == "spearman"


def test_single_leverage_point_inflates_pearson_only():
    rng = _rng(0)
    x = np.append(rng.normal(size=200), 100.0)
    y = np.append(rng.normal(size=200), 100.0)
    res = compute_series_correlation(pd.Series(x), pd.Series(y))
    assert res["pearson_r"] > 0.9
    assert abs(res["spearman_rho"]) < 0.3


def test_pairs_are_aligned_on_index_not_position():
    x = pd.Series(np.arange(10, dtype=float), index=range(0, 10))
    y = pd.Series(np.arange(10, dtype=float) ** 2, index=range(5, 15))
    res = compute_series_correlation(x, y, min_n=4)
    assert res["n_samples"] == 5
    # Cặp chung: x = 5..9 ghép với y = 0², 1², ..., 4² (theo nhãn index)
    expected = stats.pearsonr(np.arange(5, 10), np.arange(5) ** 2)[0]
    assert res["pearson_r"] == pytest.approx(expected)


def test_nan_and_infinite_pairs_are_dropped_and_counted():
    x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, np.inf, 7.0, 8.0, 9.0, 10.0])
    y = pd.Series([2.0, 4.0, 6.0, np.nan, 10.0, 12.0, 14.0, -np.inf, 18.0, 20.0])
    res = compute_series_correlation(x, y, min_n=4)
    assert res["n_samples"] == 6
    assert res["n_dropped"] == 4
    assert res["pearson_r"] == pytest.approx(1.0)


def test_constant_input_returns_nan_with_note():
    x = pd.Series(np.ones(40))
    res = compute_series_correlation(x, pd.Series(np.arange(40.0)))
    assert np.isnan(res["pearson_r"]) and np.isnan(res["spearman_rho"])
    assert "hằng" in res["note"]


def test_small_sample_below_min_n_is_not_reported():
    x = pd.Series(np.arange(20.0))
    res = compute_series_correlation(x, x * 3, min_n=30)
    assert np.isnan(res["pearson_r"])
    assert res["n_samples"] == 20
    assert "ngưỡng" in res["note"]


def test_matches_scipy_on_random_data():
    rng = _rng(1)
    x = rng.normal(size=1_000)
    y = 0.4 * x + rng.normal(size=1_000)
    res = compute_series_correlation(pd.Series(x), pd.Series(y), include_kendall=True)
    r, p = stats.pearsonr(x, y)
    rho, p_s = stats.spearmanr(x, y)
    tau, p_t = stats.kendalltau(x, y)
    assert res["pearson_r"] == pytest.approx(r, abs=1e-12)
    assert res["pearson_p"] == pytest.approx(p, rel=1e-9)
    assert res["spearman_rho"] == pytest.approx(rho, abs=1e-12)
    assert res["spearman_p"] == pytest.approx(p_s, rel=1e-9)
    assert res["kendall_tau"] == pytest.approx(tau, abs=1e-12)
    assert res["kendall_p"] == pytest.approx(p_t, rel=1e-9)


def test_confidence_intervals_contain_estimate_and_shrink_with_n():
    rng = _rng(2)
    widths = []
    for n in (100, 10_000):
        x = rng.normal(size=n)
        y = 0.5 * x + rng.normal(size=n)
        res = compute_series_correlation(pd.Series(x), pd.Series(y))
        for key in ("pearson", "spearman"):
            low, high = res[f"{key}_ci_low"], res[f"{key}_ci_high"]
            estimate = res["pearson_r"] if key == "pearson" else res["spearman_rho"]
            assert -1 <= low <= estimate <= high <= 1
        widths.append(res["pearson_ci_high"] - res["pearson_ci_low"])
    assert widths[1] < widths[0]


def test_fisher_ci_matches_closed_form():
    low, high = fisher_ci(0.3, 103)
    z, half = np.arctanh(0.3), stats.norm.ppf(0.975) / np.sqrt(100)
    assert low == pytest.approx(np.tanh(z - half))
    assert high == pytest.approx(np.tanh(z + half))
    assert all(np.isnan(fisher_ci(0.3, 3)))


def test_boolean_input_equals_point_biserial():
    rng = _rng(3)
    flag = pd.Series(rng.random(500) < 0.3)
    y = pd.Series(rng.normal(size=500) + flag.to_numpy() * 0.8)
    res = compute_series_correlation(flag, y)
    expected = stats.pointbiserialr(flag.to_numpy(), y.to_numpy())[0]
    assert res["pearson_r"] == pytest.approx(expected, abs=1e-12)


def test_string_input_is_rejected():
    labels = pd.Series(["a", "b", "c"] * 20, dtype="string")
    with pytest.raises(TypeError):
        compute_series_correlation(labels, pd.Series(np.arange(60.0)))


def test_duplicate_index_without_identical_index_is_rejected():
    x = pd.Series(np.arange(40.0), index=[0, 0] + list(range(2, 40)))
    y = pd.Series(np.arange(40.0), index=range(1, 41))
    with pytest.raises(ValueError):
        compute_series_correlation(x, y)


def test_spearman_is_invariant_to_log_but_pearson_is_not():
    rng = _rng(4)
    x = pd.Series(rng.uniform(0, 20, 2_000))
    price = pd.Series(np.exp(10 - 0.1 * x + rng.normal(0, 0.3, 2_000)))
    raw = compute_series_correlation(x, price)
    logged = compute_series_correlation(x, np.log(price))
    assert raw["spearman_rho"] == pytest.approx(logged["spearman_rho"], abs=1e-12)
    assert abs(logged["pearson_r"]) > abs(raw["pearson_r"])


def test_correlate_pairs_and_groups_return_long_table_with_context():
    rng = _rng(5)
    n = 600
    frame = pd.DataFrame(
        {
            "brand": pd.Series(np.where(np.arange(n) < 550, "big", "small"), dtype="string"),
            "car_age": rng.integers(1, 20, n).astype(float),
        }
    )
    frame["log_price"] = 10 - 0.08 * frame["car_age"] + rng.normal(0, 0.2, n)
    table = correlate_pairs(frame, [("car_age", "log_price"), ("car_age", "missing_col")], dataset_version="t")
    assert list(table.columns) == list(RESULT_COLUMNS)
    assert table.loc[0, "dataset_version"] == "t" and table.loc[0, "scope"] == "all"
    assert "Thiếu cột" in table.loc[1, "note"]
    groups = correlate_by_group(frame, [("car_age", "log_price")], "brand", min_n=100)
    by_group = groups.set_index("group_value")
    assert by_group.loc["big", "n_samples"] == 550 and by_group.loc["big", "pearson_r"] < -0.5
    assert by_group.loc["small", "n_samples"] == 50 and np.isnan(by_group.loc["small", "pearson_r"])


def test_partial_correlation_removes_shared_driver():
    rng = _rng(6)
    n = 5_000
    z = rng.normal(size=n)
    frame = pd.DataFrame({"z": z, "x": z + rng.normal(size=n), "y": z + rng.normal(size=n)})
    marginal = compute_series_correlation(frame["x"], frame["y"])
    partial = partial_correlation(frame, "x", "y", ["z"], method="pearson")
    assert marginal["pearson_r"] > 0.4
    assert abs(partial["partial_r"]) < 0.05
    assert partial["ci_low"] < partial["partial_r"] < partial["ci_high"]
    assert partial["n_samples"] == n
    # Công thức đóng cho một biến kiểm soát
    r_xy = stats.pearsonr(frame["x"], frame["y"])[0]
    r_xz = stats.pearsonr(frame["x"], frame["z"])[0]
    r_yz = stats.pearsonr(frame["y"], frame["z"])[0]
    closed = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    assert partial["partial_r"] == pytest.approx(closed, abs=1e-10)
    spearman = partial_correlation(frame, "x", "y", ["z"])
    assert spearman["method"] == "spearman" and abs(spearman["partial_r"]) < 0.06


def test_no_dataframe_corr_call_in_analysis_package():
    pattern = ".corr" + "("
    offenders = [
        str(path.relative_to(ANALYSIS_DIR))
        for path in ANALYSIS_DIR.rglob("*")
        if path.suffix in {".py", ".md"} and pattern in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
