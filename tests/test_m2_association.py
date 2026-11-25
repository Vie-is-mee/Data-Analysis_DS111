"""Kiểm thử M2-18 · association (η², Kruskal–Wallis ε², Cramér's V, Cliff's δ, xếp hạng chung)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.analysis.association import (
    categorical_association_table,
    cliffs_delta,
    common_scale_ranking,
    cramers_v,
    eta_squared,
    kruskal_epsilon_squared,
    level_summary,
)


def _frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = rng.choice(["a", "b", "c", "d"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    shift = pd.Series(groups).map({"a": 0.0, "b": 0.3, "c": 0.6, "d": 1.0}).to_numpy()
    return pd.DataFrame({"g": pd.Series(groups, dtype="string"), "y": shift + rng.normal(0, 1, n),
                         "x": rng.normal(size=n)})


def test_eta_squared_equals_r2_of_dummy_regression():
    df = _frame()
    eta = eta_squared(df["y"], df["g"])
    dummies = pd.get_dummies(df["g"], drop_first=True, dtype=float).to_numpy()
    design = np.column_stack([np.ones(len(df)), dummies])
    beta, *_ = np.linalg.lstsq(design, df["y"].to_numpy(), rcond=None)
    resid = df["y"].to_numpy() - design @ beta
    r2 = 1 - resid @ resid / ((df["y"] - df["y"].mean()) ** 2).sum()
    assert eta["eta2"] == pytest.approx(r2, abs=1e-12)
    assert 0 < eta["omega2"] < eta["eta2"]
    assert eta["k"] == 4 and eta["n"] == len(df)


def test_kruskal_epsilon_squared_matches_scipy():
    df = _frame()
    res = kruskal_epsilon_squared(df["y"], df["g"])
    h, p = stats.kruskal(*[part.to_numpy() for _, part in df.groupby("g")["y"]])
    assert res["kw_h"] == pytest.approx(h)
    assert res["kw_p"] == pytest.approx(p)
    assert res["epsilon2"] == pytest.approx(h / (len(df) - 1))


def test_cramers_v_perfect_and_independent():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.choice(list("xyz"), 5000), dtype="string")
    renamed = pd.Series(a.map({"x": "p", "y": "q", "z": "r"}).to_numpy(), dtype="string")
    assert cramers_v(a, renamed)["cramers_v"] == pytest.approx(1.0, abs=1e-3)
    assert cramers_v(a, a, bias_correction=False)["cramers_v"] == pytest.approx(1.0)
    independent = cramers_v(a, pd.Series(rng.choice(list("uv"), 5000), dtype="string"))
    assert independent["cramers_v"] < 0.05


def test_cliffs_delta_known_values():
    res = cliffs_delta(pd.Series([1.0, 2.0, 3.0, 10.0, 11.0]), pd.Series([True, True, True, False, False]))
    assert res["cliffs_delta"] == pytest.approx(-1.0)
    tie = cliffs_delta(pd.Series([1.0, 2.0, 1.0, 2.0]), pd.Series([True, True, False, False]))
    assert tie["cliffs_delta"] == pytest.approx(0.0)


def test_level_summary_flags_order_break():
    df = pd.DataFrame({
        "price": [4000.0] * 5 + [3000.0] * 5 + [8000.0] * 5 + [16000.0] * 5,
        "condition_clean": pd.Series(["salvage"] * 5 + ["fair"] * 5 + ["good"] * 5 + ["unknown"] * 5,
                                     dtype="string"),
    })
    table = level_summary(df, "price", "condition_clean", order=["salvage", "fair", "good", "excellent"])
    assert table["level"].tolist() == ["salvage", "fair", "good", "unknown"]
    assert bool(table.loc[table["level"] == "fair", "breaks_order"].iloc[0]) is True
    assert bool(table.loc[table["level"] == "good", "breaks_order"].iloc[0]) is False
    assert not table["order_monotonic"].iloc[0]
    assert table["share_pct"].sum() == pytest.approx(100.0)


def test_categorical_table_and_common_scale_ranking_share_rows():
    df = _frame()
    df.loc[0, "x"] = np.nan
    table = categorical_association_table(df, "y", ["g", "missing"], dataset_version="t")
    assert table.loc[table["column"] == "g", "eta2"].iloc[0] > 0.05
    assert "Thiếu cột" in table.loc[table["column"] == "missing", "note"].iloc[0]
    ranking = common_scale_ranking(df, "y", ["x"], ["g"], dataset_version="t")
    assert set(ranking["n_samples"]) == {len(df) - 1}
    r = stats.pearsonr(df["x"].iloc[1:], df["y"].iloc[1:])[0]
    assert ranking.set_index("feature").loc["x", "share_explained"] == pytest.approx(r**2)
    assert ranking.iloc[0]["feature"] == "g"
