"""Kiểm thử M2-06 / M2-19 / M2-20 · group_analysis, confounding và bảng tra hãng."""

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis import confounding
from src.analysis.group_analysis import (
    FOCUS_BRANDS,
    attach_brand_lookup,
    depreciation_table,
    fit_log_depreciation,
    load_brand_lookup,
    median_curve,
)

RAW_CRAIGSLIST_BRANDS = """acura alfa-romeo aston-martin audi bmw buick cadillac chevrolet chrysler datsun
dodge ferrari fiat ford gmc harley-davidson honda hyundai infiniti jaguar jeep kia lexus lincoln mazda
mercedes-benz mercury mini mitsubishi morgan nissan pontiac porsche ram rover saturn subaru tesla
toyota volkswagen volvo""".split() + ["land rover"]
RESCUED_BRANDS = ("maserati bentley lamborghini rolls-royce mclaren bugatti lotus genesis polestar rivian "
                  "lucid scion isuzu hummer oldsmobile plymouth saab suzuki smart").split()


def _synthetic(n: int = 8000, seed: int = 0, k: float = 0.10, c_odo: float = 0.04) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(1, 21, n)
    odometer = np.clip(age * 11_000 + rng.normal(0, 15_000, n), 0, None)
    log_price = np.log(30_000) - k * age - c_odo * odometer / 10_000 + rng.normal(0, 0.25, n)
    return pd.DataFrame({
        "car_age": age, "odometer": odometer, "log_price": log_price, "price": np.exp(log_price),
        "type": pd.Series(rng.choice(["sedan", "suv", "pickup"], n), dtype="string"),
        "brand": pd.Series(rng.choice(["a", "b"], n, p=[0.98, 0.02]), dtype="string"),
    })


def test_brand_lookup_covers_all_known_labels_once():
    lookup = load_brand_lookup()
    assert set(RAW_CRAIGSLIST_BRANDS) | set(RESCUED_BRANDS) | {"unknown"} <= set(lookup["manufacturer_clean"])
    focus = lookup[lookup["analysis_cluster"] != "other"]
    assert set(focus["manufacturer_clean"]) == set(FOCUS_BRANDS)
    df = pd.DataFrame({"manufacturer_clean": pd.Series(["toyota", "bmw", "newbrand"], dtype="string")})
    out, uncovered = attach_brand_lookup(df, lookup)
    assert out["analysis_cluster"].tolist() == ["JP", "DE", "other"] and uncovered == ["newbrand"]


def test_log_depreciation_recovers_known_rates():
    df = _synthetic()
    controlled = fit_log_depreciation(df, ("odometer",), ci_level=0.999)
    true_year, true_10k = 100 * (math.exp(-0.10) - 1), 100 * (math.exp(-0.04) - 1)
    assert controlled["pct_ci_low"] < true_year < controlled["pct_ci_high"]
    assert controlled["pct_10k_ci_low"] < true_10k < controlled["pct_10k_ci_high"]
    # Không kiểm soát số dặm thì tuổi xe "gánh" cả phần mất giá do số dặm → giảm nhanh hơn
    assert fit_log_depreciation(df, ())["pct_per_year"] < controlled["pct_per_year"]
    assert fit_log_depreciation(df, ("odometer", "type"))["n_params"] == 5


def test_depreciation_table_respects_min_n_window_and_missing_controls():
    df = _synthetic()
    df.loc[df.index[:50], "car_age"] = 30  # ngoài cửa sổ 1–20
    table = depreciation_table(df, "brand", ["a", "b"], specs=("none", "odometer"), min_n=200)
    small = table[table["group_value"] == "b"]
    assert small["slope_log"].isna().all() and small["note"].str.contains("không báo").all()
    big = table[(table["group_value"] == "a") & (table["controls"] == "none")]
    assert big["n"].iloc[0] == ((df["brand"] == "a") & df["car_age"].between(1, 20)).sum()
    missing = depreciation_table(df.drop(columns="type"), None, specs=("odometer_type",))
    assert missing["note"].iloc[0].startswith("Thiếu cột")


def test_median_curve_matches_groupby():
    df = pd.DataFrame({"car_age": [1, 1, 1, 2, 2, 2, 3], "price": [10.0, 20.0, 30.0, 5.0, 7.0, 9.0, 1.0]})
    curve = median_curve(df, min_n=2)
    assert curve["car_age"].tolist() == [1, 2]
    assert curve["median"].tolist() == [20.0, 7.0]
    assert curve["group_value"].unique().tolist() == ["all"]


def test_confounding_tables_on_small_frame():
    df = pd.DataFrame({
        "seller_type": pd.Series(["dealer"] * 6 + ["private"] * 4, dtype="string"),
        "condition_clean": pd.Series(["unknown"] * 4 + ["good"] * 2 + ["unknown"] + ["good"] * 3, dtype="string"),
        "condition_missing": [True] * 4 + [False] * 2 + [True] + [False] * 3,
        "car_age": [7] * 10,
        "price": [20000.0, 22000, 21000, 19000, 15000, 16000, 6000, 9000, 10000, 11000],
        "cylinders_was_missing": [True, False] * 5,
        "analysis_cluster": pd.Series(["JP"] * 5 + ["US"] * 5, dtype="string"),
        "type": pd.Series(["sedan", "suv"] * 5, dtype="string"),
    })
    shares = confounding.missingness_by(df, "condition_missing", ["seller_type"]).set_index("level")
    assert shares.loc["dealer", "flag_share_pct"] == pytest.approx(400 / 6)
    assert shares.loc["private", "flag_share_pct"] == pytest.approx(25.0)
    cell = confounding.condition_by_seller(df, age_bands=((6, 10),)).set_index(
        ["seller_type", "condition_clean"])["median_price"]
    assert cell[("dealer", "unknown")] == pytest.approx(20500.0)
    assert cell[("private", "unknown")] == pytest.approx(6000.0)
    cyl = confounding.cylinders_missing_vs_observed(df)
    assert cyl.loc[cyl["scope"] == "all", "missing_share_pct"].iloc[0] == pytest.approx(50.0)
    mix = confounding.type_mix_by_cluster(df)
    assert np.allclose(mix.groupby("cluster")["share_pct"].sum(), 100.0)
    with pytest.raises(KeyError):
        confounding.condition_by_seller(df.drop(columns="seller_type"))
