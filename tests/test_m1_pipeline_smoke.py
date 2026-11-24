"""Smoke test pipeline TV1 trên CSV giả lập có lỗi điển hình của Craigslist."""

import numpy as np
import pandas as pd

from src.core.bootstrap_and_validate import validate_handoff_dataset
from src.data.pipeline_m1 import build_clean_dataset, main
from src.data.raw_profile import PIPELINE_COLUMNS, load_raw, profile_raw


def _fake_raw(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    year = rng.integers(1995, 2022, n)
    df = pd.DataFrame({
        "id": np.arange(7_000_000_000, 7_000_000_000 + n),
        "region": rng.choice(["Boston", " seattle-tacoma ", None], n),
        "price": rng.integers(1_000, 60_000, n),
        "year": year,
        "manufacturer": rng.choice(["ford", "Chevy", "toyota", "VW", None], n),
        "model": rng.choice(["ford f-150", "camry", "civic", None], n),
        "condition": rng.choice(["good", "excellent", "like new", "mint", None], n),
        "cylinders": rng.choice(["4 cylinders", "6 cylinders", "other", None], n),
        "fuel": rng.choice(["gas", "diesel", "electric", None], n),
        "odometer": rng.integers(0, 250_000, n).astype(float),
        "title_status": rng.choice(["clean", "rebuilt", "weird", None], n),
        "transmission": rng.choice(["automatic", "manual"], n),
        "drive": rng.choice(["fwd", "4wd", None], n),
        "type": rng.choice(["sedan", "SUV", None], n),
        "state": rng.choice(["ma", "WA"], n),
        "posting_date": "2021-05-04T12:31:18-0500",
        "url": "https://example.invalid",
    })
    # Lỗi điển hình: giá 0/ảo, năm lạ, odometer khuyết, id trùng.
    df.loc[:9, "price"] = 0
    df.loc[10:14, "price"] = 1_234_567_890
    df.loc[15:19, "year"] = 1900
    df.loc[20:29, "odometer"] = np.nan
    df.loc[30, "id"] = df.loc[31, "id"]
    return df


def test_m1_pipeline_passes_handoff_validation(tmp_path):
    raw_csv = tmp_path / "vehicles.csv"
    _fake_raw().to_csv(raw_csv, index=False)
    out, store = tmp_path / "clean.parquet", tmp_path / "store.joblib"

    assert main(["--raw", str(raw_csv), "--out", str(out), "--store", str(store),
                 "--log", str(tmp_path / "log.csv")]) == 0

    report = validate_handoff_dataset(out, store)
    failed = [f"{c.name}: {c.detail}" for c in report.checks if not c.passed]
    assert report.passed, failed


def test_m1_cleaning_rules(tmp_path):
    raw_csv = tmp_path / "vehicles.csv"
    _fake_raw().to_csv(raw_csv, index=False)
    result = build_clean_dataset(load_raw(raw_csv))
    data = result.data

    assert data["price"].between(500, 150_000).all()
    assert data["year"].ge(1950).all()
    assert data["listing_id"].is_unique
    assert set(data["manufacturer_clean"]) <= {"ford", "chevrolet", "toyota", "volkswagen", "unknown"}
    assert "mint" not in set(data["condition_clean"])
    assert result.store.formula_count >= 3
    assert len(result.log) > 0 and result.log.to_dataframe()["reason"].str.len().gt(0).all()


def test_raw_profile_has_decision_columns(tmp_path):
    raw_csv = tmp_path / "vehicles.csv"
    _fake_raw(50).to_csv(raw_csv, index=False)
    profile = profile_raw(load_raw(raw_csv, usecols=None))
    assert {"keep_drop", "reason", "missing_pct"} <= set(profile.columns)
    assert set(PIPELINE_COLUMNS) <= set(profile["column"])
