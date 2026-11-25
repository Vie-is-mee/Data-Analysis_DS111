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
        "title": "2015 Toyota Camry",
        "VIN": pd.NA,
        "address_country": "US",
        "odometer_broken": 0,
        "odometer_rolled_over": 0,
        "seller_type": rng.choice(["dealer", "owner", None], n),
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


def test_m1_new_cleaning_rules(tmp_path):
    raw = _fake_raw()
    ids = raw["id"]
    # Hãng thiếu → cứu từ title, kể cả hãng ngoài 42 hãng và hãng 2 từ.
    raw.loc[100, ["manufacturer", "title"]] = [pd.NA, "2019 Maserati Ghibli S"]
    raw.loc[101, ["manufacturer", "title"]] = [pd.NA, "2016 Land Rover Range Rover"]
    raw.loc[102, ["manufacturer", "title"]] = [pd.NA, "Smart key remote for sale"]
    # Đồng hồ hỏng, tin Canada.
    raw.loc[110, "odometer_broken"] = 1
    raw.loc[111, "odometer_rolled_over"] = 1
    raw.loc[112, "address_country"] = "CA"
    # Cùng VIN, tin 121 mới hơn → giữ 121, bỏ 120.
    vin = "1M8GDM9AXKP042788"
    raw.loc[[120, 121], "VIN"] = vin
    raw.loc[120, "posting_date"] = "2021-05-01T10:00:00-0500"
    raw.loc[121, "posting_date"] = "2021-05-09T10:00:00-0500"
    for r in (100, 101, 102, 110, 111, 112, 120, 121):
        raw.loc[r, ["price", "year", "odometer"]] = [15_000, 2015, 50_000]
    raw_csv = tmp_path / "vehicles.csv"
    raw.to_csv(raw_csv, index=False)

    data = build_clean_dataset(load_raw(raw_csv), dedup_vin=True).data.set_index("listing_id")
    assert data.loc[ids[100], "manufacturer_clean"] == "maserati"
    assert data.loc[ids[101], "manufacturer_clean"] == "land rover"
    assert data.loc[ids[102], "manufacturer_clean"] == "unknown"
    assert not {ids[110], ids[111], ids[112], ids[120]} & set(data.index)
    assert ids[121] in data.index

    # Không bật cờ → giữ cả hai tin trùng VIN.
    kept = build_clean_dataset(load_raw(raw_csv)).data["listing_id"]
    assert {ids[120], ids[121]} <= set(kept)


def test_dedup_vin_keeps_valid_older_listing_when_newest_has_no_price(tmp_path):
    raw = _fake_raw()
    ids = raw["id"]
    raw.loc[[200, 201], "VIN"] = "1HGCM82633A004352"
    raw.loc[[200, 201], ["year", "odometer"]] = [2015, 50_000]
    raw.loc[200, ["posting_date", "price"]] = ["2021-05-01T10:00:00-0500", 15_000]
    raw.loc[201, ["posting_date", "price"]] = ["2021-05-09T10:00:00-0500", 0]  # "gọi để biết giá"
    raw_csv = tmp_path / "vehicles.csv"
    raw.to_csv(raw_csv, index=False)
    kept = set(build_clean_dataset(load_raw(raw_csv), dedup_vin=True).data["listing_id"])
    assert ids[200] in kept and ids[201] not in kept


def test_price_iqr_flags_outliers_without_dropping_rows(tmp_path):
    raw = _fake_raw(1_500, seed=1)
    raw["manufacturer"] = "ford"
    raw["price"] = np.random.default_rng(1).integers(10_000, 20_000, len(raw))
    raw.loc[:4, "price"] = 140_000   # trong miền tầng 1 nhưng bất thường với hãng
    raw_csv = tmp_path / "vehicles.csv"
    raw.to_csv(raw_csv, index=False)
    data = build_clean_dataset(load_raw(raw_csv)).data.set_index("listing_id")
    # TV3 chốt: chỉ gắn cờ, không xóa dòng.
    assert set(raw.loc[:4, "id"]) <= set(data.index)
    assert data.loc[raw.loc[:4, "id"], "price_outlier_iqr"].all()
    assert data["price_outlier_iqr"].mean() < 0.05


def test_iqr_bounds_learned_on_train_only(tmp_path):
    """Đổi giá của dòng val/test không được làm đổi cờ của dòng train."""
    raw = _fake_raw(1_500, seed=2)
    raw["manufacturer"] = "ford"
    raw_csv = tmp_path / "vehicles.csv"
    raw.to_csv(raw_csv, index=False)
    base = build_clean_dataset(load_raw(raw_csv)).data.set_index("listing_id")
    holdout = base.index[base["split_set"].ne("train")]
    raw.loc[raw["id"].isin(holdout), "price"] = 149_000
    raw.to_csv(raw_csv, index=False)
    again = build_clean_dataset(load_raw(raw_csv)).data.set_index("listing_id")
    train = base.index[base["split_set"].eq("train")]
    assert again.loc[train, "price_outlier_iqr"].equals(base.loc[train, "price_outlier_iqr"])


def test_condition_missing_and_manufacturer_source(tmp_path):
    raw = _fake_raw()
    ids = raw["id"]
    raw.loc[200, "condition"] = pd.NA          # thiếu thật
    raw.loc[201, "condition"] = "mint"         # nhãn lạ → unknown nhưng KHÔNG thiếu
    raw.loc[202, ["manufacturer", "title"]] = [pd.NA, "2019 Maserati Ghibli S"]
    raw.loc[203, ["manufacturer", "title"]] = ["ford", "2015 Toyota Camry"]     # mâu thuẫn
    raw.loc[204, ["manufacturer", "title"]] = ["rover", "2016 Land Rover LR4"]  # alias, không mâu thuẫn
    raw.loc[205, ["manufacturer", "title"]] = [pd.NA, "Smart key remote"]
    raw.loc[206, ["manufacturer", "title"]] = ["ram", "2008 Dodge Ram 1500"]  # cùng gia đình hãng
    for r in range(200, 207):
        raw.loc[r, ["price", "year", "odometer"]] = [15_000, 2015, 50_000]
    raw_csv = tmp_path / "vehicles.csv"
    raw.to_csv(raw_csv, index=False)
    data = build_clean_dataset(load_raw(raw_csv)).data.set_index("listing_id")

    assert data.loc[ids[200], "condition_missing"]
    assert data.loc[ids[201], ["condition_missing", "condition_clean"]].tolist() == [False, "unknown"]
    assert data.loc[ids[202], "manufacturer_source"] == "title"
    assert data.loc[ids[203], ["manufacturer_source", "manufacturer_title_conflict"]].tolist() == ["original", True]
    assert not data.loc[ids[204], "manufacturer_title_conflict"]
    assert not data.loc[ids[206], "manufacturer_title_conflict"]
    assert data.loc[ids[205], "manufacturer_source"] == "none"
    assert set(data["manufacturer_source"]) <= {"original", "title", "none"}


def test_contract_11_extra_columns(tmp_path):
    """DATA_CONTRACT 1.1: cờ boolean không rỗng, seller_type private/dealer/unknown."""
    raw = _fake_raw()
    raw.loc[300, ["cylinders", "fuel"]] = [pd.NA, "electric"]   # điện → 0, không tính khuyết
    raw.loc[301, ["cylinders", "fuel"]] = ["other", "gas"]      # không tách được số → khuyết
    for r in (300, 301):
        raw.loc[r, ["price", "year", "odometer"]] = [15_000, 2015, 50_000]
    raw_csv = tmp_path / "vehicles.csv"
    raw.to_csv(raw_csv, index=False)
    data = build_clean_dataset(load_raw(raw_csv)).data.set_index("listing_id")
    for flag in ("condition_missing", "cylinders_was_missing", "manufacturer_title_conflict",
                 "price_outlier_iqr"):
        assert data[flag].dtype == bool and data[flag].notna().all(), flag
    assert set(data["seller_type"]) <= {"private", "dealer", "unknown"}
    assert "owner" not in set(data["seller_type"]) and "private" in set(data["seller_type"])
    ids = raw["id"]
    assert data.loc[ids[300], ["cylinders_num", "cylinders_was_missing"]].tolist() == [0.0, False]
    assert data.loc[ids[301], "cylinders_was_missing"]
    assert data["cylinders_num"].notna().all()
    assert (data["car_age_squared"] == data["car_age"].astype(float) ** 2).all()


def test_raw_profile_has_decision_columns(tmp_path):
    raw_csv = tmp_path / "vehicles.csv"
    _fake_raw(50).to_csv(raw_csv, index=False)
    raw = _fake_raw(50)
    raw["description"] = "CALL OR TEXT: (555) 555-0142"
    raw.to_csv(raw_csv, index=False)
    profile = profile_raw(load_raw(raw_csv, usecols=None)).set_index("column")
    assert {"keep_drop", "reason", "missing_pct"} <= set(profile.columns)
    assert set(PIPELINE_COLUMNS) <= set(profile.index)
    # Sheet 01 lên Git → không được lộ SĐT/URL/VIN trong cột ví dụ.
    assert "555-0142" not in profile.loc["description", "example_values"]
    assert "example.invalid" not in profile.loc["url", "example_values"]
