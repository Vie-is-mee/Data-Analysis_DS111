"""VIN selection and its production manifest contract."""

import json
import sys

import pandas as pd
import pytest

from src.core.vin_reposts import GROUP_COLUMN, resolve_vin_reposts
from src.models.split_manifest import main, validate_split_manifest


OFFICIAL_EXAMPLE = "1M8GDM9AXKP042788"
WRONG_CHECK_DIGIT = "1M8GDM9A0KP042788"


def test_reposts_choose_latest_independently_of_price_and_input_order():
    listings = pd.DataFrame(
        {
            "listing_id": [31, 33, 34, 32, 40, 41, 42, 43, 44, 45, 46, 47, 48],
            "VIN": [
                OFFICIAL_EXAMPLE,
                f" {OFFICIAL_EXAMPLE.lower()} ",
                OFFICIAL_EXAMPLE,
                OFFICIAL_EXAMPLE,
                WRONG_CHECK_DIGIT,
                WRONG_CHECK_DIGIT,
                None,
                " ",
                "0" * 17,
                "0" * 17,
                "12345678901234567",
                "O" + OFFICIAL_EXAMPLE[1:],
                "OLD-CAR",
            ],
            "posting_date": [
                "2026-10-21T12:00:00-07:00",
                "2026-10-21T14:00:00-07:00",
                "2026-10-21T21:00:00Z",
                "unparseable",
                "2026-10-21",
                "2026-11-01",
                *[None] * 7,
            ],
            "price": [1] * 13,
        }
    )
    decisions = resolve_vin_reposts(listings).set_index("listing_id")
    assert decisions.index[~decisions["keep_for_abt"]].tolist() == [31, 32, 33]
    assert decisions.loc[31:34, GROUP_COLUMN].nunique() == 1
    assert decisions.loc[40:41, GROUP_COLUMN].nunique() == 1
    assert decisions.loc[42:48, GROUP_COLUMN].nunique() == 7
    assert decisions.loc[40:48, "vin_status"].tolist() == [
        "checksum_unverified",
        "checksum_unverified",
        "missing",
        "missing",
        "placeholder",
        "placeholder",
        "placeholder",
        "malformed",
        "malformed",
    ]
    shuffled = listings.assign(price=80_000).sample(frac=1, random_state=42)
    pd.testing.assert_frame_equal(
        decisions, resolve_vin_reposts(shuffled).set_index("listing_id")
    )
    assert set(decisions.columns) == {GROUP_COLUMN, "keep_for_abt", "vin_status"}
    with pytest.raises(ValueError, match="unique int64"):
        resolve_vin_reposts(pd.concat([listings, listings.iloc[:1]]))


def test_cli_builds_vin_manifest_without_exporting_source_metadata(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "private_input.parquet"
    output = tmp_path / "manifest.parquet"
    listings = pd.DataFrame(
        {
            "listing_id": list(range(1, 25)),
            "VIN": [OFFICIAL_EXAMPLE] * 3 + [WRONG_CHECK_DIGIT] * 3 + [None] * 18,
            "posting_date": ["2026-10-21"] * 24,
            "price": [15_000] * 24,
        }
    )
    listings.to_parquet(source, index=False)
    monkeypatch.setattr(
        sys, "argv", ["split_manifest", "--vin-listings", str(source), "--output", str(output)]
    )
    main()
    captured = capsys.readouterr().out
    audit = json.loads(captured)
    assert audit["rows_removed"] == 2
    assert audit["rows_kept"] == 22
    assert audit["retained_multi_listing_groups"] == 1
    assert OFFICIAL_EXAMPLE not in captured and WRONG_CHECK_DIGIT not in captured

    manifest = pd.read_parquet(output)
    assert set(manifest.columns) == {"listing_id", GROUP_COLUMN, "split_set"}
    assert manifest["listing_id"].tolist() == list(range(3, 25))
    assert manifest.loc[manifest["listing_id"].between(4, 6), "split_set"].nunique() == 1
    validate_split_manifest(manifest, manifest["listing_id"])
    with pytest.raises(SystemExit):
        main()
