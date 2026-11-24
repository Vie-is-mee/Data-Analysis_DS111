"""Resolve reposts from listing metadata without reading prices or exporting VINs."""

import re

import numpy as np
import pandas as pd


GROUP_COLUMN = "vehicle_group_id"
VIN_PATTERN = r"[A-HJ-NPR-Z0-9]{17}"
PLACEHOLDER_PATTERN = re.compile(r"^(.)\1{16}$")
SEQUENTIAL_PLACEHOLDERS = {"12345678901234567", "01234567890123456", "98765432109876543"}

# 49 CFR 565.15(c): transliteration and position weights, including position 9.
VIN_VALUES = {str(number): number for number in range(10)}
VIN_LETTERS = ("", "AJ", "BKS", "CLT", "DMU", "ENV", "FW", "GPX", "HY", "RZ")
VIN_VALUES.update(
    {
        letter: value
        for value, letters in enumerate(VIN_LETTERS)
        for letter in letters
    }
)
VIN_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)


def _passes_check_digit(vin: str) -> bool:
    total = sum(VIN_VALUES[char] * weight for char, weight in zip(vin, VIN_WEIGHTS))
    return vin[8] == "0123456789X"[total % 11]


def resolve_vin_reposts(listings: pd.DataFrame) -> pd.DataFrame:
    """Return one decision per listing, sorted by ID, with an opaque group key.

    VINs that look usable but fail the checksum still share a group on an exact
    match. Keep all of those rows. Give missing, malformed and placeholder
    values separate listing-level keys because their vehicle is unknown.
    """
    frame = listings[["listing_id", "VIN", "posting_date"]].copy()
    ids = frame["listing_id"]
    if ids.dtype != np.dtype("int64") or not ids.is_unique:
        raise ValueError("VIN handoff requires unique int64 listing_id values.")
    frame = frame.sort_values("listing_id").reset_index(drop=True)
    ids = frame["listing_id"]

    raw_vin = frame["VIN"].astype("string").str.strip()
    vin = raw_vin.where(raw_vin.str.fullmatch(r"[A-Za-z0-9]{17}", na=False)).str.upper()
    missing = raw_vin.isna() | raw_vin.eq("").fillna(False)
    placeholder = vin.str.fullmatch(PLACEHOLDER_PATTERN, na=False) | vin.isin(
        SEQUENTIAL_PLACEHOLDERS
    )
    plausible = vin.str.fullmatch(VIN_PATTERN, na=False) & ~placeholder
    candidates = frame.loc[plausible, ["listing_id"]].assign(vin=vin.loc[plausible])
    checks = {value: _passes_check_digit(value) for value in candidates["vin"].unique()}
    verified = vin.map(checks).eq(True)

    group_ids = ids.copy()
    group_ids.loc[plausible] = candidates.groupby("vin")["listing_id"].transform("min")
    timestamps = pd.to_datetime(
        frame["posting_date"].astype("string"), format="ISO8601", errors="coerce", utc=True
    )
    ordered = (
        frame.loc[verified, ["listing_id"]]
        .assign(vin=vin.loc[verified], timestamp=timestamps.loc[verified])
        .sort_values(["timestamp", "listing_id"], ascending=False, na_position="last")
    )
    discard = ordered.index[ordered["vin"].duplicated()]
    result = pd.DataFrame(
        {
            "listing_id": ids,
            GROUP_COLUMN: "listing:" + group_ids.astype("string"),
            "keep_for_abt": ~frame.index.isin(discard),
            "vin_status": pd.Series("malformed", index=frame.index, dtype="string"),
        }
    )
    result.loc[missing, "vin_status"] = "missing"
    result.loc[placeholder, "vin_status"] = "placeholder"
    result.loc[plausible, "vin_status"] = "checksum_unverified"
    result.loc[verified, "vin_status"] = "checksum_pass"
    return result
