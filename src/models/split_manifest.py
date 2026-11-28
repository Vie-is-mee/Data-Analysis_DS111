"""Tạo và kiểm định bảng chia cố định do TV3 bàn giao."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from configs.project_config import RANDOM_STATE
from src.core.vin_reposts import GROUP_COLUMN, resolve_vin_reposts

PARTITIONS = {"train", "val", "test"}


def _ids(listing_ids: pd.Series) -> pd.Index:
    ids = pd.Index(listing_ids, name="listing_id")
    if ids.dtype != np.dtype("int64") or not ids.is_unique or len(ids) < 10:
        raise ValueError("Split cần ít nhất 10 listing_id kiểu int64 và duy nhất.")
    return ids


def assign_production_splits(
    listing_ids: pd.Series, random_state: int = RANDOM_STATE
) -> pd.Series:
    """Chia theo ID khi mỗi xe nhận diện đã được đại diện bởi một tin."""
    ids = _ids(listing_ids)
    train, other = train_test_split(
        ids.sort_values(), test_size=0.30, random_state=random_state
    )
    val, test = train_test_split(other, test_size=0.50, random_state=random_state + 1)
    splits = pd.Series(index=ids, dtype="string")
    for name, keys in (("train", train), ("val", val), ("test", test)):
        splits.loc[keys] = name
    return pd.Series(
        splits.to_numpy(), index=listing_ids.index, dtype="string", name="split_set"
    )


def _group_ids(values: pd.Series) -> pd.Series:
    strings = values.astype("string")
    if strings.isna().any() or strings.str.strip().eq("").any():
        raise ValueError(
            "vehicle_group_id phải đầy đủ. Xe chưa nhận diện dùng nhóm riêng theo ID."
        )
    return strings


def _inherit_base_splits(
    frame: pd.DataFrame, base_manifest: pd.DataFrame, random_state: int
) -> pd.Series:
    """Kế thừa split_set đã khóa từ manifest phiên bản trước để tránh trôi tập test."""
    if base_manifest["split_set"].isna().any() or not set(
        base_manifest["split_set"]
    ).issubset(PARTITIONS):
        raise ValueError("Manifest gốc có nhãn split_set không hợp lệ.")
    base_by_id = (
        base_manifest.drop_duplicates("listing_id")
        .set_index("listing_id")["split_set"]
        .astype("string")
    )
    inherited = frame["listing_id"].map(base_by_id).astype("string")

    if GROUP_COLUMN in frame:
        groups = _group_ids(frame[GROUP_COLUMN])
        if GROUP_COLUMN in base_manifest:
            base_groups = _group_ids(base_manifest[GROUP_COLUMN])
            if base_manifest.groupby(base_groups)["split_set"].nunique().gt(1).any():
                raise ValueError("Manifest gốc có nhóm xe giao nhiều tập.")
            base_by_group = (
                base_manifest.assign(_g=base_groups)
                .drop_duplicates("_g")
                .set_index("_g")["split_set"]
                .astype("string")
            )
            inherited = inherited.fillna(groups.map(base_by_group).astype("string"))
        # Bảo đảm mọi tin cùng nhóm xe nhận đúng một split duy nhất.
        known = inherited.notna()
        if known.any():
            per_group = inherited[known].groupby(groups[known]).nunique()
            if per_group.gt(1).any():
                raise ValueError(
                    "Một nhóm xe ở phiên bản mới giao nhiều tập của manifest gốc."
                )
            group_map = inherited[known].groupby(groups[known]).first()
            inherited = inherited.fillna(groups.map(group_map).astype("string"))

        missing_mask = inherited.isna()
        if missing_mask.any():
            missing_frame = frame.loc[missing_mask]
            missing_groups = groups.loc[missing_mask]
            if missing_groups.nunique() >= 10:
                new_splits = make_split_manifest(
                    missing_frame, random_state=random_state
                )["split_set"]
                inherited.loc[missing_mask] = new_splits.to_numpy()
            else:
                inherited.loc[missing_mask] = "train"
    else:
        missing_mask = inherited.isna()
        if missing_mask.any():
            missing_ids = frame.loc[missing_mask, "listing_id"]
            if len(missing_ids) >= 10:
                inherited.loc[missing_mask] = assign_production_splits(
                    missing_ids, random_state
                ).to_numpy()
            else:
                inherited.loc[missing_mask] = "train"
    return inherited.astype("string")


def make_split_manifest(
    listings: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    *,
    base_manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Chia không dùng giá. Khi có nhóm, tỷ lệ xấp xỉ 70/15/15 tính theo số nhóm."""
    _ids(listings["listing_id"])
    columns = ["listing_id", *([GROUP_COLUMN] if GROUP_COLUMN in listings else [])]
    frame = listings[columns].sort_values("listing_id").reset_index(drop=True).copy()
    if base_manifest is not None:
        inherited = _inherit_base_splits(frame, base_manifest, random_state)
        if set(inherited) != PARTITIONS:
            raise ValueError("Manifest kế thừa phải giữ đủ cả ba tập train/val/test.")
        result = frame.assign(split_set=inherited)
        if GROUP_COLUMN in result:
            result[GROUP_COLUMN] = _group_ids(result[GROUP_COLUMN])
        return result
    if GROUP_COLUMN not in frame:
        return frame.assign(
            split_set=assign_production_splits(frame["listing_id"], random_state)
        )

    groups = _group_ids(frame[GROUP_COLUMN])
    if groups.nunique() < 10:
        raise ValueError("Cần tối thiểu 10 nhóm xe để chia train/val/test.")
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train, rest = next(outer.split(frame, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state + 1)
    val, test = next(inner.split(frame.iloc[rest], groups=groups.iloc[rest]))
    split = pd.Series(index=frame.index, dtype="string")
    split.iloc[train] = "train"
    split.iloc[rest[val]] = "val"
    split.iloc[rest[test]] = "test"
    return frame.assign(vehicle_group_id=groups, split_set=split)


def validate_split_manifest(
    manifest: pd.DataFrame, listing_ids: pd.Series
) -> pd.DataFrame:
    """Kiểm định phủ đúng tập ID và trả manifest theo thứ tự ID của ABT."""
    expected, actual = _ids(listing_ids), _ids(manifest["listing_id"])
    if len(expected) != len(actual) or not expected.isin(actual).all():
        raise ValueError("Manifest phải phủ đúng toàn bộ listing_id của ABT.")
    if manifest["split_set"].isna().any() or set(manifest["split_set"]) != PARTITIONS:
        raise ValueError("Manifest phải có đủ train/val/test và không có nhãn khác.")
    columns = ["listing_id", "split_set"]
    if GROUP_COLUMN in manifest:
        groups = _group_ids(manifest[GROUP_COLUMN])
        if manifest.groupby(groups)["split_set"].nunique().gt(1).any():
            raise ValueError(
                "Một nhóm xe xuất hiện ở nhiều tập. Hãy khóa lại manifest."
            )
        columns.append(GROUP_COLUMN)
    return manifest[columns].set_index("listing_id").loc[expected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--listings", type=Path, help="Danh sách ID hoặc nhóm đã khóa.")
    inputs.add_argument("--vin-listings", type=Path, help="Metadata VIN nội bộ trước dedup.")
    parser.add_argument(
        "--base-manifest",
        type=Path,
        help="Manifest phiên bản trước để kế thừa split_set cố định và tránh trôi test.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Manifest đã tồn tại. Chọn đường dẫn phiên bản mới.")
    base_manifest = (
        pd.read_parquet(args.base_manifest, engine="pyarrow")
        if args.base_manifest is not None
        else None
    )
    audit = {}
    if args.vin_listings is not None:
        data = pd.read_parquet(
            args.vin_listings, engine="pyarrow", columns=["listing_id", "VIN", "posting_date"]
        )
        decisions = resolve_vin_reposts(data)
        kept = decisions.loc[decisions["keep_for_abt"]]
        audit = {
            "rows_input": len(decisions),
            "rows_kept": len(kept),
            "rows_removed": len(decisions) - len(kept),
            "rows_by_vin_status": decisions["vin_status"].value_counts().to_dict(),
            "retained_multi_listing_groups": int(
                kept.groupby(GROUP_COLUMN).size().gt(1).sum()
            ),
        }
        data = kept
    else:
        data = pd.read_parquet(args.listings, engine="pyarrow")

    manifest = make_split_manifest(data, base_manifest=base_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as target:
        manifest.to_parquet(target, engine="pyarrow", index=False)
    audit["rows_by_split"] = manifest["split_set"].value_counts().to_dict()
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
