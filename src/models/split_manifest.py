"""Tạo và kiểm định bảng chia cố định do TV3 bàn giao."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from configs.project_config import RANDOM_STATE

PARTITIONS = {"train", "val", "test"}
GROUP_COLUMN = "vehicle_group_id"


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


def make_split_manifest(
    listings: pd.DataFrame, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Chia không dùng giá. Khi có nhóm, tỷ lệ xấp xỉ 70/15/15 tính theo số nhóm."""
    _ids(listings["listing_id"])
    columns = ["listing_id", *([GROUP_COLUMN] if GROUP_COLUMN in listings else [])]
    frame = listings[columns].sort_values("listing_id").reset_index(drop=True).copy()
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
    parser.add_argument("--listings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Manifest đã tồn tại. Chọn đường dẫn phiên bản mới.")
    data = pd.read_parquet(args.listings, engine="pyarrow")
    manifest = make_split_manifest(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.output, engine="pyarrow", index=False)
    print(manifest["split_set"].value_counts().to_string())


if __name__ == "__main__":
    main()
