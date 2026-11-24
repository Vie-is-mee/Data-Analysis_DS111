"""Kiểm tra hợp đồng cho nguồn 2026 bằng Parquet giả lập."""

import pandas as pd
import pytest

from configs.project_config import ProjectConfig
from scripts.run_m3_baseline import main as run_baseline
from src.core.bootstrap_and_validate import (
    generate_mock_clean_sample,
    validate_handoff_dataset,
)
from src.models.ablation_runner import AblationExperimentSuite
from src.models.preprocessing_bridge import CylinderModeImputer
from src.models.split_manifest import (
    GROUP_COLUMN,
    make_split_manifest,
    validate_split_manifest,
)


@pytest.fixture
def synthetic(tmp_path):
    path = tmp_path / "mock.parquet"
    frame = generate_mock_clean_sample(tmp_path / "missing.csv", path, 240)
    frame.loc[0, ["year", "posting_year", "car_age", "year_era"]] = [
        2027,
        2026,
        0,
        "Modern",
    ]
    frame.loc[0, "miles_per_year"] = frame.loc[0, "odometer"]
    frame["condition_missing"] = False
    frame["seller_type"] = pd.Series("unknown", index=frame.index, dtype="string")
    return frame


def test_2027_model_year_and_optional_fields(synthetic, tmp_path):
    path = tmp_path / "abt.parquet"
    synthetic.to_parquet(path, index=False)
    assert validate_handoff_dataset(path).passed
    synthetic.loc[0, "posting_year"] = 2025
    synthetic.to_parquet(path, index=False)
    assert not validate_handoff_dataset(path).passed


def test_split_manifest_rejects_overlap_and_missing_keys(synthetic):
    frame = synthetic[["listing_id"]].assign(
        vehicle_group_id=[f"g:{i // 3}" for i in range(len(synthetic))]
    )
    manifest = make_split_manifest(frame)
    pd.testing.assert_frame_equal(
        manifest, make_split_manifest(frame.sample(frac=1, random_state=17))
    )
    assert manifest.groupby(GROUP_COLUMN)["split_set"].nunique().max() == 1
    assert len(validate_split_manifest(manifest, synthetic["listing_id"])) == len(
        synthetic
    )
    with pytest.raises(ValueError, match="phủ đúng"):
        validate_split_manifest(manifest.iloc[1:], synthetic["listing_id"])
    broken = manifest.copy()
    broken.loc[0, "split_set"] = (
        "test" if broken.loc[0, "split_set"] != "test" else "train"
    )
    with pytest.raises(ValueError, match="nhiều tập"):
        validate_split_manifest(broken, synthetic["listing_id"])


def test_cylinder_mode_learns_only_observed_train():
    cols = [
        "manufacturer_clean",
        "model_clean",
        "cylinders_num",
        "cylinders_was_missing",
    ]
    train = pd.DataFrame(
        [
            ("ford", "focus", 4.0, False),
            ("ford", "focus", 6.0, False),
            ("ford", "f150", 6.0, False),
            ("ford", "focus", 99.0, True),
        ],
        columns=cols,
    )
    holdout = pd.DataFrame(
        [
            ("ford", "focus", 100.0, True),
            ("ford", "unknown", 100.0, True),
            ("other", "car", 100.0, True),
            ("ford", "focus", 8.0, False),
        ],
        columns=cols,
    )
    assert CylinderModeImputer().fit(train).transform(holdout)[
        "cylinders_num"
    ].tolist() == [4.0, 6.0, 6.0, 8.0]


def test_production_honors_manifest_and_groups(synthetic, tmp_path):
    config = ProjectConfig(root=tmp_path)
    config.clean_parquet.parent.mkdir(parents=True, exist_ok=True)
    listing = synthetic[["listing_id"]].assign(
        vehicle_group_id=[f"g:{i // 2}" for i in range(len(synthetic))]
    )
    manifest = make_split_manifest(listing)
    path = tmp_path / "frozen.parquet"
    manifest.to_parquet(path, index=False)
    frame = synthetic.drop(columns="split_set").merge(
        manifest.drop(columns=GROUP_COLUMN), on="listing_id"
    )
    frame["cylinders_was_missing"] = frame.index % 7 == 0
    frame.to_parquet(config.clean_parquet, index=False)
    args = {"mock": False, "fast_mode": True, "config": config, "split_manifest": path}
    with pytest.raises(ValueError, match="manifest nhóm xe"):
        AblationExperimentSuite.from_parquet(
            config.clean_parquet, mock=False, fast_mode=True, config=config
        )
    ids_only = tmp_path / "ids_only.parquet"
    manifest.drop(columns=GROUP_COLUMN).to_parquet(ids_only, index=False)
    with pytest.raises(ValueError, match="vehicle_group_id"):
        AblationExperimentSuite.from_parquet(
            config.clean_parquet, **{**args, "split_manifest": ids_only}
        )
    frame.drop(columns="cylinders_was_missing").to_parquet(
        config.clean_parquet, index=False
    )
    with pytest.raises(ValueError, match="cylinders_was_missing"):
        AblationExperimentSuite.from_parquet(config.clean_parquet, **args)
    frame.to_parquet(config.clean_parquet, index=False)
    suite = AblationExperimentSuite.from_parquet(config.clean_parquet, **args)
    assert set(suite._train_groups) == set(
        manifest.loc[manifest["split_set"].eq("train"), GROUP_COLUMN]
    )
    for fit, valid in suite._cv.split(suite.train_data, groups=suite._train_groups):
        assert set(suite._train_groups.iloc[fit]).isdisjoint(
            suite._train_groups.iloc[valid]
        )
    assert (
        run_baseline(
            ["--mode", "production", "--fast", "--split-manifest", str(path)],
            config=config,
        )
        == 0
    )
    pred = pd.read_parquet(config.predictions_parquet)
    assert set(pred["listing_id"]) == set(
        manifest.loc[manifest["split_set"].eq("test"), "listing_id"]
    )
    assert GROUP_COLUMN not in pred and "cylinders_was_missing" not in pred
    frame.loc[0, "split_set"] = (
        "test" if frame.loc[0, "split_set"] != "test" else "train"
    )
    frame.to_parquet(config.clean_parquet, index=False)
    with pytest.raises(ValueError, match="chưa khớp"):
        AblationExperimentSuite.from_parquet(config.clean_parquet, **args)
