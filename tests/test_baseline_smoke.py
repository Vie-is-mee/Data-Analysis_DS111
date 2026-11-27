"""Kiểm chứng DS111 bằng Parquet, huấn luyện sklearn/LightGBM và Excel thật."""

from collections import Counter
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from openpyxl.worksheet.datavalidation import DataValidation
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from configs.project_config import REQUIRED_CLEAN_SCHEMA, ProjectConfig
from scripts.run_m3_baseline import main as run_baseline
from src.core import FormulaMetadata, FormulaStore
from src.core.bootstrap_and_validate import (
    generate_mock_clean_sample,
    main,
    validate_handoff_dataset,
)
from src.models.ablation_runner import (
    AblationExperimentSuite,
    assign_production_splits,
    read_raw_exp0_features,
)
from src.models.feature_ranking import (
    compute_vif_table,
    rank_features_on_common_support,
)
from src.models.metrics import (
    compute_duan_smearing_factor,
    evaluate_regression,
    inverse_log_transform,
)
from src.models.preprocessing_bridge import (
    OutOfFoldTargetEncoder,
    build_feature_preprocessor,
)
from src.models.workbook_sync import (
    TRACKING_SHEETS,
    export_experiment_tracking_to_excel,
)

PREDICTION_COLUMNS = {
    "listing_id": "int64",
    "split_set": "string",
    "experiment_id": "string",
    "model_name": "string",
    "actual_price_usd": "float64",
    "predicted_price_usd": "float64",
    "residual_usd": "float64",
    "absolute_error_usd": "float64",
    "dataset_version": "string",
}


@pytest.fixture
def mock_data(tmp_path):
    path = tmp_path / "mock_clean_sample.parquet"
    generate_mock_clean_sample(tmp_path / "absent.csv", path, n_samples=600)
    return path, pd.read_parquet(path, engine="pyarrow")


def test_formula_store_separation(tmp_path):
    index = pd.Index([110, 205, 306], name="listing_id")
    source = pd.DataFrame({"odometer": [10.0, 50.0, 90.0]}, index=index)
    meta = FormulaMetadata(
        "log_odometer",
        "log_odometer",
        "Step3",
        "ln(1+d)",
        r"\ln(1+d)",
        ["odometer"],
        "log_odometer",
        "Khấu hao",
        ["OLS"],
    )
    store = FormulaStore()

    def transform(frame):
        result = np.log1p(frame["odometer"])
        result.attrs["formula"] = "metadata must stay separate"
        return result

    result = store.apply_and_register(source, transform, meta)
    assert result.attrs == {}
    assert all(isinstance(value, pd.Series) for value in store._series_memory.values())
    assert all(
        isinstance(value, FormulaMetadata)
        for value in store._metadata_registry.values()
    )
    assert store.get_metadata("log_odometer").stats_before["mean"] == 50.0
    with pytest.raises(ValueError, match="đã tồn tại"):
        store.register_formula(result, meta)

    result.iloc[0] = -1.0
    meta.input_columns.clear()
    path = tmp_path / "formulas.joblib"
    store.save(path)
    restored = FormulaStore.load(path)
    assert restored.get_metadata("log_odometer").input_columns == ["odometer"]
    assert restored.to_dataframe_report().shape[0] == 1
    pd.testing.assert_series_equal(
        restored.get_series("log_odometer"),
        np.log1p(source["odometer"]).rename("log_odometer"),
    )


def test_bootstrap_and_validator_pass(mock_data):
    path, data = mock_data
    store = path.with_name("mock_formula_store.joblib")
    report = validate_handoff_dataset(path, store)
    assert report.passed and report.row_count == 600
    assert main(["validate", "--path", str(path), "--store", str(store)]) == 0
    assert pq.read_metadata(path).metadata[b"ds111.mock_source"] == b"synthetic"
    assert {
        column: str(data[column].dtype) for column in REQUIRED_CLEAN_SCHEMA
    } == REQUIRED_CLEAN_SCHEMA
    assert data["split_set"].value_counts().to_dict() == {
        "train": 420,
        "val": 90,
        "test": 90,
    }
    restored = FormulaStore.load(store)
    assert restored.formula_count == 6
    pd.testing.assert_series_equal(restored.get_series("log_price"), data["log_price"])


@pytest.mark.parametrize(
    "fault", ["log_tolerance", "price_dtype", "id_duplicate", "alias", "empty_text"]
)
def test_validator_rejects_contract_errors(mock_data, fault):
    path, data = mock_data
    if fault == "log_tolerance":
        data.loc[0, "log_price"] += 2e-8
    elif fault == "price_dtype":
        data["price"] = data["price"].astype("float32")
    elif fault == "id_duplicate":
        data.loc[0, ["listing_id", "row_id"]] = data.loc[1, ["listing_id", "row_id"]]
    elif fault == "alias":
        data.loc[0, "row_id"] = -1
    else:
        data.loc[0, "fuel"] = "   "
    data.to_parquet(path, engine="pyarrow", index=False)
    assert not validate_handoff_dataset(path).passed


def test_metrics_r2_mean_baseline_and_smearing():
    actual = np.array([10.0, 20.0, 30.0])
    mean = evaluate_regression(actual, np.full(3, actual.mean()), 1)
    perfect = evaluate_regression(actual, actual, 1)
    assert mean.r2 == 0.0 and mean.ss_res == mean.ss_tot == 200.0
    assert mean.y_mean_baseline == 20.0
    assert mean.mse == pytest.approx(200 / 3)
    assert mean.rmse == pytest.approx(np.sqrt(200 / 3))
    assert perfect.r2 == 1.0 and perfect.adj_r2 == 1.0 and perfect.mse == 0.0 and perfect.mape_pct == 0.0
    assert np.isnan(evaluate_regression(actual, actual, 2).adj_r2)
    assert np.isnan(evaluate_regression(np.full(3, 100.1), np.full(3, 100.1), 1).r2)
    pred = np.full(3, 5.0)
    factors = compute_duan_smearing_factor(pred + [-1.0, 0.0, 1.0], pred)
    assert factors >= 1.0
    assert compute_duan_smearing_factor(pred - 1, pred) == pytest.approx(np.exp(-1))
    np.testing.assert_allclose(
        inverse_log_transform(pred, factors), np.exp(pred) * factors
    )


def test_target_encoder_uses_out_of_fold_and_train_only_prior():
    frame = pd.DataFrame(
        {"category": ["a", "b", "c", "d", "e", "f"]}, index=range(10, 16)
    )
    target = np.array([0.0, 1.0, 2.0, 39.0, 4.0, 105.0])
    encoder = OutOfFoldTargetEncoder(n_splits=3, smoothing=2, random_state=42)
    encoded = encoder.fit_transform(frame, target)
    for train, holdout in KFold(3, shuffle=True, random_state=42).split(frame):
        np.testing.assert_allclose(encoded[holdout, 0], target[train].mean())
    transformed = encoder.transform(pd.DataFrame({"category": ["a", "unseen"]}))
    np.testing.assert_allclose(
        transformed[:, 0], [2 * target.mean() / 3, target.mean()]
    )
    assert set(encoder.mappings_["category"].index) == set(frame["category"])

    with pytest.raises(ValueError, match="nhãn/khóa"):
        build_feature_preprocessor("numeric_only", ["log_price"], [], [], [])


def test_ranking_uses_complete_common_support_and_vif_has_intercept():
    data = pd.DataFrame(
        {
            "x": [2.0, 3.0, 9.0, -2.0, 0.0, np.nan],
            "same": [4.0, 6.0, 18.0, -4.0, 0.0, 7.0],
            "constant": 10.0,
            "log_price": [1.0, 2.0, 5.0, np.nan, 1.0, 2.0],
        }
    )
    vif = compute_vif_table(data, ["x", "same", "constant"]).set_index("feature")
    assert np.isinf(vif.loc["x", "vif"]) and np.isnan(vif.loc["constant", "vif"])
    rank = rank_features_on_common_support(data, ["x", "same", "constant"])
    assert rank["n_samples"].tolist() == [4, 4, 4]
    assert rank.loc[rank["feature"].eq("constant"), "beta_ridge"].item() == 0


def test_end_to_end_ablation_pipeline_fast(mock_data, tmp_path, monkeypatch):
    _, data = mock_data
    train_ids = set(data.loc[data["split_set"].eq("train"), "listing_id"])
    test = data.loc[data["split_set"].eq("test")].sample(frac=1, random_state=7)
    test_ids = set(test["listing_id"])
    fit, predict = Pipeline.fit, Pipeline.predict
    test_predictions = []

    def audited_fit(self, X, y, *args, **kwargs):
        if "regression" in self.named_steps:
            assert set(X.index) <= train_ids
            assert set(X.columns).isdisjoint(
                {"price", "log_price", "listing_id", "split_set"}
            )
        return fit(self, X, y, *args, **kwargs)

    def audited_predict(self, X, *args, **kwargs):
        if "regression" in self.named_steps and test_ids.intersection(X.index):
            test_predictions.append(set(X.index))
        return predict(self, X, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", audited_fit)
    monkeypatch.setattr(Pipeline, "predict", audited_predict)
    suite = AblationExperimentSuite(
        data, fast_mode=True, dataset_version="mock-synthetic"
    )
    registry = suite.run_all()
    assert not test_predictions
    entries = [registry.get(key) for key in registry.list_ids()]
    assert Counter(entry.experiment_stage.split("_", 1)[0] for entry in entries) == {
        "Exp0": 1,
        "Exp1": 1,
        "Exp2": 2,
        "Exp3A": 1,
        "Exp3B": 2,
        "Exp4": 5,
    }
    assert {
        entry.model_name
        for entry in entries
        if entry.experiment_stage == "Exp4_Benchmark"
    } == {
        "OLS",
        "Ridge",
        "DecisionTree",
        "RandomForest",
        "LightGBM",
    }
    for entry in entries:
        assert all(np.isfinite(value) for value in entry.cv_metrics_mean.values())
        assert "mse" in entry.cv_metrics_mean
        assert entry.train_metrics is not None
        assert all(
            np.isfinite(value)
            for value in asdict(entry.train_metrics).values()
            if value is not None
        )
        assert all(
            np.isfinite(value)
            for value in asdict(entry.val_metrics).values()
            if value is not None
        )
        assert entry.test_metrics is None and entry.y_pred_test is None
    for linear_id in (
        "Exp2_OLS_AgeSquared",
        "Exp3A_OLS",
        "Exp3B_OLS",
        "Exp3B_OLS_TitleOrdinal",
        "Exp4_OLS",
        "Exp4_Ridge",
    ):
        assert "car_age_squared" in registry.get(linear_id).feature_names
    summary_df = registry.to_summary_dataframe()
    assert {"train_r2", "train_mse", "train_cv_r2_gap", "cv_train_rmse_gap"}.issubset(
        summary_df.columns
    )
    winner = registry.get_best_model()
    assert winner.cv_metrics_mean["rmse"] == min(
        entry.cv_metrics_mean["rmse"] for entry in entries
    )
    assert suite.evaluate_final() is winner
    assert suite.evaluate_final() is winner
    assert test_predictions == [test_ids]
    assert sum(entry.test_metrics is not None for entry in entries) == 1

    path = registry.export_predictions_for_dashboard(
        winner.model_id, test, tmp_path / "predictions_test.parquet"
    )
    result = pd.read_parquet(path, engine="pyarrow")
    assert result["listing_id"].tolist() == test["listing_id"].tolist()
    assert not result.duplicated(["listing_id", "experiment_id", "model_name"]).any()
    assert set(result["split_set"]) == {"test"}
    assert set(result["dataset_version"]) == {"mock-synthetic"}
    assert {c: str(result[c].dtype) for c in PREDICTION_COLUMNS} == PREDICTION_COLUMNS
    assert np.isfinite(result["predicted_price_usd"]).all()
    assert result["predicted_price_usd"].gt(0).all()
    np.testing.assert_allclose(
        result["residual_usd"],
        result["actual_price_usd"] - result["predicted_price_usd"],
    )
    np.testing.assert_allclose(
        result["absolute_error_usd"], result["residual_usd"].abs()
    )
    np.testing.assert_allclose(
        result["abs_pct_error"],
        100 * result["absolute_error_usd"] / result["actual_price_usd"],
    )
    with pytest.raises(ValueError, match="chỉ được chứa test"):
        registry.export_predictions_for_dashboard(
            winner.model_id, test.assign(split_set="val"), path
        )
    saved = joblib.load(registry.save_registry(tmp_path / "registry.joblib"))
    assert saved[winner.model_id].fitted_pipeline is None
    assert saved[winner.model_id].test_metrics == winner.test_metrics

    workbook = tmp_path / "Project_Workbook.xlsx"
    book = Workbook()
    book.active.title = "00_TV1"
    for number in range(1, 8):
        book.create_sheet(f"0{number}_TV1")
    for number, sheet in enumerate(book.worksheets):
        sheet["A1"] = number
        sheet["A2"] = "=A1*2"
        sheet["A2"].font = Font(bold=True)
        sheet.merge_cells("D1:E1")
        sheet.add_data_validation(DataValidation(type="whole", sqref="B2:B4"))
    book.save(workbook)
    book.close()
    diagnostics = pd.DataFrame({"feature": ["year"], "vif": [np.inf]})
    for _ in range(2):
        export_experiment_tracking_to_excel(
            registry, diagnostics, diagnostics, workbook
        )
    with workbook.open("rb") as stream:
        book = load_workbook(stream)
    try:
        assert book.sheetnames == [f"0{i}_TV1" for i in range(8)] + list(
            TRACKING_SHEETS
        )
        for number, sheet in enumerate(book.worksheets[:8]):
            assert sheet["A1"].value == number and sheet["A2"].value == "=A1*2"
            assert sheet["A2"].font.bold and "D1:E1" in sheet.merged_cells
            assert str(sheet.data_validations.dataValidation[0].sqref) == "B2:B4"
        assert book[TRACKING_SHEETS[0]].max_row == 13
        assert book[TRACKING_SHEETS[1]]["B2"].value == "∞"
        assert book[TRACKING_SHEETS[2]].max_row >= 8
    finally:
        book.close()


def test_production_split_and_raw_path(mock_data, tmp_path):
    _, data = mock_data
    config = ProjectConfig(root=tmp_path / "production")
    config.ensure_output_directories()
    required = data[list(REQUIRED_CLEAN_SCHEMA)]
    required.to_parquet(config.clean_parquet, index=False, engine="pyarrow")
    assert run_baseline(["--mode", "production", "--fast"], config=config) == 0
    unchanged = pd.read_parquet(config.clean_parquet, engine="pyarrow")
    pd.testing.assert_frame_equal(required, unchanged)
    manifest = pd.read_parquet(
        config.models_dir / "split_assignments.parquet", engine="pyarrow"
    )
    shuffled = required.sample(frac=1, random_state=9)
    independent = shuffled.assign(
        split_set=assign_production_splits(shuffled["listing_id"])
    )
    expected = independent.set_index("listing_id")["split_set"].sort_index()
    pd.testing.assert_series_equal(
        manifest.set_index("listing_id")["split_set"], expected
    )
    assert expected.value_counts().to_dict() == {"train": 420, "val": 90, "test": 90}
    prediction = pd.read_parquet(config.predictions_parquet, engine="pyarrow")
    assert set(prediction["listing_id"]) == set(expected.index[expected.eq("test")])
    assert set(prediction["dataset_version"]) == {"v1.0"}
    assert config.workbook_xlsx.with_name("Project_Workbook_M3_Tracking.xlsx").is_file()
    assert not config.workbook_xlsx.exists()

    config.raw_csv.parent.mkdir(parents=True, exist_ok=True)
    raw = data[["listing_id", "year", "odometer"]].rename(columns={"listing_id": "id"})
    raw["odometer"] += 100
    raw.to_csv(config.raw_csv, index=False)
    features = read_raw_exp0_features(config.raw_csv, shuffled["listing_id"])
    np.testing.assert_allclose(features.loc[raw["id"], "odometer"], raw["odometer"])
    raw.iloc[1:].to_csv(config.raw_csv, index=False)
    assert read_raw_exp0_features(config.raw_csv, shuffled["listing_id"]) is None


def test_bootstrap_reads_and_deduplicates_actual_csv(mock_data, tmp_path):
    _, data = mock_data
    raw = data.rename(
        columns={
            "listing_id": "id",
            "manufacturer_clean": "manufacturer",
            "model_clean": "model",
            "condition_clean": "condition",
            "title_status_clean": "title_status",
        }
    ).assign(
        posting_date="2021-04-18T12:00:00-04:00",
        cylinders=data["cylinders_num"].astype(int).astype(str) + " cylinders",
    )
    csv = tmp_path / "vehicles.csv"
    pd.concat([raw, raw.iloc[:25]]).to_csv(csv, index=False)
    output = tmp_path / "csv_sample.parquet"
    sampled = generate_mock_clean_sample(
        csv, output, n_samples=600, use_synthetic_if_missing=False
    )
    assert sampled["listing_id"].is_unique and set(sampled["listing_id"]) == set(
        raw["id"]
    )
    assert pq.read_metadata(output).metadata[b"ds111.mock_source"] == b"csv_sample"
    assert validate_handoff_dataset(
        output, output.with_name("mock_formula_store.joblib")
    ).passed
