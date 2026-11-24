"""Điểm chạy baseline DS111 trên dữ liệu mồi hoặc dữ liệu TV1 bàn giao."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threadpoolctl import threadpool_limits

from configs.project_config import CONFIG, ProjectConfig
from src.core.bootstrap_and_validate import (
    generate_mock_clean_sample,
    validate_handoff_dataset,
)
from src.models.ablation_runner import LINEARIZED_NUMERIC, AblationExperimentSuite
from src.models.feature_ranking import (
    compute_vif_table,
    rank_features_on_common_support,
)
from src.models.workbook_sync import export_experiment_tracking_to_excel


def main(argv=None, *, config: ProjectConfig = CONFIG) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "production"), default="mock")
    parser.add_argument(
        "--fast", action="store_true", help="Giảm số cây, giữ nguyên các split và fold."
    )
    parser.add_argument(
        "--split-manifest", type=Path, help="Manifest đã khóa cho production."
    )
    args = parser.parse_args(argv)
    start = perf_counter()

    mock = args.mode == "mock"
    path = config.mock_parquet if mock else config.clean_parquet
    print(f"[1/6] Chuẩn bị đầu vào {args.mode} từ {path}")
    if mock and not path.exists():
        generate_mock_clean_sample(config.raw_csv, path, config.mock_sample_rows)

    print("[2/6] Kiểm định hợp đồng dữ liệu")
    if not validate_handoff_dataset(path).passed:
        return 1
    suite = AblationExperimentSuite.from_parquet(
        path,
        mock=mock,
        fast_mode=args.fast,
        config=config,
        split_manifest=args.split_manifest,
    )
    assignments = suite.split_assignments
    counts = assignments["split_set"].value_counts().to_dict()
    print(f"Phiên bản {suite.dataset_version}. Số dòng từng tập: {counts}")
    if suite.raw_exp0_features is None:
        print("Exp0 dùng proxy raw-minimal từ mẫu đã lọc.")

    print("[3/6] Tính VIF và ranking trên train")
    train = suite.train_data
    with threadpool_limits(limits=1, user_api="blas"):
        vif = compute_vif_table(train, ["year", "posting_year", *LINEARIZED_NUMERIC])
        ranking = rank_features_on_common_support(train, list(LINEARIZED_NUMERIC))

    print("[4/6] Chạy Exp0–Exp4 và khóa mô hình theo CV USD")
    registry = suite.run_all()
    winner = suite.evaluate_final()

    print("[5/6] In tổng hợp và lưu bảng kết quả")
    config.ensure_output_directories()
    summary = registry.to_summary_dataframe()
    columns = ["model_id", "cv_rmse_mean", "val_rmse", "val_r2", "test_rmse", "test_r2"]
    print(summary[columns].round(4).to_string(index=False, na_rep="–"))
    summary.to_csv(
        config.models_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig"
    )
    split_output = config.models_dir / "split_assignments.parquet"
    if (
        args.split_manifest is None
        or args.split_manifest.resolve() != split_output.resolve()
    ):
        assignments.to_parquet(split_output, engine="pyarrow", index=False)

    print("[6/6] Xuất dự báo test và Workbook TV3")
    test = suite.test_data
    registry.export_predictions_for_dashboard(
        winner.model_id, test, config.predictions_parquet
    )
    workbook = config.workbook_xlsx.with_name("Project_Workbook_M3_Tracking.xlsx")
    export_experiment_tracking_to_excel(registry, vif, ranking, workbook)
    print(
        f"[PASS] {winner.model_id}, {len(test):,} dự báo test. Hoàn thành trong {perf_counter() - start:.2f} giây."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
