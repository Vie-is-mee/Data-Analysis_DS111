"""Điểm chạy baseline DS111 trên dữ liệu mồi hoặc dữ liệu TV1 bàn giao."""

import argparse
import sys
from dataclasses import replace
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
    parser.add_argument("--abt", type=Path, help="Đường dẫn ABT phiên bản được bàn giao.")
    parser.add_argument("--raw-csv", type=Path, help="CSV gốc tương ứng để ghép Exp0.")
    parser.add_argument("--dataset-version", help="Nhãn phiên bản của ABT production.")
    parser.add_argument("--output-root", type=Path, help="Gốc lưu kết quả của lần chạy.")
    parser.add_argument(
        "--require-raw-exp0", action="store_true", help="Dừng nếu Exp0 phải dùng proxy."
    )
    args = parser.parse_args(argv)
    mock = args.mode == "mock"
    if mock and (args.abt or args.raw_csv or args.dataset_version is not None or args.require_raw_exp0):
        parser.error("--abt, --raw-csv, --dataset-version và --require-raw-exp0 chỉ dùng cho production.")
    version = args.dataset_version.strip() if args.dataset_version is not None else None
    if version == "" or (args.abt is not None and version is None):
        parser.error("Dùng --dataset-version để ghi đúng phiên bản của ABT chỉ định bằng --abt.")
    start = perf_counter()

    path = config.mock_parquet if mock else (args.abt or config.clean_parquet)
    output_config = replace(config, root=args.output_root) if args.output_root else config
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
        raw_csv=args.raw_csv,
        dataset_version=version,
    )
    assignments = suite.split_assignments
    counts = assignments["split_set"].value_counts().to_dict()
    print(f"Phiên bản {suite.dataset_version}. Số dòng từng tập: {counts}")
    if suite.raw_exp0_features is None:
        if args.require_raw_exp0:
            print("Không đủ dữ liệu CSV gốc để ghép Exp0 theo listing_id.", file=sys.stderr)
            return 1
        print("Exp0 dùng proxy raw-minimal từ mẫu đã lọc.")

    print("[3/6] Tính VIF và ranking trên train")
    train = suite.train_data
    if not mock and "cylinders_was_missing" in train:
        train = train.loc[~train["cylinders_was_missing"]]
    with threadpool_limits(limits=1, user_api="blas"):
        vif = compute_vif_table(train, ["year", "posting_year", *LINEARIZED_NUMERIC])
        ranking = rank_features_on_common_support(train, list(LINEARIZED_NUMERIC))

    print("[4/6] Chạy Exp0–Exp4 và khóa mô hình theo CV USD")
    registry = suite.run_all()
    winner = suite.evaluate_final()

    print("[5/6] In tổng hợp và lưu bảng kết quả")
    output_config.ensure_output_directories()
    summary = registry.to_summary_dataframe()
    columns = [
        "model_id",
        "train_r2",
        "cv_r2_mean",
        "cv_rmse_mean",
        "val_rmse",
        "test_rmse",
        "test_r2",
    ]
    print(summary[columns].round(4).to_string(index=False, na_rep="–"))
    summary.to_csv(
        output_config.models_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig"
    )
    split_output = output_config.models_dir / "split_assignments.parquet"
    if (
        args.split_manifest is None
        or args.split_manifest.resolve() != split_output.resolve()
    ):
        assignments.to_parquet(split_output, engine="pyarrow", index=False)

    print("[6/6] Xuất dự báo test và Workbook TV3")
    test = suite.test_data
    registry.export_predictions_for_dashboard(
        winner.model_id, test, output_config.predictions_parquet
    )
    workbook = output_config.workbook_xlsx.with_name("Project_Workbook_M3_Tracking.xlsx")
    export_experiment_tracking_to_excel(registry, vif, ranking, workbook)
    print(
        f"[PASS] {winner.model_id}, {len(test):,} dự báo test. Hoàn thành trong {perf_counter() - start:.2f} giây."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
