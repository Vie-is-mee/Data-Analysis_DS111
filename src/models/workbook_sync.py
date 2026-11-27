"""Xuất riêng các sheet kỹ thuật của TV3."""

import math
import os
import textwrap
import warnings
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from src.models.model_registry import ModelRegistry

TRACKING_SHEETS = (
    "08_Experiment_Tracking",
    "08b_VIF_Feature_Ranking",
    "08c_Metrics_Explanation",
)

_METRICS = [
    (
        "R²",
        "1 − SS_res / SS_tot. SS_res là tổng bình phương phần dư. SS_tot đo độ lệch quanh trung bình y của tập đang đánh giá.",
    ),
    (
        "Adjusted R²",
        "1 − (1 − R²)(n − 1)/(n − p − 1). Ô trống khi nhãn hằng hoặc n ≤ p + 1. Với cây, p là số cột đầu vào chứ không đại diện bậc tự do của mô hình.",
    ),
    (
        "MSE / RMSE / MAE",
        "MSE = SS_res / n; RMSE = √MSE (cùng đơn vị USD, phạt mạnh sai số lớn ở đuôi); MAE = trung bình |sai số|. Các cột train/val/test chính tính bằng USD.",
    ),
    (
        "Kiểm tra Underfit / Overfit",
        "So train_r2 với cv_r2_mean (train_cv_r2_gap) và train_rmse với cv_rmse_mean (cv_train_rmse_gap). Cả hai cùng kém là Underfit; train cao vượt trội còn CV thấp là Overfit.",
    ),
    ("MAPE (%)", "Trung bình |phần dư / giá thực| × 100 trên nhãn khác 0."),
    (
        "Thang log",
        "Cột *_log_* đo trực tiếp trên log tự nhiên trước nghịch đảo. MAPE theo log không diễn giải bằng phần trăm USD.",
    ),
    (
        "Hiệu chỉnh Duan (toàn cục & phân vị)",
        "Dự báo USD = exp(dự báo log) × hệ số Duan từ phần dư OOF trên train. Nhánh Exp5 dùng Duan theo 10 phân vị dự báo để xử lý phương sai không đồng nhất giữa xe phổ thông và xe cổ.",
    ),
    (
        "Exp5 — Phần dư hao mòn phân cấp",
        "Khớp đường hao mòn cơ sở (tuổi gãy khúc 15/25, số dặm, xi-lanh) trong từng fold train rồi co mượt Bayes phân cấp 4 tầng (Hãng → Hãng|Kiểu → Hãng|Gốc dòng xe → Hãng|Dòng) theo GroupKFold; so trực tiếp với Target Encoding phẳng.",
    ),
    (
        "Tuyến tính theo tham số & Đa thức",
        "ln(P) = ln(P₀) − k × car_age − β × ln(1 + odometer). Bổ sung car_age² (hồi quy đa thức bậc 2 theo tuổi xe) vẫn tuyến tính theo hệ số và bắt được vùng xe cổ tăng giá.",
    ),
    (
        "VIF và ranking",
        "VIF > 5 cảnh báo, > 10 nghiêm trọng. OLS/Ridge dùng cùng mẫu train và cùng biến đã chuẩn hóa. Dấu hệ số mô tả liên hệ có điều kiện.",
    ),
    (
        "Chọn mô hình",
        "Xếp theo RMSE USD trung bình của CV trên train. Validation phá hòa. Chỉ mô hình đã chốt có số đo test.",
    ),
    (
        "Phiên bản dữ liệu",
        "Kết quả mang tiền tố mock chỉ dùng kiểm thử. Manifests chia tập và dataset_version xác định lần chạy.",
    ),
]


def _write_table(sheet, frame, start=1, *, max_width=35):
    headers = list(frame.columns)
    for row_index, values in enumerate(
        (headers, *frame.itertuples(index=False, name=None)), start
    ):
        for col_index, value in enumerate(values, 1):
            cell = sheet.cell(row_index, col_index)
            if isinstance(value, (float, np.floating)) and math.isinf(value):
                value = "∞" if value > 0 else "−∞"
            cell.value = None if pd.isna(value) else value
            if isinstance(value, str):
                cell.data_type = "s"
            if row_index == start:
                cell.fill = PatternFill("solid", fgColor="17425F")
                cell.font = Font(color="FFFFFF", bold=True)
            elif isinstance(value, (float, np.floating)):
                cell.number_format = "#,##0.0000" if abs(value) < 100 else "#,##0.00"
            cell.alignment = Alignment(vertical="center", wrap_text=True)

    for index, column in enumerate(headers, 1):
        letter = get_column_letter(index)
        lengths = [
            len(column),
            *(len(str(value)) for value in frame[column] if pd.notna(value)),
        ]
        width = min(max(max(lengths) + 3, 12), max_width)
        sheet.column_dimensions[letter].width = max(
            sheet.column_dimensions[letter].width, width
        )
    for row in sheet.iter_rows(
        min_row=start, max_row=start + len(frame), max_col=len(headers)
    ):
        lines = max(
            len(
                textwrap.wrap(
                    str(cell.value),
                    max(8, int(sheet.column_dimensions[cell.column_letter].width) - 3),
                )
            )
            if cell.value is not None
            else 1
            for cell in row
        )
        sheet.row_dimensions[row[0].row].height = max(20, 15 * lines + 5)


def export_experiment_tracking_to_excel(
    registry: ModelRegistry,
    vif_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    workbook_path: Path,
) -> Path:
    path = Path(workbook_path)
    if path.exists():
        # Nếu openpyxl phát hiện thành phần mà nó sẽ loại bỏ, giữ file nguồn.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=UserWarning, module="openpyxl")
            book = load_workbook(path)
    else:
        book = Workbook()
        book.remove(book.active)
    for name in TRACKING_SHEETS:
        position = book.index(book[name]) if name in book else len(book.sheetnames)
        if name in book:
            book.remove(book[name])
        book.create_sheet(name, position)
    summary, diagnostics, explanation = (book[name] for name in TRACKING_SHEETS)

    results = registry.to_summary_dataframe()
    _write_table(summary, results, max_width=46)
    summary.auto_filter.ref = summary.dimensions
    summary.freeze_panes = "B2"
    _write_table(diagnostics, vif_df)
    _write_table(diagnostics, ranking_df, start=len(vif_df) + 4)
    diagnostics.freeze_panes = "B2"
    _write_table(
        explanation,
        pd.DataFrame(_METRICS, columns=["Chỉ số", "Cách đọc"]),
        max_width=85,
    )
    explanation.freeze_panes = "A2"

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{uuid4().hex}.xlsx")
    try:
        book.save(temp)
        os.replace(temp, path)
    finally:
        book.close()
        temp.unlink(missing_ok=True)
    return path
