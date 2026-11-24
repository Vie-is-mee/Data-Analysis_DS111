"""M1-01 / M1-12 / M1-13 / M1-14 · Xuất Project_Workbook.xlsx (Sheet 00–07).

Workbook trình bày theo CRISP-DM (Bài 1): Sheet 01 = Data Understanding,
Sheet 02–06 = Data Preparation, file parquet bàn giao = ABT.

- Chỉ thay các sheet của TV1 (00–07); giữ nguyên các sheet 08* do TV3 ghi
  bằng ``src.models.workbook_sync`` vào cùng file.
- Nguồn: parquet sạch, FormulaStore, log biến đổi, hồ sơ cột, CSV gốc (Sheet 07).

Chạy từ gốc repo (sau ``python -m src.data.pipeline_m1``):
    python -X utf8 -m src.data.workbook_exporter
"""

from __future__ import annotations

import argparse
import os
import warnings
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy import stats

from configs.project_config import (
    CONFIG,
    HIGH_CARDINALITY_CATEGORICAL,
    LOW_CARDINALITY_CATEGORICAL,
    REQUIRED_CLEAN_SCHEMA,
)
from src.core.formula_store import FormulaStore
from src.data import formatter, selector
from src.data.pipeline_m1 import DEFAULT_LOG, DEFAULT_STORE

M1_SHEETS = (
    "00_Log", "01_Raw_Profile", "02_Missing", "03_Format",
    "04_Features", "05_Encoding", "06_Selection", "07_Sample_Comparison",
)
M3_PLACEHOLDER = "08_Experiment_Tracking"
PROFILE_CSV = CONFIG.root / "workbook" / "01_Raw_Profile.csv"
SAMPLE_SIZE = 300

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(bold=True, color="FFFFFF")
TITLE_FONT = Font(bold=True, size=13)

# ── Nội dung mô tả (lý do theo slide môn học) ─────────────────────────────────
MISSING_METHODS: list[tuple[str, str, str, str]] = [
    # (cột gốc, cách xử lý, lý do, căn cứ)
    ("price", "Xóa dòng", "Biến mục tiêu, không được điền", "Bài 2 · Xóa dòng"),
    ("year", "Xóa dòng", "Không tính được car_age", "Bài 2 · Xóa dòng"),
    ("odometer", "Xóa dòng; xóa cả dòng đồng hồ hỏng/quay vòng",
     "Điền theo nhóm là học từ dữ liệu trước khi chia tập; số dặm tự báo sai không tin cậy",
     "Bài 2 · Xóa dòng"),
    ("manufacturer", "Cứu từ title, còn lại → 'unknown'",
     "Thiếu không ngẫu nhiên: hãng ngoài danh sách 42 hãng của Craigslist",
     "Bài 2 · Tạo đặc trưng từ chuỗi + giá trị cố định"),
    ("condition", "→ 'unknown' (nhãn riêng)",
     "Thiếu 46%: điền mode sẽ bóp méo phân phối; 'unknown' = Missingness Indicator",
     "Bài 2 · Giá trị cố định / Missingness Indicator"),
    ("cylinders", "Xe điện → 0; còn lại median theo (hãng, dòng xe) → hãng → toàn bộ",
     "Hợp đồng yêu cầu số hữu hạn; median bền với ngoại lệ", "Bài 2 · Median"),
    ("title_status", "→ 'unknown'", "Nhãn pháp lý, không suy đoán được", "Bài 2 · Giá trị cố định"),
    ("fuel / transmission / drive / type", "→ 'unknown'", "Biến phân loại, giữ dòng",
     "Bài 2 · Giá trị cố định"),
]

ENCODING_PLAN: dict[str, tuple[str, str, str]] = {
    # cột: (loại biến, mã hóa đề xuất, ghi chú)
    "condition_clean": ("Định tính có thứ tự", "Ordinal (salvage=0 … new=5)", "'unknown' xử lý riêng"),
    "title_status_clean": ("Định tính (thứ tự chỉ là giả thuyết)", "One-Hot; đối chứng Ordinal",
                           "Không kết luận Ordinal trước khi có số liệu (Exp 3)"),
    "year_era": ("Định tính có thứ tự", "Ordinal", "Classic < Old < Mid < Modern"),
}
for _c in LOW_CARDINALITY_CATEGORICAL:
    ENCODING_PLAN[_c] = ("Định tính không thứ tự, ít nhãn", "One-Hot", "")
for _c in HIGH_CARDINALITY_CATEGORICAL:
    ENCODING_PLAN[_c] = ("Định tính không thứ tự, nhiều nhãn", "Target Encoding theo fold (TV3)",
                         "Ngoài slide (slide dạy Label/One-Hot/Ordinal) — cần giải thích khi bảo vệ")

COLUMN_ROLES: dict[str, str] = {
    "listing_id": "Khóa", "price": "Mục tiêu (USD)", "log_price": "Mục tiêu (log)",
    "split_set": "Chia tập (TV3)", "year_era": "Nhóm phân tích (TV2)",
    "model_clean": "Mô tả (quá nhiều nhãn)", "region": "Mô tả", "posting_year": "Trung gian",
}
NUMERIC_FEATURES = ["car_age", "year", "odometer", "log_odometer", "miles_per_year", "cylinders_num"]

# Cặp (cột gốc, cột sạch) cho Sheet 07
COMPARE_PAIRS = [
    ("price", "price"), ("year", "year"), ("odometer", "odometer"),
    ("manufacturer", "manufacturer_clean"), ("model", "model_clean"),
    ("condition", "condition_clean"), ("cylinders", "cylinders_num"),
    ("title_status", "title_status_clean"), ("fuel", "fuel"), ("type", "type"),
    ("drive", "drive"), ("state", "state"),
]


# ── Ghi bảng ─────────────────────────────────────────────────────────────────
def _write_title(ws, row: int, text: str) -> int:
    ws.cell(row, 1, text).font = TITLE_FONT
    return row + 1


def _write_table(ws, df: pd.DataFrame, start_row: int = 1, max_width: int = 60) -> int:
    """Ghi DataFrame có tiêu đề định dạng; trả về dòng trống kế tiếp."""
    for j, col in enumerate(df.columns, 1):
        cell = ws.cell(start_row, j, str(col))
        cell.fill, cell.font = HEADER_FILL, HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for i, row in enumerate(df.itertuples(index=False), start_row + 1):
        for j, value in enumerate(row, 1):
            if isinstance(value, (np.generic,)):
                value = value.item()
            if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
                value = None
            elif isinstance(value, float) and np.isinf(value):
                value = "∞"  # Excel không lưu được inf (vd VIF khi cộng tuyến hoàn hảo)
            ws.cell(i, j, value)
    for j, col in enumerate(df.columns, 1):
        letter = get_column_letter(j)
        sample = [str(col)] + [str(v) for v in df.iloc[:200, j - 1].tolist()]
        width = min(max_width, max(10, max(len(s) for s in sample) + 2))
        ws.column_dimensions[letter].width = max(ws.column_dimensions[letter].width or 0, width)
    return start_row + len(df) + 2


# ── Nội dung từng sheet ──────────────────────────────────────────────────────
def build_sheet00(log: pd.DataFrame, clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = int(log["rows_before"].iloc[0])
    meta = pd.DataFrame([
        ("Ngày xuất", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Phiên bản dữ liệu", CONFIG.clean_parquet.name),
        ("Số dòng gốc", raw_rows),
        ("Số dòng sạch (ABT)", len(clean)),
        ("Tỷ lệ giữ", f"{len(clean) / raw_rows:.1%}"),
        ("Số cột bàn giao", clean.shape[1]),
        ("Quy trình", "CRISP-DM: Sheet 01 = Data Understanding; 02–06 = Data Preparation; parquet = ABT"),
    ], columns=["Mục", "Giá trị"])
    return meta, log


def build_sheet02(log: pd.DataFrame, profile: pd.DataFrame | None) -> pd.DataFrame:
    pct = {} if profile is None else dict(zip(profile["column"], profile["missing_pct"]))
    rows = []
    for col, method, reason, ref in MISSING_METHODS:
        first = col.split(" / ")[0]
        rows.append({"Cột gốc": col, "% khuyết (gốc)": pct.get(first), "Cách xử lý": method,
                     "Lý do": reason, "Căn cứ (slide)": ref})
    return pd.DataFrame(rows)


def build_sheet05(clean: pd.DataFrame) -> pd.DataFrame:
    rows = [{"Cột": c, "Số nhãn": int(clean[c].nunique()), "Loại biến": kind,
             "Mã hóa đề xuất": enc, "Ghi chú": note, "Thực hiện": "TV3 (trong từng fold)"}
            for c, (kind, enc, note) in ENCODING_PLAN.items() if c in clean.columns]
    return pd.DataFrame(rows)


def build_sheet06(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dictionary = pd.DataFrame([
        {"Cột": c, "Kiểu": t, "Vai trò": COLUMN_ROLES.get(c, "Đặc trưng"),
         "Số giá trị khác nhau": int(clean[c].nunique())}
        for c, t in REQUIRED_CLEAN_SCHEMA.items()
    ])
    cols = NUMERIC_FEATURES + ["log_price"]
    target_corr = []
    for c in NUMERIC_FEATURES:
        r, p_r = stats.pearsonr(clean[c], clean["log_price"])
        rho, p_s = stats.spearmanr(clean[c], clean["log_price"])
        target_corr.append({"Biến": c, "Pearson r": round(r, 4), "p (Pearson)": p_r,
                            "Spearman ρ": round(rho, 4), "p (Spearman)": p_s, "n": len(clean)})
    matrix = clean[cols].corr(method="pearson").round(3).reset_index().rename(columns={"index": "Biến"})
    vif = selector.vif_report(clean)
    return dictionary, pd.DataFrame(target_corr), matrix, vif


def build_sheet07(clean: pd.DataFrame, raw: pd.DataFrame | None, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Đối chiếu 1–1 theo listing_id cố định (seed 42) trước/sau làm sạch."""
    ids = clean["listing_id"].sort_values().sample(min(n, len(clean)), random_state=CONFIG.random_state)
    after = clean.set_index("listing_id").loc[ids]
    out = pd.DataFrame({"listing_id": ids.to_numpy()})
    if raw is None:
        return out.assign(ghi_chu="Không có CSV gốc để đối chiếu")
    before = raw.assign(listing_id=pd.to_numeric(raw["id"], errors="coerce")).dropna(subset=["listing_id"])
    before = before.astype({"listing_id": "int64"}).drop_duplicates("listing_id").set_index("listing_id")
    before = before.reindex(ids)
    changed = pd.Series([[] for _ in range(len(ids))])
    for raw_col, clean_col in COMPARE_PAIRS:
        b = before[raw_col].reset_index(drop=True) if raw_col in before else pd.Series([None] * len(ids))
        a = after[clean_col].reset_index(drop=True)
        out[f"{raw_col} (gốc)"] = b.astype("object").where(b.notna(), None)
        out[f"{clean_col} (sạch)"] = a.astype("object")
        # cylinders: '6 cylinders' → 6 chỉ là đổi định dạng; so sánh sau khi tách số
        # để chỉ đánh dấu khi giá trị thực sự đổi (ví dụ khuyết → điền median).
        b_cmp = _as_text(formatter.parse_cylinders(b)) if raw_col == "cylinders" else \
            b.astype("string").fillna("").str.strip().str.lower()
        diff = b_cmp != _as_text(a)
        for i in np.flatnonzero(diff.to_numpy()):
            changed[i].append(raw_col)
    out.insert(1, "Trường đã thay đổi", changed.map(", ".join).replace("", "—"))
    return out


def _as_text(s: pd.Series) -> pd.Series:
    """Chuẩn để so sánh: số nguyên không có '.0', chuỗi chữ thường."""
    if pd.api.types.is_numeric_dtype(s):
        s = s.map(lambda v: f"{v:g}" if pd.notna(v) else "")
    return s.astype("string").fillna("").str.strip().str.lower()


# ── Ghi file ─────────────────────────────────────────────────────────────────
def export_workbook(clean: pd.DataFrame, log: pd.DataFrame, store: FormulaStore,
                    profile: pd.DataFrame | None, raw: pd.DataFrame | None, path: Path) -> Path:
    path = Path(path)
    if path.exists():
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=UserWarning, module="openpyxl")
            book = load_workbook(path)
    else:
        book = Workbook()
        book.remove(book.active)
    for i, name in enumerate(M1_SHEETS):
        if name in book:
            book.remove(book[name])
        book.create_sheet(name, i)
    if not any(n.startswith("08") for n in book.sheetnames):
        ws = book.create_sheet(M3_PLACEHOLDER)
        ws["A1"] = "TV3 ghi sheet này bằng src.models.workbook_sync (Exp 0–4)."

    meta, full_log = build_sheet00(log, clean)
    ws = book["00_Log"]
    r = _write_title(ws, 1, "Nhật ký biến đổi dữ liệu TV1")
    r = _write_table(ws, meta, r)
    _write_table(ws, full_log, r)

    ws = book["01_Raw_Profile"]
    _write_table(ws, profile if profile is not None
                 else pd.DataFrame({"ghi_chu": ["Chạy src.data.raw_profile để tạo hồ sơ cột"]}))

    ws = book["02_Missing"]
    r = _write_title(ws, 1, "Cách xử lý dữ liệu khuyết theo cột")
    r = _write_table(ws, build_sheet02(log, profile), r)
    r = _write_title(ws, r, "Nhật ký các bước (số dòng trước/sau)")
    _write_table(ws, log[log["sheet"] == "02_Missing"], r)

    ws = book["03_Format"]
    r = _write_title(ws, 1, "Chuẩn hóa, lọc miền, lọc ngoại lệ (số dòng trước/sau)")
    _write_table(ws, log[log["sheet"] == "03_Format"], r)

    ws = book["04_Features"]
    r = _write_title(ws, 1, "Công thức đặc trưng (FormulaStore) — thống kê trước/sau")
    _write_table(ws, store.to_dataframe_report(), r)

    ws = book["05_Encoding"]
    r = _write_title(ws, 1, "Kế hoạch mã hóa — TV1 bàn giao nhãn dạng chuỗi, TV3 mã hóa trong từng fold")
    r = _write_table(ws, build_sheet05(clean), r)
    ws.cell(r, 1, "Nhánh chính không dùng Label Encoding cho biến không thứ tự (Gate 1).")

    dictionary, target_corr, matrix, vif = build_sheet06(clean)
    ws = book["06_Selection"]
    r = _write_title(ws, 1, "Từ điển 22 cột bàn giao")
    r = _write_table(ws, dictionary, r)
    r = _write_title(ws, r, "Tương quan với log_price (Bài 4: Pearson/Spearman kèm p-value)")
    r = _write_table(ws, target_corr, r)
    r = _write_title(ws, r, "Ma trận tương quan Pearson (tô màu như heatmap)")
    m_start = r
    r = _write_table(ws, matrix, r)
    first, last = get_column_letter(2), get_column_letter(matrix.shape[1])
    ws.conditional_formatting.add(
        f"{first}{m_start + 1}:{last}{m_start + len(matrix)}",
        ColorScaleRule(start_type="num", start_value=-1, start_color="5B9BD5",
                       mid_type="num", mid_value=0, mid_color="FFFFFF",
                       end_type="num", end_value=1, end_color="F8696B"),
    )
    r = _write_title(ws, r, "VIF (bổ sung; > 5 cảnh báo, > 10 nghiêm trọng)")
    _write_table(ws, vif, r)

    ws = book["07_Sample_Comparison"]
    r = _write_title(ws, 1, f"Đối chiếu {SAMPLE_SIZE} listing_id cố định (seed {CONFIG.random_state}) trước/sau làm sạch")
    _write_table(ws, build_sheet07(clean, raw), r, max_width=28)

    for name in M1_SHEETS:
        book[name].freeze_panes = "A2"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{uuid4().hex}.xlsx")
    try:
        book.save(temp)
        os.replace(temp, path)
    finally:
        book.close()
        temp.unlink(missing_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, default=CONFIG.clean_parquet)
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--profile", type=Path, default=PROFILE_CSV)
    p.add_argument("--raw", type=Path, default=CONFIG.raw_csv)
    p.add_argument("--out", type=Path, default=CONFIG.workbook_xlsx)
    args = p.parse_args(argv)

    clean = pd.read_parquet(args.parquet)
    log = pd.read_csv(args.log)
    store = FormulaStore.load(args.store)
    profile = pd.read_csv(args.profile) if args.profile.is_file() else None
    raw = (pd.read_csv(args.raw, usecols=["id"] + [c for c, _ in COMPARE_PAIRS], dtype="string")
           if args.raw.is_file() else None)
    out = export_workbook(clean, log, store, profile, raw, args.out)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
